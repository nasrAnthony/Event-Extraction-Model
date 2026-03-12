import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GroupShuffleSplit, KFold
from transformers import AutoTokenizer

from helpers.utils import load_config, compute_num_stats
from helpers.dataset import merge_labels
from helpers.losses import bio_loss
from helpers.metrics import find_best_threshold_peak, boundary_metrics_peak
from helpers.train_utils import make_loader, run_epoch
from models.dom_extractor import init_model_and_optim, set_bert_trainable


# config --------------------------------------------------------------
cfg = load_config()
torch.manual_seed(cfg["training"]["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# load data and preprocess -----------------------------------------------
df = pd.read_csv("data/full_data.csv")

df = merge_labels(df, "label")
df = df.sort_values(["source", "rendering_order"]).reset_index(drop=True) # extra cleaninig

df["in_event"] = df["event_id"].notna().astype(int) # binary event flag column
m = df["event_id"].notna() # mask for events only

# take index of first row of each event (event boundary)
first_idx = df.loc[m].groupby(["source", "event_id"], sort=False).head(1).index 
df.loc[first_idx, "start_event"] = 1
df["start_event"] = df["start_event"].fillna(0).astype(int)

# outside, begin, inside tagging
df["bio"] = 0 # outside
df.loc[df["in_event"].eq(1), "bio"] = 2 # events inside
df.loc[df["start_event"].eq(1), "bio"] = 1 # boundary -> beginning

print(f"Pages: {df['source'].nunique()}")
print(f"Total nodes: {len(df)}")
print(f"Label counts:\n{df['label'].value_counts()}")


# Params ---------------------------------------------------------------
# sort labels, make them integers for training and return mapping to labels for eval
LABELS = sorted(df["label"].unique().tolist())
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

TAG_VOCAB = {t: i for i, t in enumerate(sorted(df["tag"].astype(str).unique().tolist()))}
PARENT_TAG_VOCAB = {t: i for i, t in enumerate(sorted(df["parent_tag"].astype(str).unique().tolist()))}

NUM_COLS = cfg["features"]["num_cols"]
BOOL_COLS = cfg["features"]["bool_cols"]

print(f"Labels: {LABELS}")
print(f"Tag vocab size: {len(TAG_VOCAB)}")
print(f"Parent tag vocab size: {len(PARENT_TAG_VOCAB)}")
print(f"Num features: {len(NUM_COLS)}")
print(f"Bool features: {len(BOOL_COLS)}")


# Train/Test Split ------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg["training"]["seed"])
train_idx, test_idx = next(gss.split(df, groups=df["source"]))

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

cv_sources = train_df["source"].unique()
test_sources = test_df["source"].unique()

print(f"\nTrain pages: {len(cv_sources)}")
print(f"Test pages: {len(test_sources)}")


# Tokenizer -------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])


# Cross Validation -----------------------------------------------------
N_SPLITS = min(cfg["training"]["n_splits"], len(cv_sources))
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=cfg["training"]["seed"])

cv_results = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(cv_sources), start=1):
    
    torch.cuda.empty_cache()
    
    fold_train_sources = set(cv_sources[tr_idx])
    fold_val_sources = set(cv_sources[va_idx])

    fold_train_df = df[df["source"].isin(fold_train_sources)].copy()
    fold_val_df = df[df["source"].isin(fold_val_sources)].copy()

    num_mean, num_std = compute_num_stats(fold_train_df, NUM_COLS)
    loss_fn = bio_loss(DEVICE)
    
    print(f"\n===== Fold {fold}/{N_SPLITS} =====")
    print(f"Train pages: {fold_train_df['source'].nunique()} Val pages: {fold_val_df['source'].nunique()}")

    train_loader = make_loader(
        fold_train_df, tokenizer,
        TAG_VOCAB, PARENT_TAG_VOCAB,
        NUM_COLS, BOOL_COLS,
        num_mean, num_std,
        batch_size=cfg["training"]["batch_size"],
        max_tokens=cfg["model"]["max_tokens"],
        shuffle=True
    )
    
    val_loader = make_loader(
        fold_val_df, tokenizer,
        TAG_VOCAB, PARENT_TAG_VOCAB,
        NUM_COLS, BOOL_COLS,
        num_mean, num_std,
        batch_size=cfg["training"]["batch_size"],
        max_tokens=cfg["model"]["max_tokens"],
        shuffle=False
    )

    model, optimizer = init_model_and_optim(
        cfg,
        tag_vocab_size=len(TAG_VOCAB),
        parent_tag_vocab_size=len(PARENT_TAG_VOCAB),
        num_numeric_features=len(NUM_COLS),
        num_bool_features=len(BOOL_COLS),
        device=DEVICE
    )

    set_bert_trainable(model, False)
    best = {"f1": -1.0, "th": 0.5, "state": None}

    for epoch in range(cfg["training"]["epochs"]):

        if epoch == cfg["training"]["freeze_epochs"]:
            set_bert_trainable(model, True)

        tr_loss = run_epoch(model, optimizer, train_loader, loss_fn, DEVICE, training=True)
        val_loss = run_epoch(model, optimizer, val_loader, loss_fn, DEVICE, training=False)

        th, f1 = find_best_threshold_peak( val_loader, model, DEVICE,
                                          nms_k=cfg["inference"]["nms_k"],
                                          min_gap=cfg["inference"]["min_gap"],
                                          tol=cfg["inference"]["tol"]
                                          )

        if f1 > best["f1"]:
            best["f1"] = f1
            best["th"] = th
            best["state"] ={k: v.cpu() for k, v in model.state_dict().items()}
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | tr_loss={tr_loss:.4f} val_loss={val_loss:.4f} best_f1={best['f1']:.4f} best_th={best['th']:.2f}")


    model.load_state_dict(best["state"])

    bp, br, bf1 = boundary_metrics_peak(
        val_loader, model, DEVICE,
        threshold=best["th"],
        nms_k=cfg["inference"]["nms_k"],
        min_gap=cfg["inference"]["min_gap"],
        tol=cfg["inference"]["tol"]
    )
    
    print(f"Fold {fold} | P={bp:.4f} R={br:.4f} F1={bf1:.4f} (th={best['th']:.2f})")

    cv_results.append({
        "bf1": bf1,
        "th": best["th"],
    })

    del model, optimizer
    torch.cuda.empty_cache()

# CV summary
mean_bf1 = float(np.mean([x["bf1"] for x in cv_results]))
mean_th = float(np.mean([x["th"] for x in cv_results]))
best_th_cv = mean_th

print(f"\n===== CV Summary =====")
print(f"Mean F1: {mean_bf1:.4f}")
print(f"Mean threshold: {mean_th:.4f}")
print(f"Using threshold for final eval: {best_th_cv:.4f}")


# Final train on all data -----------------------------------------------------
print("\nTraining final model on all CV data...")
final_train_df = df[df["source"].isin(set(cv_sources))].copy()
final_num_mean, final_num_std = compute_num_stats(final_train_df, NUM_COLS)

train_loader = make_loader(
    final_train_df, tokenizer,
    TAG_VOCAB, PARENT_TAG_VOCAB,
    NUM_COLS, BOOL_COLS,
    final_num_mean, final_num_std,
    batch_size=cfg["training"]["batch_size"],
    max_tokens=cfg["model"]["max_tokens"],
    shuffle=True
)

loss_fn = bio_loss(DEVICE)

model, optimizer = init_model_and_optim(
    cfg,
    tag_vocab_size=len(TAG_VOCAB),
    parent_tag_vocab_size=len(PARENT_TAG_VOCAB),
    num_numeric_features=len(NUM_COLS),
    num_bool_features=len(BOOL_COLS),
    device=DEVICE
)

set_bert_trainable(model, False)

for epoch in range(cfg["training"]["epochs"]):
    if epoch == cfg["training"]["freeze_epochs"]:
        set_bert_trainable(model, True)

    tr_loss = run_epoch(model, optimizer, train_loader, loss_fn, DEVICE, training=True)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:02d} | tr_loss={tr_loss:.4f}")

print("Final model training done.")


# Holdout evaluation ------------------------------------------------------------
test_loader = make_loader(
    test_df, tokenizer,
    TAG_VOCAB, PARENT_TAG_VOCAB,
    NUM_COLS, BOOL_COLS,
    final_num_mean, final_num_std,
    batch_size=cfg["training"]["batch_size"],
    max_tokens=cfg["model"]["max_tokens"],
    shuffle=False
)

p, r, f1 = boundary_metrics_peak(
    test_loader, model, DEVICE,
    threshold=best_th_cv,
    nms_k=cfg["inference"]["nms_k"],
    min_gap=cfg["inference"]["min_gap"],
    tol=cfg["inference"]["tol"]
)

print(f"\n===== HOLDOUT TEST =====")
print(f"START: P={p:.4f} R={r:.4f} F1={f1:.4f}")
print(f"Threshold used: {best_th_cv:.4f}")


# Save checkpoint ----------------------------------------------------------------
checkpoint = {
    # model weights
    "model_state_dict": model.state_dict(),

    # vocabs needed to rebuild the model and dataset
    "label2id": label2id,
    "id2label": id2label,
    "tag_vocab": TAG_VOCAB,
    "parent_tag_vocab": PARENT_TAG_VOCAB,

    # normalization stats needed for inference
    "num_mean": final_num_mean,
    "num_std": final_num_std,

    # feature columns so inference knows what to expect
    "num_cols": NUM_COLS,
    "bool_cols": BOOL_COLS,

    # best threshold from CV
    "best_th": best_th_cv,

    # config used for this run
    "cfg": cfg,
}

torch.save(checkpoint, "models/dom_extractor_checkpoint.pt")
print("Checkpoint saved to models/dom_extractor_checkpoint.pt")