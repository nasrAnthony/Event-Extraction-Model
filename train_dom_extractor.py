import copy
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GroupShuffleSplit, KFold
from transformers import AutoTokenizer

from helpers.dataset import merge_labels, compute_num_stats
from helpers.losses import make_losses_for_train_df
from helpers.metrics import find_best_threshold_peak, boundary_metrics_peak, field_metrics_fast
from helpers.train_utils import make_loaders, run_epoch
from models.dom_extractor import init_model_and_optim, set_bert_trainable


# config --------------------------------------
MODEL_NAME = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
FREEZE_EPOCHS = 4
BATCH_SIZE = 2
MAX_TOKENS = 64
NMS_K = 1
MIN_GAP = 2
TOL = 1

# load data and preprocess -------------------------------------------------
df = pd.read_csv("data/full_data.csv")

df = merge_labels(df, "label")
df = df.sort_values(["source", "rendering_order"]).reset_index(drop=True) # extra cleaninig
df["start_event"] = 0

df["in_event"] = df["event_id"].notna().astype(int) # binary event flag column
m = df["event_id"].notna() # mask for events only

# take index of first row of each event (event boundary)
first_idx = df.loc[m].groupby(["source", "event_id"], sort=False).head(1).index 
df.loc[first_idx, "start_event"] = 1

# outside, begin, inside tagging
df["bio"] = 0 # outside
df.loc[df["in_event"].eq(1), "bio"] = 2 # events inside
df.loc[df["start_event"].eq(1), "bio"] = 1 # boundary -> beginning


# Params --------------------------------------------------

# sort labels, make them integers for training and return mapping to labels for eval
LABELS = sorted(df["label"].unique().tolist())
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

TAG_VOCAB = {t: i for i, t in enumerate(sorted(df["tag"].astype(str).unique().tolist()))}
PARENT_TAG_VOCAB = {t: i for i, t in enumerate(sorted(df["parent_tag"].astype(str).unique().tolist()))}

NUM_COLS = [
    "depth","sibling_index","children_count","same_tag_sibling_count",
    "same_text_sibling_count","text_length","word_count",
    "letter_ratio","digit_ratio","whitespace_ratio","attribute_count"
]

BOOL_COLS = [
    "has_link","link_is_absolute","parent_has_link","is_leaf",
    "contains_date","contains_time","starts_with_digit","ends_with_digit",
    "has_class","has_id",
    "attr_has_word_name","attr_has_word_date","attr_has_word_time","attr_has_word_location","attr_has_word_link",
    "text_has_word_name","text_has_word_date","text_word_time","text_word_description","text_word_location"
]

# Train/Test Split -----------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df["source"]))

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

cv_sources = train_df["source"].unique()
test_sources = test_df["source"].unique()


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =============================================================================
# CROSS VALIDATION
# =============================================================================

N_SPLITS = min(5, len(cv_sources))
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

cv_results = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(cv_sources), start=1):

    fold_train_sources = set(cv_sources[tr_idx])
    fold_val_sources = set(cv_sources[va_idx])

    fold_train_df = df[df["source"].isin(fold_train_sources)].copy()
    fold_val_df = df[df["source"].isin(fold_val_sources)].copy()

    num_mean, num_std = compute_num_stats(fold_train_df, NUM_COLS)

    field_loss_fn, bio_loss_fn, in_event_loss_fn = make_losses_for_train_df(
        fold_train_df,
        LABELS,
        DEVICE,
        other_scale=0.01,
        weight_cap=50.0
    )
    print(f"\n===== Fold {fold}/{N_SPLITS} =====")
    print("Train pages:", fold_train_df["source"].nunique(), "Val pages:", fold_val_df["source"].nunique())

    train_loader, val_loader = make_loaders(
        fold_train_df,
        fold_val_df,
        tokenizer,
        label2id,
        TAG_VOCAB,
        PARENT_TAG_VOCAB,
        NUM_COLS,
        BOOL_COLS,
        num_mean,
        num_std,
        batch_size=BATCH_SIZE,
        max_tokens=MAX_TOKENS
    )

    model, optimizer = init_model_and_optim(
        text_model_name=MODEL_NAME,
        num_field_labels=len(LABELS),
        tag_vocab_size=len(TAG_VOCAB),
        parent_tag_vocab_size=len(PARENT_TAG_VOCAB),
        num_numeric_features=len(NUM_COLS),
        num_bool_features=len(BOOL_COLS),
        device=DEVICE
    )

    set_bert_trainable(model, False)

    best = {"f1": -1.0, "th": 0.5, "state": None}

    for epoch in range(EPOCHS):

        if epoch == FREEZE_EPOCHS:
            set_bert_trainable(model, True)

        tr_loss = run_epoch(
            model, optimizer, train_loader,
            field_loss_fn, bio_loss_fn, in_event_loss_fn,
            DEVICE,
            training=True
        )

        va_loss = run_epoch(
            model, optimizer, val_loader,
            field_loss_fn, bio_loss_fn, in_event_loss_fn,
            DEVICE,
            training=False
        )

        th, f1 = find_best_threshold_peak(
            val_loader,
            model,
            DEVICE,
            nms_k=NMS_K,
            min_gap=MIN_GAP,
            tol=TOL
        )

        if f1 > best["f1"]:
            best["f1"] = f1
            best["th"] = th
            best["state"] = copy.deepcopy(model.state_dict())
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} best_startF1={best['f1']:.4f} best_th={best['th']:.2f}")


    model.load_state_dict(best["state"])

    bp, br, bf1 = boundary_metrics_peak(
        val_loader,
        model,
        DEVICE,
        threshold=best["th"],
        nms_k=NMS_K,
        min_gap=MIN_GAP,
        tol=TOL
    )

    field_micro_f1 = field_metrics_fast(val_loader, model, DEVICE, label2id, average="micro")
    field_macro_f1 = field_metrics_fast(val_loader, model, DEVICE, label2id, average="macro")
    
    print(
        f"Fold {fold} | START (peak-based): P={bp:.4f} R={br:.4f} F1={bf1:.4f} (th={best['th']:.2f}, nms_k={NMS_K}, min_gap={MIN_GAP}, tol=±{TOL})"
        f" | field: microF1={field_micro_f1:.4f} macroF1={field_macro_f1:.4f}"
    )

    cv_results.append({
        "bf1": bf1,
        "th": best["th"],
        "field_micro": field_micro_f1,
        "field_macro": field_macro_f1
    })


# =============================================================================
# FINAL TRAIN ON ALL CV DATA
# =============================================================================

final_train_df = df[df["source"].isin(set(cv_sources))].copy()

final_num_mean, final_num_std = compute_num_stats(final_train_df, NUM_COLS)

train_loader, _ = make_loaders(
    final_train_df,
    final_train_df,
    tokenizer,
    label2id,
    TAG_VOCAB,
    PARENT_TAG_VOCAB,
    NUM_COLS,
    BOOL_COLS,
    final_num_mean,
    final_num_std,
    batch_size=BATCH_SIZE,
    max_tokens=MAX_TOKENS
)

field_loss_fn, bio_loss_fn, in_event_loss_fn = make_losses_for_train_df(
    final_train_df,
    LABELS,
    DEVICE,
    other_scale=0.01
)

model, optimizer = init_model_and_optim(
    MODEL_NAME,
    len(LABELS),
    len(TAG_VOCAB),
    len(PARENT_TAG_VOCAB),
    len(NUM_COLS),
    len(BOOL_COLS),
    DEVICE
)

set_bert_trainable(model, False)

for epoch in range(EPOCHS):

    if epoch == FREEZE_EPOCHS:
        set_bert_trainable(model, True)

    run_epoch(
        model, optimizer,
        train_loader,
        field_loss_fn, bio_loss_fn, in_event_loss_fn,
        DEVICE, training=True
    )


# =============================================================================
# HOLDOUT EVALUATION
# =============================================================================

test_loader, _ = make_loaders(
    test_df, test_df, tokenizer,
    label2id,
    TAG_VOCAB,
    PARENT_TAG_VOCAB,
    NUM_COLS,
    BOOL_COLS,
    final_num_mean,
    final_num_std,
    batch_size=BATCH_SIZE,
    max_tokens=MAX_TOKENS
)

best_th_cv = float(np.mean([x["th"] for x in cv_results]))

p, r, f1 = boundary_metrics_peak(
    test_loader, model, DEVICE,
    threshold=best_th_cv,
    nms_k=NMS_K, min_gap=MIN_GAP, tol=TOL
)

field_micro = field_metrics_fast(test_loader, model, DEVICE, label2id, average="micro")
field_macro = field_metrics_fast(test_loader, model, DEVICE, label2id, average="macro")


print("\n===== HOLDOUT TEST =====")
print(f"START: P={p:.4f} R={r:.4f} F1={f1:.4f}")
print(f"FIELD: microF1={field_micro:.4f} macroF1={field_macro:.4f}")
