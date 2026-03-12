import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from helpers.utils import load_config # probably won't be able to run this..
from helpers.dataset import merge_labels

ROOT = Path(__file__).resolve().parent.parent


def build_features(df, tfidf, tag_columns, parent_columns,
                   num_cols, bool_cols, fit=False,
                   use_tag=True, use_parent_tag=True):
    """
    Builds feature matrix for a dataframe of nodes.
    fit=True when building from training data (fits tfidf).
    fit=False for val/test/inference (transforms only).
    """
    if fit:
        X_text = tfidf.fit_transform(df["text_context"].astype(str)).toarray()
    else:
        X_text = tfidf.transform(df["text_context"].astype(str)).toarray()

    num_feats = df[num_cols].fillna(0).astype("float32").values
    bool_feats = df[bool_cols].fillna(0).astype("float32").values

    parts = [X_text, num_feats, bool_feats]

    if use_tag:
        tag_onehot = pd.get_dummies(df["tag"].astype(str))
        parts.append(
            tag_onehot.reindex(columns=tag_columns, fill_value=0).astype("float32").values
        )

    if use_parent_tag:
        parent_onehot = pd.get_dummies(df["parent_tag"].astype(str))
        parts.append(
            parent_onehot.reindex(columns=parent_columns, fill_value=0).astype("float32").values
        )

    return np.hstack(parts)


def evaluate(y_true, y_pred, classes):
    """prints full evaluation suite"""
    print(classification_report(y_true, y_pred, target_names=classes))
    print(f"Micro F1: {f1_score(y_true, y_pred, average='micro'):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


cfg = load_config()
NUM_COLS = cfg["features"]["num_cols"]
BOOL_COLS = cfg["features"]["bool_cols"]
USE_TAG = cfg["model"].get("use_tag", True)
USE_PARENT_TAG = cfg["model"].get("use_parent_tag", True)
SEED = cfg["training"]["seed"]
N_SPLITS = cfg["training"]["n_splits"]
# ── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv(ROOT / "data" / "full_data.csv")
df = merge_labels(df, "label")
field_df = df[df["label"] != "Other"].copy()
print(f"Training samples: {len(field_df)}")
print(f"Label distribution:\n{field_df['label'].value_counts()}")
# ── ENCODE LABELS ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(field_df["label"])
print(f"\nClasses: {list(le.classes_)}")
# ── FEATURE COLUMN STRUCTURE ──────────────────────────────────────────────────
# built from full field_df so all tags/parent tags are covered
tag_columns = sorted(field_df["tag"].astype(str).unique().tolist())
parent_columns = sorted(field_df["parent_tag"].astype(str).unique().tolist())
# ── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────
train_df, test_df, y_train, y_test = train_test_split(
    field_df, y, test_size=0.2, random_state=SEED, stratify=y
)
tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words="english")
X_train = build_features(train_df, tfidf, tag_columns, parent_columns,
                         NUM_COLS, BOOL_COLS, fit=True,
                         use_tag=USE_TAG, use_parent_tag=USE_PARENT_TAG)
X_test = build_features(test_df, tfidf, tag_columns, parent_columns,
                        NUM_COLS, BOOL_COLS, fit=False,
                        use_tag=USE_TAG, use_parent_tag=USE_PARENT_TAG)
# ── CROSS VALIDATION (optional) ───────────────────────────────────────────────
if N_SPLITS > 1:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    micro_scores, macro_scores = [], []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(field_df), start=1):
        fold_train = field_df.iloc[tr_idx]
        fold_val = field_df.iloc[va_idx]
        fold_tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words="english")
        X_tr = build_features(fold_train, fold_tfidf, tag_columns, parent_columns,
                              NUM_COLS, BOOL_COLS, fit=True,
                              use_tag=USE_TAG, use_parent_tag=USE_PARENT_TAG)
        X_va = build_features(fold_val, fold_tfidf, tag_columns, parent_columns,
                              NUM_COLS, BOOL_COLS, fit=False,
                              use_tag=USE_TAG, use_parent_tag=USE_PARENT_TAG)
        fold_clf = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED
        )
        fold_clf.fit(X_tr, y[tr_idx])
        y_pred_fold = fold_clf.predict(X_va)
        micro = f1_score(y[va_idx], y_pred_fold, average="micro")
        macro = f1_score(y[va_idx], y_pred_fold, average="macro")
        micro_scores.append(micro)
        macro_scores.append(macro)
        print(f"Fold {fold} | Micro F1={micro:.4f}  Macro F1={macro:.4f}")
    print(f"\n===== CV Summary =====")
    print(f"Mean Micro F1: {np.mean(micro_scores):.4f}")
    print(f"Mean Macro F1: {np.mean(macro_scores):.4f}")
# ── TRAIN FINAL MODEL ON TRAIN SPLIT ──────────────────────────────────────────
print("\nTraining GradientBoostingClassifier...")
clf = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED
)
clf.fit(X_train, y_train)
# ── EVALUATE ON TEST SPLIT ────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
print("\n===== Field Classifier Evaluation =====")
evaluate(y_test, y_pred, le.classes_)
# ── SAVE ──────────────────────────────────────────────────────────────────────
bundle = {
    "clf": clf,
    "tfidf": tfidf,
    "label_encoder": le,
    "tag_columns": tag_columns,
    "parent_columns": parent_columns,
    "num_cols": NUM_COLS,
    "bool_cols": BOOL_COLS,
    "use_tag": USE_TAG,
    "use_parent_tag": USE_PARENT_TAG,
}
save_path = ROOT / "models" / "field_classifier.joblib"
joblib.dump(bundle, save_path)
print(f"\nSaved to {save_path}")