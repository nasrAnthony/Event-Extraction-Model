import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from scipy.sparse import hstack, csr_matrix

from helpers.utils import load_config

ROOT = Path(__file__).resolve().parent.parent


def build_features(df, tfidf, tag_columns, parent_columns, num_cols, bool_cols, fit=False):
    """
    Builds feature matrix for a dataframe of nodes.
    fit=True when building from training data (fits tfidf).
    fit=False for val/test/inference (transforms only).
    """
    # text features
    if fit:
        X_text = tfidf.fit_transform(df["text_context"].astype(str))
    else:
        X_text = tfidf.transform(df["text_context"].astype(str))

    # numeric features
    num_feats = df[num_cols].fillna(0).astype("float32").values

    # bool features
    bool_feats = df[bool_cols].fillna(0).astype("float32").values

    # tag one-hot
    tag_onehot = pd.get_dummies(df["tag"].astype(str))
    tag_onehot = tag_onehot.reindex(columns=tag_columns, fill_value=0).astype("float32").values

    # parent tag one-hot
    parent_onehot = pd.get_dummies(df["parent_tag"].astype(str))
    parent_onehot = parent_onehot.reindex(columns=parent_columns, fill_value=0).astype("float32").values

    X_struct = csr_matrix(np.hstack([num_feats, bool_feats, tag_onehot, parent_onehot]))

    return hstack([X_text, X_struct])


if __name__ == "__main__":

    cfg = load_config()
    NUM_COLS = cfg["features"]["num_cols"]
    BOOL_COLS = cfg["features"]["bool_cols"]

    # load data ------------------------------------------------------------------
    df = pd.read_csv(ROOT / "data" / "full_data.csv")

    df = df[df["label"] != "Other"].copy() # remove "Other" nodes
    print(f"Training samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # encode labels --------------------------------------------------------------
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    print(f"Classes: {list(le.classes_)}")

    # teain/test split -----------------------------------------------------------
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=cfg["training"]["seed"], stratify=y
    )

    # features ------------------------------------------------------------------
    tag_columns = sorted(df["tag"].astype(str).unique().tolist())
    parent_columns = sorted(df["parent_tag"].astype(str).unique().tolist())

    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2), stop_words="english")

    X_train = build_features(train_df, tfidf, tag_columns, parent_columns,
                             NUM_COLS, BOOL_COLS, fit=True)
    X_test = build_features(test_df, tfidf, tag_columns, parent_columns,
                            NUM_COLS, BOOL_COLS, fit=False)

    # training ------------------------------------------------------------------
    print("\nTraining GradientBoostingClassifier...")
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=cfg["training"]["seed"]
    )
    clf.fit(X_train.toarray(), y_train)

    # evaluation ----------------------------------------------------------------
    y_pred = clf.predict(X_test.toarray())

    print("\n===== Field Classifier Evaluation =====")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"Micro F1: {f1_score(y_test, y_pred, average='micro'):.4f}")
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")

    # save model ----------------------------------------------------------------
    bundle = {
        "clf": clf,
        "tfidf": tfidf,
        "label_encoder": le,
        "tag_columns": tag_columns,
        "parent_columns": parent_columns,
        "num_cols": NUM_COLS,
        "bool_cols": BOOL_COLS,
    }

    joblib.dump(bundle, ROOT / "models" / "field_classifier.joblib")
    print("\nSaved to models/field_classifier.joblib")