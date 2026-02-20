import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# loading data -----------------------------------------------------------------
# dataset = pd.read_csv("../data/full_data.csv") # for notebook
dataset = pd.read_csv(DATA / "full_data.csv")
dataset["is_event"] = dataset["event_id"].notna().astype(int)

cols = ["is_event", "source", "event_id", "label", "attributes", "link"] # columns not used for classifier
X = dataset.drop(columns=cols, axis=1)
y = dataset["is_event"]

# train/test split by source ----------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_idx, test_idx = next(gss.split(X, y, groups=dataset["source"]))

X_train, X_test = dataset.iloc[train_idx][X.columns], dataset.iloc[test_idx][X.columns]
y_train, y_test = dataset.iloc[train_idx]["is_event"], dataset.iloc[test_idx]["is_event"]

# # check to ensure the sources are separated
# train_sources = set(dataset.iloc[train_idx]["source"])
# test_sources  = set(dataset.iloc[test_idx]["source"])
# print("overlap:", train_sources & test_sources)

# CatBoost model-----------------------------------------------------------------
cat_features = ["tag", "parent_tag"] # categorical features
text_features = ["text_context"]
# link and attribute have too many missing rows to be usable

for c in text_features:
    X_train[c] = X_train[c].fillna("").astype(str)
    X_test[c]  = X_test[c].fillna("").astype(str)
    
classifier = CatBoostClassifier(iterations=1000, task_type="GPU", verbose=False)
print("Starting Training..")
classifier.fit(X_train,y_train,cat_features=cat_features, text_features=text_features) # implement early stopping or not needed?
print("Training Done! Saving Model :)")
classifier.save_model(fname="classifier.cbm") # @Yhilal02 need to fix/test this when integrating pipeline

y_pred = classifier.predict(X_test)

# Metrics -----------------------------------------------------------------------
# confusion matrix/accuracy score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(cm)
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1 Score: {f1:.2f}")

# # cross validation (used in baseline test, revisit later)
# cv_out = cross_validate(
#     estimator=CatBoostClassifier(verbose=False),
#     X=X_train,
#     y=y_train,
#     params={"cat_features": cat_features, "text_features": text_features}
# )

# scores = cv_out["test_score"]
# print("Accuracy: {:.2f} %".format(scores.mean() * 100))
# print("Standard Deviation: {:.2f} %".format(scores.std() * 100))

