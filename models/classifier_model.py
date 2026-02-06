import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score

# loading data
dataset = pd.read_csv("Data.csv") #@Yhilal02 adjust path for robustness

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # will have to split by groups later

# catboost model training (very basic rn)
# could do xgboost for speed 
cat_cols = X_train.select_dtypes(include=["object"]).columns
cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

classifier = CatBoostClassifier(
    task_type="GPU",
    devices="0",
    verbose=False
)
classifier.fit(X_train, y_train, cat_features=cat_idx)

# test set prediction
y_pred = classifier.predict(X_test)

# Metrics --------------------------------------------------
# confusion matrix/accuracy score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(cm)
print(f"Accuracy: {acc:.2f}")

# # cross validation (maybe for later, takes some time)
# cv_out = cross_validate(
#     estimator=CatBoostClassifier(verbose=False),
#     X=X_train,
#     y=y_train,
#     cv=3,  # small dataset
#     params={"cat_features": cat_idx}
# )

# scores = cv_out["test_score"]
# print("Accuracy: {:.2f} %".format(scores.mean() * 100))
# print("Standard Deviation: {:.2f} %".format(scores.std() * 100))



