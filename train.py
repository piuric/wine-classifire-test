import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
import pickle, os, json


# load wine quality data (red + white)
print("Loading data...")
red = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    sep=";")
white = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    sep=";")

red["type"] = 0
white["type"] = 1
df = pd.concat([red, white], ignore_index=True)

# binary target: good wine if quality >= 7
df["good"] = (df["quality"] >= 7).astype(int)
df.drop("quality", axis=1, inplace=True)

print(f"Total samples: {len(df)}")
print(f"Features: {df.shape[1] - 1}")
print(f"Good: {df['good'].sum()}, Not good: {(df['good']==0).sum()}")

X = df.drop("good", axis=1)
y = df["good"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

pickle.dump(scaler, open("model/scaler.pkl", "wb"))
json.dump(list(X.columns), open("model/features.json", "w"))

# save test set so we can try it in the app
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df["good"] = y_test.values
test_df.to_csv("data/test.csv", index=False)
print(f"Test set saved: {len(test_df)} rows")

# models to train
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
}

results = {}

for name, clf in models.items():
    clf.fit(X_train_sc, y_train)
    preds = clf.predict(X_test_sc)
    probs = clf.predict_proba(X_test_sc)[:, 1]

    m = {
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "AUC": round(roc_auc_score(y_test, probs), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "F1": round(f1_score(y_test, preds), 4),
        "MCC": round(matthews_corrcoef(y_test, preds), 4),
    }
    results[name] = m

    fname = name.lower().replace(" ", "_")
    pickle.dump(clf, open(f"model/{fname}.pkl", "wb"))

    print(f"\n{name}:")
    for k, v in m.items():
        print(f"  {k}: {v}")

with open("model/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll done! Models saved to model/")
