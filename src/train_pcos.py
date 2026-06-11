# src/train_pcos.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def build_pipeline(num_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    pre = ColumnTransformer([("num", num_pipe, num_cols)], remainder="drop")
    clf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

def main(input_csv="data/processed/pcos_cleaned.csv", out_model="models/pcos_model.joblib"):
    df = pd.read_csv(input_csv)
    if "pcos_label" not in df.columns:
        raise ValueError("pcos_label not found in cleaned dataset")
    IMPORTANT_FEATURES = [
        "follicle_r",
        "follicle_l",
        "hirsutism",
        "amh",
        "bmi",
        "lh",
        "weight",
        "age",
        "irregular_cycle",
        "height",
        "acne"
    ]
    print("Locking training to features:", IMPORTANT_FEATURES)
    X = df[IMPORTANT_FEATURES]
    y = df["pcos_label"]

    # If extreme class imbalance, warn
    print("Label distribution:\n", y.value_counts())
    # numeric features only
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    pipe = build_pipeline(num_cols)

    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # quick grid
    params = {
        "clf__n_estimators": [200, 300],
        "clf__max_depth": [None, 10, 20]
    }
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, params, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    print("Best params:", gs.best_params_)

    # save model
    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out_model)
    print("Saved model to", out_model)

    # evaluate
    probs = best.predict_proba(X_test)[:,1]
    preds = best.predict(X_test)
    print("AUC:", roc_auc_score(y_test, probs))
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/pcos_cleaned.csv")
    parser.add_argument("--out", default="models/pcos_model.joblib")
    args = parser.parse_args()
    main(args.data, args.out)
