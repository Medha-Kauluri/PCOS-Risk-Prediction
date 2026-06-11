# src/preprocess_pcos.py

import pandas as pd
import numpy as np
from pathlib import Path


def load_pcos_excel(path: str, sheet: str = "Full_new") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    # normalise column names
    df.columns = [c.strip() for c in df.columns]
    return df


def _find(df: pd.DataFrame, candidates):
    """Return first column whose name contains any of candidates (case-insensitive)."""
    for key in candidates:
        for c in df.columns:
            if key.lower() in c.lower():
                return c
    return None


def clean_pcos_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # --- Label ---
    if "PCOS (Y/N)" in df.columns:
        df = df.rename(columns={"PCOS (Y/N)": "pcos_label"})
    elif "PCOS" in df.columns:
        df = df.rename(columns={"PCOS": "pcos_label"})
    else:
        raise ValueError("Could not find PCOS label column")

    df["pcos_label"] = pd.to_numeric(df["pcos_label"], errors="coerce").fillna(0).astype(int)

    # --- locate columns used in modelling ---
    age_col = _find(df, ["age (yrs)", "age"])
    weight_col = _find(df, ["weight", "weight (kg)"])
    height_col = _find(df, ["height", "height(cm)", "height (cm)"])
    amh_col = _find(df, ["amh"])
    lh_col = _find(df, ["lh(" , "lh"])
    follicle_l_col = _find(df, ["follicle no. (l)", "follicle no l", "follicle_l"])
    follicle_r_col = _find(df, ["follicle no. (r)", "follicle no r", "follicle_r"])
    hirs_col = _find(df, ["hair growth", "hirsutism"])
    acne_col = _find(df, ["pimples", "acne"])
    irregular_col = _find(df, ["cycle(r/i)", "cycle (r/i)", "cycle", "irregular"])

    # Build X with the exact feature names used in the model comparison
    X = pd.DataFrame()

    if age_col:
        X["age"] = df[age_col]
    if weight_col:
        X["weight"] = df[weight_col]
    if height_col:
        X["height"] = df[height_col]
    if amh_col:
        X["amh"] = df[amh_col]
    if lh_col:
        X["lh"] = df[lh_col]
    if follicle_l_col:
        X["follicle_l"] = df[follicle_l_col]
    if follicle_r_col:
        X["follicle_r"] = df[follicle_r_col]
    if hirs_col:
        X["hirsutism"] = df[hirs_col]
    if acne_col:
        X["acne"] = df[acne_col]
    if irregular_col:
        X["irregular_cycle"] = df[irregular_col]

    # BMI – either present or computed
    if "bmi" not in X.columns and weight_col and height_col:
        h = pd.to_numeric(df[height_col], errors="coerce")
        w = pd.to_numeric(df[weight_col], errors="coerce")
        X["bmi"] = w / ((h / 100.0) ** 2)
    elif "BMI" in df.columns and "bmi" not in X.columns:
        X["bmi"] = df["BMI"]

    # Convert yes/no style columns to 0/1
    for col in ["hirsutism", "acne", "irregular_cycle"]:
        if col in X.columns:
            s = X[col].astype(str).str.strip().str.lower()
            X[col] = (
                s.replace(
                    {
                        "y": 1,
                        "yes": 1,
                        "1": 1,
                        "n": 0,
                        "no": 0,
                        "0": 0,
                        "nan": np.nan,
                        "": np.nan,
                    }
                )
                .astype(float)
            )

    # Make everything numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Drop all-NaN columns
    X = X.dropna(axis=1, how="all")

    # Fill numeric NaNs with median
    for c in X.columns:
        if c != "pcos_label":
            X[c] = X[c].fillna(X[c].median())

    # add label at the end
    X["pcos_label"] = df["pcos_label"].astype(int)

    return X


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/PCOS_data_without_infertility.xlsx")
    parser.add_argument("--out", default="data/processed/pcos_cleaned.csv")
    args = parser.parse_args()

    raw = load_pcos_excel(args.input)
    cleaned = clean_pcos_df(raw)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.out, index=False)
    print("Saved cleaned PCOS dataset to", args.out)
