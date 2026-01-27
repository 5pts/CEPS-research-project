import numpy as np
import pandas as pd
from pathlib import Path


WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_FILE = WORKSPACE / "rescued_data" / "merged_rescued_all_with_pca_ses.csv"
REPORT_FILE = WORKSPACE / "results" / "phase3" / "ses_pca_report.txt"


PCA_COLS = [
    "parent_edu_max",
    "family_econ",
    "home_books",
    "has_desk",
    "has_computer",
]


def zscore_frame(df):
    return (df - df.mean()) / df.std(ddof=0)


def main():
    df = pd.read_csv(DATA_FILE, low_memory=False)
    missing = [c for c in PCA_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing PCA inputs: {missing}")

    # Treat cog_score=0 as missing (non-participant)
    if "cog_score" in df.columns:
        df.loc[df["cog_score"] == 0, "cog_score"] = np.nan

    X = df[PCA_COLS].copy()
    invert_computer = False
    if "has_computer" in X.columns:
        if X["has_computer"].corr(X["family_econ"]) < 0:
            X["has_computer"] = 1 - X["has_computer"]
            invert_computer = True
    if X.isna().any().any():
        raise ValueError("PCA inputs contain missing values.")

    Xz = zscore_frame(X)
    cov = np.cov(Xz.T, ddof=0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    pc1 = Xz.values @ eigvecs[:, 0]
    # Align sign so higher scores indicate higher SES
    if np.corrcoef(pc1, X["family_econ"].values)[0, 1] < 0:
        pc1 = -pc1
        eigvecs[:, 0] = -eigvecs[:, 0]

    df["ses_pca"] = pc1
    median = df["ses_pca"].median()
    df["ses_pca_group"] = (df["ses_pca"] >= median).astype(int)

    df.to_csv(DATA_FILE, index=False)

    explained = eigvals / eigvals.sum()
    with REPORT_FILE.open("w", encoding="utf-8") as f:
        f.write("SES PCA Report\n")
        f.write("================\n")
        f.write(f"Inputs: {', '.join(PCA_COLS)}\n\n")
        if invert_computer:
            f.write("Note: has_computer inverted (1 - value) due to negative SES correlation.\n\n")
        f.write("Explained variance ratio:\n")
        for i, ratio in enumerate(explained[:5], start=1):
            f.write(f"  PC{i}: {ratio:.4f}\n")
        f.write("\nPC1 loadings:\n")
        for name, loading in zip(PCA_COLS, eigvecs[:, 0]):
            f.write(f"  {name}: {loading:.4f}\n")

    print(f"[DONE] Updated {DATA_FILE}")
    print(f"[DONE] Saved PCA report to {REPORT_FILE}")


if __name__ == "__main__":
    main()
