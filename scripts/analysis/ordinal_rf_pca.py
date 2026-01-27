import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier


WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_FILE = WORKSPACE / "rescued_data" / "merged_rescued_all_with_pca_ses.csv"
OUTPUT_FILE = WORKSPACE / "results" / "phase3" / "ordinal_rf_feature_importance.csv"


def main():
    df = pd.read_csv(DATA_FILE, low_memory=False)
    cols = ["expect_edu_raw", "bonding_idx", "linking_idx", "ses_pca", "hukou_type", "cog_score"]
    df = df[cols].dropna()
    df = df[df["expect_edu_raw"].between(1, 9)].copy()

    X = df[["bonding_idx", "linking_idx", "ses_pca", "hukou_type", "cog_score"]]
    y = df["expect_edu_raw"].astype(int)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=80,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    importance = pd.DataFrame(
        {"feature": X.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT_FILE, index=False)
    print(f"[DONE] Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
