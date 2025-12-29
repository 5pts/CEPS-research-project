import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Config
ROOT = Path.cwd()
DEFAULT_DATA_FILE = ROOT / "rescued_data" / "merged_rescued_all.csv"
DATA_FILE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总\rescued_data\merged_rescued_all.csv")
REPORT_FILE = ROOT / "final_data_check_report.txt"

def _resolve_data_file():
    if DEFAULT_DATA_FILE.exists():
        return DEFAULT_DATA_FILE
    return DATA_FILE


def _load_df(data_file):
    return pd.read_csv(data_file)


def _missingness_summary(df, target_var):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    lines = []
    lines.append("\n--- 3. Missing Values Check ---")
    if missing.empty:
        lines.append("[PASS] No missing values detected.")
    else:
        for name, count in missing.items():
            rate = count / len(df)
            lines.append(f"{name}: {count} missing ({rate:.2%})")
    if target_var in df.columns:
        nulls = df[target_var].isnull().sum()
        if nulls > 0:
            lines.append(f"[FAIL] Target '{target_var}' has {nulls} missing values!")
        else:
            lines.append(f"[PASS] Target '{target_var}' has 0 missing values.")
    else:
        lines.append(f"[FAIL] '{target_var}' column missing!")
    return lines


def _missingness_health_check(df, focus_vars, compare_vars):
    lines = []
    lines.append("\n--- 6. Missingness Health Check ---")
    if not focus_vars:
        lines.append("[WARN] No focus variables available for missingness checks.")
        return lines

    def summarize_missing_vs(miss_var, comp_var):
        s = df[comp_var]
        miss = df[miss_var].isna()
        if s.isna().mean() > 0.5:
            return None
        nunique = s.dropna().nunique()
        if nunique <= 20:
            grp = df.groupby(s)[miss_var].apply(lambda x: x.isna().mean())
            if len(grp) < 2:
                return None
            max_cat = grp.idxmax()
            min_cat = grp.idxmin()
            return {
                "type": "categorical",
                "max_cat": max_cat,
                "max_rate": float(grp.max()),
                "min_cat": min_cat,
                "min_rate": float(grp.min()),
                "diff": float(grp.max() - grp.min()),
            }
        miss_mean = s[miss].dropna().mean()
        ok_mean = s[~miss].dropna().mean()
        if np.isnan(miss_mean) or np.isnan(ok_mean):
            return None
        return {
            "type": "numeric",
            "miss_mean": float(miss_mean),
            "ok_mean": float(ok_mean),
            "diff": float(miss_mean - ok_mean),
        }

    flagged = []
    for mv in focus_vars:
        if mv not in df.columns:
            continue
        miss_rate = df[mv].isna().mean()
        if miss_rate == 0:
            continue
        results = []
        for cv in compare_vars:
            if cv == mv or cv not in df.columns:
                continue
            res = summarize_missing_vs(mv, cv)
            if res is None:
                continue
            res["compare_var"] = cv
            results.append(res)
        results.sort(key=lambda x: abs(x["diff"]), reverse=True)
        lines.append(f"{mv} (missing rate {miss_rate:.2%})")
        if not results:
            lines.append("  - no usable covariates for comparison")
            continue
        for res in results[:3]:
            cv = res["compare_var"]
            if res["type"] == "categorical":
                lines.append(
                    f"  - {cv}: missing rate diff {res['diff']:.2%} "
                    f"(max {res['max_cat']}={res['max_rate']:.2%}, "
                    f"min {res['min_cat']}={res['min_rate']:.2%})"
                )
                if res["diff"] >= 0.05:
                    flagged.append((mv, cv, res["diff"], "categorical"))
            else:
                lines.append(
                    f"  - {cv}: mean diff {res['diff']:.3f} "
                    f"(missing mean {res['miss_mean']:.3f}, "
                    f"non-missing mean {res['ok_mean']:.3f})"
                )
                sd = df[cv].dropna().std()
                if sd and abs(res["diff"]) >= 0.3 * sd:
                    flagged.append((mv, cv, res["diff"], "numeric"))

    lines.append("Heuristic flags (potentially non-random missingness):")
    if flagged:
        for mv, cv, diff, t in flagged[:20]:
            if t == "categorical":
                lines.append(f"- {mv} vs {cv}: missing-rate spread {diff:.2%}")
            else:
                lines.append(f"- {mv} vs {cv}: mean gap {diff:.3f}")
    else:
        lines.append("- none triggered by simple thresholds")
    return lines


def _visualization_checks(root):
    lines = []
    lines.append("\n--- 7. Visualization Checks ---")
    pngs = list(root.rglob("*.png"))
    if not pngs:
        lines.append("[WARN] No PNG files found.")
        return lines

    try:
        from PIL import Image

        pil_ok = True
    except Exception:
        pil_ok = False

    lines.append(f"PNG files found: {len(pngs)}")
    for p in sorted(pngs):
        size = p.stat().st_size
        rel = p.relative_to(root)
        if size == 0:
            lines.append(f"[FAIL] {rel} is 0 bytes")
            continue
        if pil_ok:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                if w < 200 or h < 200:
                    lines.append(f"[WARN] {rel} small image ({w}x{h})")
            except Exception as exc:
                lines.append(f"[FAIL] {rel} cannot be opened: {exc}")
        else:
            lines.append(f"[WARN] PIL not available, skipped opening {rel}")

    readme = root / "figures_readme.md"
    if readme.exists():
        refs = set()
        try:
            raw = readme.read_bytes()
            for match in re.findall(rb"[A-Za-z0-9_\-]+\.png", raw):
                refs.add(match.decode("ascii", errors="ignore"))
        except Exception:
            refs = set()
        if refs:
            for ref in sorted(refs):
                matches = list(root.rglob(os.path.basename(ref)))
                if matches:
                    rels = ", ".join(str(m.relative_to(root)) for m in matches)
                    lines.append(f"[PASS] {ref}: FOUND ({rels})")
                else:
                    lines.append(f"[FAIL] {ref}: MISSING")
    return lines


def check_data():
    data_file = _resolve_data_file()
    print(f"Checking file: {data_file}")
    if not data_file.exists():
        print("[ERROR] File not found!")
        return

    df = _load_df(data_file)
    report_lines = []

    # 1. Basic Shape
    report_lines.append("\n--- 1. Basic Shape ---")
    report_lines.append(f"Rows: {len(df)}")
    report_lines.append(f"Columns: {list(df.columns)}")
    
    # 2. ID Uniqueness
    report_lines.append("\n--- 2. ID Uniqueness ---")
    if "ids" in df.columns:
        n_unique = df["ids"].nunique()
        report_lines.append(f"Unique IDs: {n_unique}")
        if n_unique != len(df):
            report_lines.append(f"[WARN] ID duplicates found! Dups: {len(df) - n_unique}")
        else:
            report_lines.append("[PASS] IDs are unique.")
    else:
        report_lines.append("[ERROR] 'ids' column missing!")

    # 3. Missing Values Check
    report_lines.extend(_missingness_summary(df, target_var="expect_college"))

    # Critical Check: Target Variable distribution
    if "expect_college" in df.columns:
        dist = df["expect_college"].value_counts(normalize=True)
        report_lines.append("Distribution of expect_college:")
        report_lines.append(str(dist))
        if dist.max() > 0.95:
            report_lines.append("[WARN] Target imbalance is very high (>95%)!")
    else:
        report_lines.append("[FAIL] 'expect_college' column missing!")

    # 4. Key Predictors Check
    report_lines.append("\n--- 4. Key Predictors Check ---")
    predictors = ["ses_self", "bonding_idx", "teacher_praise", "teacher_talk", "cog_score"]
    for p in predictors:
        if p in df.columns:
            nulls = df[p].isnull().sum()
            report_lines.append(f"{p}: {nulls} missing")
            # Check for placeholder/sentinel values like -999, -99, -888.
            if pd.api.types.is_numeric_dtype(df[p]):
                min_val = df[p].min()
                if min_val <= -99 or (min_val in (-1, -2, -3, -8, -9) and df[p].nunique() < 15):
                    report_lines.append(
                        f"[WARN] {p} has possible sentinel values (min={min_val}). "
                        "Check if these are error codes."
                    )
        else:
            report_lines.append(f"[WARN] Predictor {p} missing from dataset.")

    # 5. Linkage Columns
    report_lines.append("\n--- 5. Linkage Columns ---")
    if "clsids" in df.columns:
        report_lines.append("[PASS] Linkage key 'clsids' is present.")
    else:
        report_lines.append("[WARN] Linkage key 'clsids' missing.")
    if "schids" in df.columns:
        report_lines.append("[PASS] Linkage key 'schids' is present.")
    else:
        report_lines.append("[INFO] Linkage key 'schids' not present in rescued dataset.")

    # 6. Missingness health check
    focus_vars = [v for v in predictors if v in df.columns]
    compare_vars = [c for c in df.columns if c not in ("ids", "clsids")]
    report_lines.extend(_missingness_health_check(df, focus_vars, compare_vars))

    # 6b. Missingness impact summary (simple heuristic)
    report_lines.append("\n--- 6b. Missingness Impact Summary ---")
    total_missing = df[focus_vars].isna().any(axis=1).sum() if focus_vars else 0
    total_rate = total_missing / len(df) if len(df) else 0
    if total_rate <= 0.02:
        report_lines.append(
            f"[PASS] Overall missingness across key predictors is low ({total_rate:.2%}). "
            "Impact on conclusions is likely limited."
        )
    else:
        report_lines.append(
            f"[WARN] Overall missingness across key predictors is {total_rate:.2%}. "
            "Consider sensitivity checks or imputation."
        )

    # 7. Visualizations
    report_lines.extend(_visualization_checks(ROOT))

    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport saved to {REPORT_FILE}")

if __name__ == "__main__":
    check_data()
