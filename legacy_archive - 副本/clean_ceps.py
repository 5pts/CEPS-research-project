#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CEPS 数据清洗脚本
- 读取四类 .dta 数据（学生、家长、任课教师、校领导）
- 统一列名格式、去除字符串空白、处理常见问卷缺失编码
- 去重并输出为 CSV/Parquet
"""
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# 如果 pandas 报错，先确认已安装：pip install pandas
# 如仍报错，可尝试卸载重装：pip uninstall pandas -y && pip install pandas
import pandas as pd
import pyreadstat

# 常见问卷缺失/无效编码（根据经验值，可在运行后根据报告调整）
SENTINEL_VALUES = {
    -9, -8, -7, -6, -5, -4, -3, -2, -1,
    97, 98, 99,
    997, 998, 999,
    9997, 9998, 9999
}

WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_PATHS = {
    "student": WORKSPACE / "学生数据" / "cepsw2studentCN.dta",
    "parent": WORKSPACE / "家长数据" / "cepsw2parentCN.dta",
    "teacher": WORKSPACE / "任课教师数据" / "cepsw2teacherCN.dta",
    "principal": WORKSPACE / "校领导学校数据" / "cepsw2principalCN.dta",
}

OUTPUT_DIR = WORKSPACE / "cleaned"
REPORT_DIR = WORKSPACE / "cleaned_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def to_snake(name: str) -> str:
    # 中文列名保持原样，仅做常规清理
    n = name.strip()
    # 替换空格、全角空格、点、斜杠、连字符
    n = re.sub(r"[\s\u3000\.\/\-]+", "_", n)
    # 驼峰转下划线
    n = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", n)
    n = n.lower()
    # 去除首尾下划线、合并重复下划线
    n = re.sub(r"_+", "_", n).strip("_")
    return n


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = [to_snake(c) for c in df.columns]
    df.columns = new_cols
    return df


def strip_string_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
        # 将空字符串统一为缺失
        df[c] = df[c].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return df


def replace_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    # 仅对整数/浮点列执行替换；避免误伤真实取值
    num_cols = df.select_dtypes(include=["number"]).columns
    for c in num_cols:
        s = df[c]
        # 仅在列中确实存在哨兵值时替换
        present = [v for v in SENTINEL_VALUES if v in set(pd.unique(s.dropna()))]
        if present:
            df[c] = s.replace({v: pd.NA for v in present})
    return df


def detect_id_columns(df: pd.DataFrame) -> List[str]:
    # 常见 ID 列名关键词
    candidates = []
    for c in df.columns:
        if re.search(r"(^|_)id(_|$)", c):
            candidates.append(c)
        elif re.search(r"(student|parent|teacher|principal|school).*id", c):
            candidates.append(c)
        elif c in {"sid", "pid", "tid", "gid", "schid", "schoolid"}:
            candidates.append(c)
    # 过滤：唯一性较高的列更可能是主键
    result = []
    for c in candidates:
        unique_ratio = df[c].nunique(dropna=True) / max(len(df), 1)
        if unique_ratio > 0.5:  # 半数以上唯一
            result.append(c)
    return result or candidates


def drop_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    id_cols = detect_id_columns(df)
    before = len(df)
    if id_cols:
        df = df.drop_duplicates(subset=id_cols, keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    after = len(df)
    return df, id_cols


def summarize(df: pd.DataFrame, name: str, id_cols: List[str]) -> None:
    lines = []
    lines.append(f"数据集: {name}")
    lines.append(f"行数: {len(df)}, 列数: {df.shape[1]}")
    lines.append(f"检测到可能的ID列: {', '.join(id_cols) if id_cols else '无'}")
    # 缺失率概览（前 30 列）
    na_rate = (df.isna().sum() / len(df)).sort_values(ascending=False)
    top_na = na_rate.head(30)
    lines.append("缺失率TOP30列:")
    for c, r in top_na.items():
        lines.append(f"  - {c}: {r:.2%}")
    # 类型概览
    dtypes = df.dtypes.astype(str)
    lines.append("数据类型（前 30 列）:")
    for c, t in dtypes.head(30).items():
        lines.append(f"  - {c}: {t}")
    report_path = REPORT_DIR / f"{name}_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def clean_one_stats(kind: str, path: Path) -> Dict[str, object]:
    if not path.exists():
        print(f"[WARN] 缺失数据文件: {path}")
        return
    print(f"[INFO] 读取 {kind}: {path}")
    df, meta = pyreadstat.read_dta(str(path))
    print(f"[INFO] 原始形状: {df.shape}")

    df = standardize_columns(df)
    df = strip_string_whitespace(df)
    df = replace_sentinels(df)

    df, id_cols = drop_duplicates(df)
    print(f"[INFO] 去重后形状: {df.shape}; ID列: {id_cols}")

    # 输出
    base = path.stem + "_clean"
    csv_path = OUTPUT_DIR / f"{base}.csv"
    parquet_path = OUTPUT_DIR / f"{base}.parquet"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"[WARN] 写入 Parquet 失败（可安装 pyarrow 以提升性能）: {e}")

    summarize(df, kind, id_cols)
    print(f"[DONE] 输出: {csv_path}")
    # 汇总统计返回
    missing_total = int(df.isna().sum().sum())
    missing_rate = float(missing_total / df.size) if df.size else 0.0
    return {
        "name": kind,
        "rows": len(df),
        "cols": df.shape[1],
        "id_cols": id_cols,
        "csv_path": str(csv_path),
        "parquet_path": str(parquet_path),
        "missing_total": missing_total,
        "missing_rate": missing_rate,
    }


def write_overall_summary(stats: List[Dict[str, object]]) -> None:
    lines = []
    lines.append("CEPS 数据汇总表")
    lines.append(f"输出目录: {OUTPUT_DIR}")
    lines.append("")
    for s in stats:
        if not s:
            continue
        lines.append(f"数据集: {s.get('name')}")
        lines.append(f"  行数: {s.get('rows')}, 列数: {s.get('cols')}")
        id_cols = s.get('id_cols') or []
        lines.append(f"  可能的ID列: {', '.join(id_cols) if id_cols else '无'}")
        mt = s.get('missing_total') or 0
        mr = float(s.get('missing_rate') or 0.0)
        lines.append(f"  缺失总数: {int(mt)}, 缺失率: {mr:.2%}")
        lines.append(f"  CSV路径: {s.get('csv_path')}")
        lines.append(f"  Parquet路径: {s.get('parquet_path')}")
        lines.append("")
    lines.append(f"报告位置: {REPORT_DIR}")
    out_path = REPORT_DIR / "数据汇总表.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[DONE] 生成汇总表: {out_path}")


def analyze_student_topic() -> None:
    # 加载数据
    student_file = OUTPUT_DIR / "cepsw2studentCN_clean.parquet"
    if not student_file.exists():
         student_file = OUTPUT_DIR / "cepsw2studentCN_clean.csv"
    
    if not student_file.exists():
        print("[WARN] 无法找到清洗后的学生数据，跳过分析")
        return

    print(f"[INFO] 读取学生数据进行分析: {student_file}")
    try:
        if student_file.suffix == '.parquet':
            df = pd.read_parquet(student_file)
        else:
            df = pd.read_csv(student_file, low_memory=False)
    except Exception as e:
        print(f"[ERROR] 读取失败: {e}")
        return

    # 变量映射（请根据变量对照说明表.xls进行调整）
    # Bonding SC（师生关系/支持）：候选变量（示例，需按代码表确认）
    BONDING_CANDIDATES = [
        # 如：老师是否谈心、是否课后辅导、是否严格管理等
        # 下面是示例候选，若不存在不会报错
        "w2c11a", "w2c09a", "w2c09b"
    ]
    # Bridging SC（同伴网络/学校氛围）：候选变量（示例，需按代码表确认）
    BRIDGING_CANDIDATES = [
        # 如：班级同学计划上大学比例/周围朋友成绩等
        "w2c12", "w2c14a", "w2c14b", "w2c22a", "w2c22b"
    ]
    # SES（社会经济地位）候选（示例，需按代码表确认）
    SES_CANDIDATES = [
        # 父母受教育水平/职业/家庭资源等，学生表中可能无完整SES
        # 可后续用家长表合并；此处先在学生表中尝试
        "w2a0603"  # 作为示例占位
    ]
    # 户籍变量（示例，需按代码表确认）
    HUKOU_VAR_CANDIDATES = ["hukou", "hukou_status", "hukoutype"]
    # 教育期望（升学信心）：候选（示例，需按代码表确认）
    EXPECT_CANDIDATES = ["w2c12", "w2c11a"]  # 以是否期望上大学/普高为例

    def pick_existing(cols):
        return [c for c in cols if c in df.columns]

    bonding_vars = pick_existing(BONDING_CANDIDATES)
    bridging_vars = pick_existing(BRIDGING_CANDIDATES)
    ses_vars = pick_existing(SES_CANDIDATES)
    hukou_var = next((c for c in HUKOU_VAR_CANDIDATES if c in df.columns), None)
    expect_var = next((c for c in EXPECT_CANDIDATES if c in df.columns), None)

    # 安全标准化 + 求均值作为指数
    def zscore_series(s: pd.Series) -> pd.Series:
        s_num = pd.to_numeric(s, errors="coerce")
        mu = s_num.mean(skipna=True)
        sd = s_num.std(skipna=True)
        if pd.isna(sd) or sd == 0:
            return pd.Series(pd.NA, index=s.index)
        return (s_num - mu) / sd

    def index_from_vars(vars_list: List[str]) -> pd.Series:
        if not vars_list:
            return pd.Series(pd.NA, index=df.index)
        Z = []
        for c in vars_list:
            Z.append(zscore_series(df[c]))
        Z = pd.concat(Z, axis=1)
        return Z.mean(axis=1, skipna=True)

    df["bonding_sc_idx"] = index_from_vars(bonding_vars)
    df["bridging_sc_idx"] = index_from_vars(bridging_vars)
    df["ses_idx"] = index_from_vars(ses_vars)

    # 户籍分类
    if hukou_var:
        hv = pd.to_numeric(df[hukou_var], errors="coerce")
        # 兜底：尝试将 {1=城市, 2=农村} 或 {0/1} 映射到 ['城市','农村']
        df["hukou_cat"] = hv.map({1: "城市", 2: "农村", 0: "城市"}).astype("object")
    else:
        df["hukou_cat"] = pd.NA

    # 教育期望：二值化（1=是/期待上大学或普高；2/0=否）
    def to_binary_expect(s: pd.Series) -> pd.Series:
        sn = pd.to_numeric(s, errors="coerce")
        return sn.map({1: 1, 2: 0, 0: 0})

    if expect_var:
        df["expect_univ_bin"] = to_binary_expect(df[expect_var])
    else:
        df["expect_univ_bin"] = pd.NA

    # 图表输出目录
    FIGURES_DIR = WORKSPACE / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 生成基础可视化（如果 matplotlib 可用）
    try:
        import matplotlib.pyplot as plt
        # 设置中文字体，解决绘图乱码问题
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False


        # Bonding SC vs 期望概率
        if df["bonding_sc_idx"].notna().any() and df["expect_univ_bin"].notna().any():
            tmp = df[["bonding_sc_idx", "expect_univ_bin"]].dropna()
            tmp["bin"] = pd.qcut(tmp["bonding_sc_idx"], q=10, duplicates="drop")
            g = tmp.groupby("bin")["expect_univ_bin"].mean()
            plt.figure(figsize=(8, 4))
            g.plot(marker="o")
            plt.title("师生关系（Bonding SC）与升学期望概率")
            plt.ylabel("期望概率")
            plt.xlabel("Bonding SC 分位区间")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "bonding_vs_expect_prob.png")
            plt.close()

        # Bridging SC vs 期望概率
        if df["bridging_sc_idx"].notna().any() and df["expect_univ_bin"].notna().any():
            tmp = df[["bridging_sc_idx", "expect_univ_bin"]].dropna()
            tmp["bin"] = pd.qcut(tmp["bridging_sc_idx"], q=10, duplicates="drop")
            g = tmp.groupby("bin")["expect_univ_bin"].mean()
            plt.figure(figsize=(8, 4))
            g.plot(marker="o", color="orange")
            plt.title("同伴氛围（Bridging SC）与升学期望概率")
            plt.ylabel("期望概率")
            plt.xlabel("Bridging SC 分位区间")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "bridging_vs_expect_prob.png")
            plt.close()

        # SES 调节效应（低/高 SES）
        if df["ses_idx"].notna().any() and df["bonding_sc_idx"].notna().any() and df["expect_univ_bin"].notna().any():
            tmp = df[["ses_idx", "bonding_sc_idx", "expect_univ_bin"]].dropna()
            median = tmp["ses_idx"].median()
            tmp["ses_group"] = (tmp["ses_idx"] >= median).map({True: "高SES", False: "低SES"})
            lines = {}
            for grp, sub in tmp.groupby("ses_group"):
                sub["bin"] = pd.qcut(sub["bonding_sc_idx"], q=6, duplicates="drop")
                lines[grp] = sub.groupby("bin")["expect_univ_bin"].mean()
            plt.figure(figsize=(8, 4))
            for grp, series in lines.items():
                series.plot(marker="o", label=grp)
            plt.title("SES 调节下的师生关系-升学期望关系")
            plt.ylabel("期望概率")
            plt.xlabel("Bonding SC 分位区间")
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "bonding_expect_by_SES.png")
            plt.close()

        # 户籍调节效应（城市/农村）
        if df["hukou_cat"].notna().any() and df["bonding_sc_idx"].notna().any() and df["expect_univ_bin"].notna().any():
            tmp = df[["hukou_cat", "bonding_sc_idx", "expect_univ_bin"]].dropna()
            lines = {}
            for grp, sub in tmp.groupby("hukou_cat"):
                sub["bin"] = pd.qcut(sub["bonding_sc_idx"], q=6, duplicates="drop")
                lines[grp] = sub.groupby("bin")["expect_univ_bin"].mean()
            plt.figure(figsize=(8, 4))
            for grp, series in lines.items():
                series.plot(marker="o", label=grp)
            plt.title("户籍调节下的师生关系-升学期望关系")
            plt.ylabel("期望概率")
            plt.xlabel("Bonding SC 分位区间")
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "bonding_expect_by_hukou.png")
            plt.close()

        print(f"[DONE] 图表输出目录: {FIGURES_DIR}")
    except Exception as e:
        print(f"[WARN] 可视化跳过（建议安装 matplotlib）：{e}")

    # 文本报告
    lines = []
    lines.append("研究主题可视化与处理报告")
    lines.append("主题：学校社会资本对不同家庭背景学生教育期望的补偿效应")
    lines.append("")
    lines.append(f"Bonding SC 变量（存在的）: {', '.join(bonding_vars) if bonding_vars else '未找到'}")
    lines.append(f"Bridging SC 变量（存在的）: {', '.join(bridging_vars) if bridging_vars else '未找到'}")
    lines.append(f"SES 变量（存在的）: {', '.join(ses_vars) if ses_vars else '未找到'}")
    lines.append(f"户籍变量: {hukou_var or '未找到'}")
    lines.append(f"教育期望变量: {expect_var or '未找到'}")
    lines.append("")
    # 缺失率与样本量
    for v in ["bonding_sc_idx", "bridging_sc_idx", "ses_idx", "hukou_cat", "expect_univ_bin"]:
        if v in df.columns:
            na = df[v].isna().sum()
            lines.append(f"{v} 缺失: {na}（{na/len(df):.2%}）")
        else:
             lines.append(f"{v} 缺失: (变量不存在)")
    report_path = REPORT_DIR / "研究主题可视化与处理报告.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[DONE] 研究主题报告: {report_path}")


def gather_figures() -> None:
    """将工作目录下所有 png/jpg/jpeg 图片汇总到 figures 文件夹"""
    FIGURES_DIR = WORKSPACE / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg"}
    moved = 0
    for root, dirs, files in os.walk(WORKSPACE):
        if Path(root) == FIGURES_DIR:
            continue
        for fn in files:
            ext = Path(fn).suffix.lower()
            if ext in exts:
                src = Path(root) / fn
                dst = FIGURES_DIR / fn
                try:
                    # 名称冲突时加序号后缀
                    candidate = dst
                    i = 1
                    while candidate.exists() and candidate.read_bytes() != src.read_bytes():
                        candidate = FIGURES_DIR / f"{Path(fn).stem}_{i}{ext}"
                        i += 1
                    import shutil
                    if not candidate.exists():
                        shutil.copy2(src, candidate)
                        moved += 1
                except Exception as e:
                    print(f"[WARN] 复制图像失败: {src} -> {dst}: {e}")
    print(f"[DONE] 已汇总图像至: {FIGURES_DIR}，新增数量: {moved}")


def main():
    stats: List[Dict[str, object]] = []
    for k, p in DATA_PATHS.items():
        try:
            stats.append(clean_one_stats(k, p))
        except Exception as e:
            print(f"[ERROR] 处理 {k} 失败: {e}")
            
    write_overall_summary(stats)
    
    # 研究主题可视化与数据处理（生成 PNG）
    try:
        analyze_student_topic()
    except Exception as e:
         print(f"[ERROR] 分析主题失败: {e}")
         
    # 汇总所有图片到一个统一文件夹
    gather_figures()
    print(f"清洗完成。输出目录: {OUTPUT_DIR}; 报告目录: {REPORT_DIR}")


if __name__ == "__main__":
    main()
