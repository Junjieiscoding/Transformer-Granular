#!/usr/bin/env python3
"""
step7_visualization.py — Granular‑ICU 研究 · 可视化总汇
=====================================================
• 功能
  1. 自动遍历  step6 生成的 SHAP 结果目录  →  `./output_dynamic/shap_analysis_<COHORT>`
  2. 若检测到同名  `*_shap_vals.npy`  与  `*_X_plot.parquet`  →  绘制 **SHAP summary plot (beeswarm)**
  3. 同时为每个模型输出 **Top‑N bar 图** (默认 20) —— bar 图可基于 `*_feature_importance.csv`
  4. 所有图片保存到  `./output_dynamic/step7_visualizations_<COHORT>/`，并写入 `README_visuals.txt`

• 运行示例
    python step7_visualization.py                # 处理全部模型
    python step7_visualization.py --top 30       # 修改 Top‑N
    python step7_visualization.py --models lstm transformer

"""
import argparse, os, glob, re, sys, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ╭──────────────────────────────────────────────────────────╮
# │ 用户可修改区域                                           │
# ╰──────────────────────────────────────────────────────────╯
COHORT_KEY = "aki"                   # 与 step6 保持一致
OUTPUT_ROOT = Path("./output_dynamic")
TOP_N_DEFAULT = 20                   # bar 图默认显示前 N 个特征
FONT_FAMILY = "DejaVu Sans"          # 统一字体（科研常用）

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 12,
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#222",
    "xtick.color": "#222",
    "ytick.color": "#222",
})

# ╭──────────────────────────────────────────────────────────╮
# │ 辅助函数                                                │
# ╰──────────────────────────────────────────────────────────╯

def make_output_dir(cohort: str) -> Path:
    out_dir = OUTPUT_ROOT / f"step7_visualizations_{cohort}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def draw_summary_plot(shap_vals_np: np.ndarray, X_df: pd.DataFrame,
                      out_png: Path, title: str):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_np, X_df, feature_names=X_df.columns,
                      show=False, color_bar=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def draw_bar_plot(csv_path: Path, top_n: int, out_png: Path, title: str):
    df = pd.read_csv(csv_path)
    df = df.nlargest(top_n, "importance")[::-1]  # 反转使条形从上到下递增
    plt.figure(figsize=(8, 0.35 * len(df) + 1))
    plt.barh(df["feature"], df["importance"], color="#3182bd")  # 高级蓝色
    plt.xlabel("Mean |SHAP value|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ╭──────────────────────────────────────────────────────────╮
# │ 主流程                                                  │
# ╰──────────────────────────────────────────────────────────╯

def main():
    parser = argparse.ArgumentParser(
        description="Generate SHAP visualizations (summary + bar) for each model.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--cohort", default=COHORT_KEY,
                        help="Cohort key, e.g., aki / hf (default: %(default)s)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specify subset of models to plot, e.g., lstm transformer")
    parser.add_argument("--top", type=int, default=TOP_N_DEFAULT,
                        help="Top‑N features for bar plot (default: %(default)s)")
    args = parser.parse_args()

    shap_dir = OUTPUT_ROOT / f"shap_analysis_{args.cohort}"
    if not shap_dir.exists():
        sys.exit(f"[Error] SHAP dir not found: {shap_dir}")

    out_dir = make_output_dir(args.cohort)
    readme_lines = []

    # 搜索 shap_vals.npy
    for npy_path in shap_dir.glob("*_shap_vals.npy"):
        m = re.match(r"(\w+)_" + re.escape(args.cohort) + r"_shap_vals\.npy", npy_path.name)
        if not m:
            continue
        model_tag = m.group(1)
        if args.models and model_tag not in args.models:
            continue

        # 配套文件
        x_path = shap_dir / f"{model_tag}_{args.cohort}_X_plot.parquet"
        csv_path = shap_dir / f"{model_tag}_{args.cohort}_feature_importance.csv"
        if not (x_path.exists() and csv_path.exists()):
            print(f"[Warn] Missing companion files for {model_tag}, skip.")
            continue

        print(f"→ Processing {model_tag.upper()} …")
        shap_vals = np.load(npy_path)
        X_df = pd.read_parquet(x_path)

        # ─ summary plot ─────────────────────────────────────
        sum_png = out_dir / f"{model_tag}_{args.cohort}_summary.png"
        draw_summary_plot(shap_vals, X_df, sum_png,
                          f"SHAP Summary Plot ({model_tag.upper()})")
        readme_lines.append(f"{sum_png.name}: SHAP summary (beeswarm) for {model_tag.upper()}")

        # ─ bar plot ─────────────────────────────────────────
        bar_png = out_dir / f"{model_tag}_{args.cohort}_top{args.top}_bar.png"
        draw_bar_plot(csv_path, args.top, bar_png,
                      f"Top‑{args.top} Features ({model_tag.upper()})")
        readme_lines.append(f"{bar_png.name}: Top‑{args.top} mean |SHAP| bar for {model_tag.upper()}")

    # 写 README
    if readme_lines:
        (out_dir / "README_visuals.txt").write_text("\n".join(readme_lines), encoding="utf‑8")
        print("\nAll images saved to", out_dir)
    else:
        print("No matching SHAP files found; nothing plotted.")


if __name__ == "__main__":
    main()
