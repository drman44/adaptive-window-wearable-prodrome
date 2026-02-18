from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def save_tiff(png_path: Path):
    Image.open(png_path).save(png_path.with_suffix(".tiff"), dpi=(300,300))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--tables_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    results = Path(args.results_dir)
    tables = Path(args.tables_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_case = pd.read_csv(results/"summary_case.csv")
    df_ctrl = pd.read_csv(tables/"TableS1_control_metrics.csv") if (tables/"TableS1_control_metrics.csv").exists() else pd.DataFrame()

    # Figure2
    plt.figure(constrained_layout=True)
    plt.scatter(np.arange(len(df_case)), df_case["ratio"].astype(float))
    if len(df_ctrl)>0 and df_ctrl["ratio_median"].notna().any():
        plt.scatter(np.arange(len(df_ctrl)) + len(df_case) + 2, df_ctrl["ratio_median"].astype(float))
    plt.xlabel("Cases / baseline participants")
    plt.ylabel("Amplitude ratio (pre/reference)")
    plt.title("QC-optimized window ratios: AF-labeled cases vs pseudo-onset baseline")
    f2 = out/"Figure2_ratio_cases_vs_controls.png"
    plt.savefig(f2, dpi=300)
    plt.close()
    save_tiff(f2)

    # Figure3
    plt.figure(constrained_layout=True)
    patients = df_case["patient"].astype(str).tolist() if "patient" in df_case.columns else [str(i) for i in range(len(df_case))]
    scores = df_case["prodrome_score"].astype(float).to_numpy()
    x = np.arange(len(patients))
    for i,s in enumerate(scores):
        if np.isfinite(s):
            plt.bar(i,s)
    plt.xticks(x, patients)
    plt.xlabel("Patient")
    plt.ylabel("Composite prodrome score")
    plt.title("Prodrome score by patient (QC fail-safe excludes non-computable cases)")
    for i,s in enumerate(scores):
        if not np.isfinite(s):
            plt.text(i, 0, "QC-excluded", ha="center", va="bottom", rotation=90)
    f3 = out/"Figure3_prodrome_score_by_patient.png"
    plt.savefig(f3, dpi=300)
    plt.close()
    save_tiff(f3)

    # Figure4
    if "motion_pre" in df_case.columns:
        plt.figure(constrained_layout=True)
        plt.scatter(df_case["motion_pre"].astype(float), df_case["ratio"].astype(float))
        plt.xlabel("Motion (std accX)")
        plt.ylabel("Amplitude ratio (pre/reference)")
        plt.title("Motion sensitivity: ratio vs motion")
        f4 = out/"Figure4_motion_vs_ratio.png"
        plt.savefig(f4, dpi=300)
        plt.close()
        save_tiff(f4)

    print("Wrote figures to", out)

if __name__ == "__main__":
    main()
