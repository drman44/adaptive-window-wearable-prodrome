from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    results = Path(args.results_dir)
    out = Path(args.out_dir)
    (out/"Tables").mkdir(parents=True, exist_ok=True)

    df_case = pd.read_csv(results/"summary_case.csv")
    df_ctrl = pd.read_csv(results/"summary_control.csv") if (results/"summary_control.csv").exists() else pd.DataFrame()

    t1 = pd.DataFrame([
        {
            "Group":"AF cases",
            "N": int(df_case["patient"].nunique()) if "patient" in df_case.columns else len(df_case),
            "Notes":"Onset defined as first AF=1 sample; if absent, first AF=0.5. ECG annotation index mapped to PPG index using fs_ann."
        },
        {
            "Group":"Label-negative baseline participants",
            "N": int(df_ctrl["ppg_file"].nunique()) if len(df_ctrl)>0 and "ppg_file" in df_ctrl.columns else 0,
            "Notes":"Participants without AF labels in this dataset; pseudo-onset sampling (K per participant; ±25-minute margin) used as baseline distribution (not guaranteed true negatives)."
        },
    ])
    t1.to_csv(out/"Tables/Table1_cohort_summary.csv", index=False)
    with pd.ExcelWriter(out/"Tables/Table1_cohort_summary.xlsx", engine="openpyxl") as w:
        t1.to_excel(w, index=False, sheet_name="Table1")

    keep = [
        "patient","onset_type","pre_window_min","window_name","window_rank",
        "snr","motion_pre","peaks_pre","amp_pre","amp_ctrl","ratio","log_ratio",
        "ibi_std","prodrome_score","missing_reason"
    ]
    cols = [c for c in keep if c in df_case.columns]
    t2 = df_case[cols].copy()
    t2.to_csv(out/"Tables/Table2_case_patient_level_metrics.csv", index=False)
    with pd.ExcelWriter(out/"Tables/Table2_case_patient_level_metrics.xlsx", engine="openpyxl") as w:
        t2.to_excel(w, index=False, sheet_name="Table2")

    if len(df_ctrl)>0:
        s1 = df_ctrl[["ppg_file","k_used","ratio_median","ratio_iqr_low","ratio_iqr_high"]].copy()
    else:
        s1 = pd.DataFrame(columns=["ppg_file","k_used","ratio_median","ratio_iqr_low","ratio_iqr_high"])
    s1.to_csv(out/"Tables/TableS1_control_metrics.csv", index=False)
    with pd.ExcelWriter(out/"Tables/TableS1_control_metrics.xlsx", engine="openpyxl") as w:
        s1.to_excel(w, index=False, sheet_name="TableS1")

    print("Wrote tables to", out/"Tables")

if __name__ == "__main__":
    main()
