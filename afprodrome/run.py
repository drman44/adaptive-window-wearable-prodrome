from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd

from .adaptive_windows import default_candidate_windows, Window
from .io_hdf5 import estimate_fs, load_ecg_mat_v73, load_ppg_mat_v73
from .signals import bandpass, peak_indices, ibi_from_peaks


def find_onset_index(af: np.ndarray):
    idx1 = np.where(af == 1)[0]
    idx05 = np.where(af == 0.5)[0]
    if len(idx1) > 0:
        return int(idx1[0]), "AF=1"
    if len(idx05) > 0:
        return int(idx05[0]), "AF=0.5"
    return None, "none"


def window_to_samples(fs: float, start_min: float, end_min: float):
    s = int(round(fs * 60 * start_min))
    e = int(round(fs * 60 * end_min))
    return s, e


def safe_slice(x: np.ndarray, s: int, e: int):
    n = len(x)
    s2 = max(0, s)
    e2 = min(n, e)
    if e2 <= s2:
        return None
    return x[s2:e2]


def snr_proxy(ppg_seg: np.ndarray, fs: float) -> float:
    """Outcome-independent SNR proxy: std(bandpassed)/std(residual)."""
    if ppg_seg is None or len(ppg_seg) < int(fs * 10):
        return float("nan")
    raw = ppg_seg.astype(float)
    bp = bandpass(raw, fs)
    resid = raw - bp
    num = float(np.nanstd(bp))
    den = float(np.nanstd(resid)) + 1e-9
    return num / den


def amp_std_via_peaks(ppg_seg: np.ndarray, fs: float, min_peaks: int = 30):
    ppg_f = bandpass(ppg_seg, fs)
    peaks = peak_indices(ppg_f, fs, min_distance_sec=0.4)
    if len(peaks) < min_peaks:
        return np.nan, int(len(peaks)), np.array([], dtype=int)
    return float(np.nanstd(ppg_f[peaks])), int(len(peaks)), peaks


def ibi_std_from_peaks(peaks: np.ndarray, fs: float, min_intervals: int = 10) -> float:
    ibi = ibi_from_peaks(peaks, fs)
    if len(ibi) < min_intervals:
        return float("nan")
    return float(np.nanstd(ibi))


def analyze_episode(
    ppg_path: Path,
    onset_ppg_index: int,
    windows: list[Window],
    ctrl_window=(10, 20),
    motion_threshold: float | None = None,
    min_peaks: int = 30,
):
    p = load_ppg_mat_v73(ppg_path)
    fs = estimate_fs(p.ppg_green, p.ts)

    cs_rel, ce_rel = window_to_samples(fs, ctrl_window[0], ctrl_window[1])
    ctrl = safe_slice(p.ppg_green, onset_ppg_index + cs_rel, onset_ppg_index + ce_rel)
    if ctrl is None:
        return None

    amp_ctrl, nctrl, _ = amp_std_via_peaks(ctrl, fs, min_peaks=min_peaks)
    if not np.isfinite(amp_ctrl) or amp_ctrl <= 0:
        return None

    candidates = []
    for w in windows:
        s_rel, e_rel = window_to_samples(fs, w.start_min, w.end_min)
        pre = safe_slice(p.ppg_green, onset_ppg_index + s_rel, onset_ppg_index + e_rel)
        if pre is None:
            continue

        motion_pre = np.nan
        if p.acc_x is not None:
            acc_pre = safe_slice(p.acc_x, onset_ppg_index + s_rel, onset_ppg_index + e_rel)
            if acc_pre is not None and len(acc_pre) > 100:
                motion_pre = float(np.nanstd(acc_pre))

        if motion_threshold is not None and np.isfinite(motion_pre) and motion_pre > motion_threshold:
            continue

        amp_pre, npre, peaks = amp_std_via_peaks(pre, fs, min_peaks=min_peaks)
        if not np.isfinite(amp_pre):
            continue

        snr = snr_proxy(pre, fs)
        if not np.isfinite(snr):
            continue

        ibi_std = ibi_std_from_peaks(peaks, fs)

        candidates.append(
            dict(
                window_name=w.name,
                pre_window_min=f"{w.start_min}→{w.end_min}",
                s_rel=s_rel,
                e_rel=e_rel,
                snr=float(snr),
                motion_pre=motion_pre,
                peaks_pre=int(npre),
                amp_pre=float(amp_pre),
                ibi_std=float(ibi_std),
            )
        )

    if len(candidates) == 0:
        return None

    candidates = sorted(candidates, key=lambda d: d["snr"], reverse=True)
    best = candidates[0]
    best["window_rank"] = 0

    ratio = best["amp_pre"] / amp_ctrl if amp_ctrl > 0 else np.nan

    out = dict(
        ppg_file=ppg_path.name,
        fs_ppg=float(fs),
        ppg_index=int(onset_ppg_index),
        ctrl_window_min=f"{ctrl_window[0]}→{ctrl_window[1]}",
        amp_ctrl=float(amp_ctrl),
        peaks_ctrl=int(nctrl),
        ratio=float(ratio),
        log_ratio=float(np.log(ratio)) if np.isfinite(ratio) and ratio > 0 else np.nan,
        **best,
    )
    return out


def compute_prodrome_score(df: pd.DataFrame, w_ratio=1.0, w_ibi=0.7, w_motion=-0.5):
    def z(col):
        x = df[col].astype(float).to_numpy()
        mu = np.nanmean(x)
        sd = np.nanstd(x) + 1e-9
        return (x - mu) / sd

    df = df.copy()
    df["z_ratio"] = z("ratio")
    df["z_ibi"] = z("ibi_std")
    df["z_motion"] = z("motion_pre")
    df["prodrome_score"] = w_ratio * df["z_ratio"] + w_ibi * df["z_ibi"] + w_motion * df["z_motion"]
    return df


def analyze_controls(ppg_paths, k_pseudo: int, windows, ctrl_window=(10, 20), seed: int = 42, margin_min: int = 25):
    rng = np.random.default_rng(seed)
    rows = []
    for ppg_path in ppg_paths:
        p = load_ppg_mat_v73(ppg_path)
        fs = estimate_fs(p.ppg_green, p.ts)
        n = len(p.ppg_green)

        margin = int(fs * 60 * margin_min)
        if n <= 2 * margin:
            continue

        onsets = rng.integers(margin, n - margin, size=k_pseudo)
        ratios = []
        for onset in onsets:
            best = analyze_episode(ppg_path, int(onset), windows, ctrl_window=ctrl_window)
            if best is None or not np.isfinite(best.get("ratio", np.nan)):
                continue
            ratios.append(best["ratio"])

        if len(ratios) == 0:
            continue

        rows.append(
            dict(
                ppg_file=ppg_path.name,
                k_used=int(len(ratios)),
                ratio_median=float(np.nanmedian(ratios)),
                ratio_iqr_low=float(np.nanpercentile(ratios, 25)),
                ratio_iqr_high=float(np.nanpercentile(ratios, 75)),
            )
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--k_pseudo", type=int, default=20)
    ap.add_argument("--margin_min", type=int, default=25)
    ap.add_argument("--fs_ann", type=float, default=12.8)
    ap.add_argument("--min_peaks", type=int, default=30)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ecg_files = sorted(data_dir.glob("*_ECG_*.mat"))
    ppg_files = sorted(data_dir.glob("*_PPG_*.mat"))

    case_rows = []
    for ecg_fp in ecg_files:
        rec = load_ecg_mat_v73(ecg_fp)
        onset, onset_type = find_onset_index(rec.af)
        if onset is None:
            continue

        patient = ecg_fp.name[:2]
        ppg_candidates = [p for p in ppg_files if p.name.startswith(patient + "_PPG")]
        if not ppg_candidates:
            continue

        ppg_fp = ppg_candidates[0]
        p = load_ppg_mat_v73(ppg_fp)
        fs_ppg = estimate_fs(p.ppg_green, p.ts)

        onset_ppg_index = int(round(onset / float(args.fs_ann) * fs_ppg))

        best = analyze_episode(ppg_fp, onset_ppg_index, default_candidate_windows(), min_peaks=args.min_peaks)
        if best is None:
            continue
        best.update(dict(patient=patient, onset_type=onset_type, group="CASE"))
        case_rows.append(best)

    df_case = pd.DataFrame(case_rows)
    if len(df_case) > 0:
        df_case = compute_prodrome_score(df_case)

    def miss_reason(row):
        reasons = []
        if not np.isfinite(row.get("motion_pre", np.nan)):
            reasons.append("missing_acc_or_motion")
        if not np.isfinite(row.get("ibi_std", np.nan)):
            reasons.append("insufficient_valid_IBI")
        if not np.isfinite(row.get("prodrome_score", np.nan)):
            reasons.append("score_not_computable")
        return ";".join(sorted(set(reasons))) if reasons else "none"

    if len(df_case) > 0:
        df_case["missing_reason"] = df_case.apply(miss_reason, axis=1)

    df_ctrl = analyze_controls(ppg_files, k_pseudo=args.k_pseudo, windows=default_candidate_windows(), margin_min=args.margin_min)
    if len(df_ctrl) > 0:
        df_ctrl["group"] = "CONTROL"

    df_case.to_csv(out_dir / "summary_case.csv", index=False)
    df_ctrl.to_csv(out_dir / "summary_control.csv", index=False)

    run_cfg = dict(
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        k_pseudo=int(args.k_pseudo),
        margin_min=int(args.margin_min),
        fs_ann=float(args.fs_ann),
        min_peaks=int(args.min_peaks),
        calibration="Outcome-independent SNR proxy under QC constraints; ratio computed after selection",
    )
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    print("Done. CASE rows:", len(df_case), "CONTROL rows:", len(df_ctrl))


if __name__ == "__main__":
    main()
