# Method: Motion-aware Adaptive Windowing (Adaptive-Window)

## Goal
Standardize **wearable prodrome analysis** around AF onset in the presence of:
- variable timing of pre-onset changes (heterogeneity),
- motion artifacts (activity confounding),
- limited labeled events.

## Definitions
- **Onset index**: first sample where the AF annotation transitions to AF (e.g., `AF=1`) or transitional label (`AF=0.5`) if defined by the dataset.
- **Pre-onset candidate windows**: a small set of interpretable windows, for example:
  - W1: -20 → -10 minutes
  - W2: -15 → -10 minutes
  - W3: -10 → -5 minutes
  - W4 (sensitivity): -5 → -2 minutes (near-onset)
- **Control window**: a fixed baseline window far from onset (e.g., +10 → +20 minutes) or pseudo-onset baselines (see below).

## Motion-aware gating
For each candidate pre-onset window:
- compute a simple motion variability metric from accelerometer (e.g., STD on Acc X).
- optionally exclude windows exceeding a motion threshold, or apply a penalty term in scoring.

## Feature extraction (minimal, interpretable)
From PPG (after optional bandpass filtering):
- **PPG pulse amplitude variability** (e.g., STD of pulse amplitudes from detected peaks).
From peak-to-peak intervals (IBI) derived from PPG peaks:
- **IBI variability** (e.g., STD of IBI, RMSSD, pNN50).
From accelerometer:
- **Motion variability** (e.g., STD).

## Adaptive selection
Select the **best** pre-onset window per episode based on a conservative objective, e.g.:
- maximize a pre/control contrast metric while meeting motion criteria and minimum peak counts.

## Pseudo-onset baseline (controls)
For control recordings, create **K pseudo-onsets** uniformly over valid time to generate a baseline distribution.
This increases robustness when the number of distinct control subjects is small.

## Reporting
Always report:
- chosen window per episode,
- feature values in pre and control windows,
- motion metric,
- the scoring rule and any thresholds,
- sensitivity analyses (e.g., with/without motion gating, with/without ratio capping).

> Important: This framework is intended as a **methodology/software** contribution rather than a claim of universal clinical prediction.
