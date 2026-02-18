# Adaptive Window Wearable Prodrome

Motion-aware adaptive windowing framework for standardized wearable prodrome analysis.

## Overview

This repository provides a reproducible software pipeline for detecting pre-event physiological shifts in wearable photoplethysmography (PPG) data.

The framework implements:

- QC-optimized adaptive window calibration  
- Motion-aware artifact penalization  
- Pseudo-onset control sampling  
- Fully scripted table and figure generation  

Designed for pilot atrial fibrillation (AF) prodrome research, the pipeline emphasizes reproducibility and transparency over statistical overfitting.

## Key Features

- Outcome-independent window calibration (QC/SNR-based)
- Subject-level pseudo-onset control baseline (K=20 default)
- Multimodal prodrome scoring (PPG amplitude, IBI variability, motion penalty)
- Fully scriptable outputs for tables and publication-ready figures

## Installation

```bash
pip install -r requirements.txt
