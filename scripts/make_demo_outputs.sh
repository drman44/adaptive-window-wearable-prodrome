#!/usr/bin/env bash
set -euo pipefail

python -m afprodrome.run --data_dir data/raw/Data --out_dir results --k_pseudo 20 --margin_min 25 --fs_ann 12.8

python scripts/build_tables.py --results_dir results --out_dir submission_outputs
python scripts/make_figures.py --results_dir results --tables_dir submission_outputs/Tables --out_dir submission_outputs/Figures

echo "Done: submission_outputs/Tables and submission_outputs/Figures"
