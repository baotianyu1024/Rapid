#!/bin/bash
set -euo pipefail
cd /mnt/hugui/longjiu.bty
export MPLCONFIGDIR=/tmp/rapid_mpl
python reproduction/scripts/placement_statistics.py
python reproduction/scripts/report.py
python reproduction/scripts/compare_paper.py
python reproduction/scripts/plot_paper_panels.py
python reproduction/scripts/plot_balance.py
python reproduction/scripts/compare_all_figures.py
python reproduction/scripts/final_findings.py
python reproduction/scripts/plot_final_deltas.py
python reproduction/scripts/plot_main_results.py
python reproduction/scripts/build_report_pdf.py
