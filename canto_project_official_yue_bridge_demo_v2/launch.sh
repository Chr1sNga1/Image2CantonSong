#!/usr/bin/env bash
set -euo pipefail
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yue_project_clean
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 127.0.0.1
