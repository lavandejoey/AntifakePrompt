#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status.
set -euo pipefail
# Enable debugging output
#set -x

export CUDA_VISIBLE_DEVICES=0
datetime="$(date '+%Y%m%d_%H%M%S')"
result_dir="results/${datetime}_Antifake_blip2"
mkdir -p "${result_dir}"
#data_root="/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"
data_root="/home/infres/ziyliu-24/data/FakeParts2DataMock"

source /home/infres/ziyliu-24/miniconda3/etc/profile.d/conda.sh
conda activate antifake310

python3 "AntifakeEval.py" \
    --data_root "${data_root}" \
    --pred_csv "${result_dir}/predictions.csv"
