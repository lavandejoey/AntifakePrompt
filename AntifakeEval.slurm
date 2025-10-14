#!/bin/bash
#SBATCH --job-name=AntifakeEval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=A100,L40S,A40
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# -------- shell hygiene --------
set -euo pipefail
umask 077
mkdir -p logs

# -------- print job header --------
echo "================= SLURM JOB START ================="
echo "Job:    $SLURM_JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Node:   ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-unknown}  (${SLURM_JOB_GPUS:-not-set})"
echo "Start:  $(date)"
echo "==================================================="

# -------- conda / modules --------
source /home/infres/ziyliu-24/miniconda3/etc/profile.d/conda.sh
conda activate antifake310
#module load cuda/12.4.1 ||  module load cuda/12.1

# -------- reproducibility & logging --------
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

# -------- CUDA / NCCL sanity --------
# Expose only the GPUs SLURM granted this job step
export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
# Reasonable defaults; flip IB if your fabric supports it
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
# If your cluster has no InfiniBand, uncomment:
# export NCCL_IB_DISABLE=1
# Optional perf knobs (tune as needed)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MALLOC_TRIM_THRESHOLD_=134217728   # 128MB
NP=${SLURM_GPUS_ON_NODE:-2}

# -------- paths / args (edit me) --------
PROJECT_ROOT="/home/infres/ziyliu-24/FakeParts2/Detectors/AntifakePrompt"
cd $PROJECT_ROOT || exit 1
DATA_ROOT="${PROJECT_ROOT}/data"

# -------- quick hardware + torch sanity --------
nvidia-smi -L || true
srun --gres=gpu:"${NP}" \
python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count():", torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, "No CUDA devices visible to this job."
import transformers
print(f"transformers.__version__: {transformers.__version__}")
try:
    import flash_attn
    print("flash_attn.__version__:", flash_attn.__version__)
except ImportError:
    print("flash_attn not installed")
PY

# -------- run the actual job --------
srun --gres=gpu:"${NP}" \
  python3 ${PROJECT_ROOT}/test.py \
    --real_dir "${DATA_ROOT}/0_real/" \
    --fake_dir "${DATA_ROOT}/1_fake/"

EXIT_CODE=$?

echo "================== SLURM JOB END =================="
echo "End:   $(date)"
echo "Exit:  ${EXIT_CODE}"
echo "==================================================="
exit "${EXIT_CODE}"