#!/bin/bash
#SBATCH --job-name=run_sweep_id_3h
#SBATCH --account=aip-wolfg
#SBATCH --time=3:00:00
#SBATCH --mem=400G
#SBATCH -c 48
#SBATCH --gpus-per-node=h100:4
#SBATCH --chdir=/home/h/horoist/isoloco
#SBATCH --output=/home/h/horoist/isoloco/scripts/logs/%x-%j.log
#SBATCH --error=/home/h/horoist/isoloco/scripts/logs/%x-%j.log

set -Eeuo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: sbatch $0 <SWEEP_ID>"
  exit 1
fi

SWEEP_ID="$1"

module load httpproxy
source .venv/bin/activate

export SAVE_DIR="/home/h/horoist/links/scratch/isoloco_ckpts"
export DATA_DIR="/home/h/horoist/links/projects/aip-wolfg/horoist/datasets"
export OMP_NUM_THREADS=12

srun wandb agent "${SWEEP_ID}" --count 1
