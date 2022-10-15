#!/usr/bin/env bash
#SBATCH --account=nesi99999
#SBATCH --time=00-15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=A100:2
#SBATCH --mem=20GB
#SBATCH --output logs/%j-%x.out
#SBATCH --error logs/%j-%x.out
#SBATCH --export=NONE

export SLURM_EXPORT_ENV=ALL

# exit on errors, undefined variables and errors in pipes
set -euo pipefail

module purge
module load Python/3.9.9-gimkl-2020a CUDA/11.3.1
. venv/bin/activate

RESULTS_DIR="$(pwd)/results/${SLURM_JOB_ID}-${SLURM_JOB_NAME}"
mkdir -p $RESULTS_DIR

STATS_INTERVAL=5
nvidia-smi --query-gpu=timestamp,uuid,clocks_throttle_reasons.sw_thermal_slowdown,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.sm \
    --format=csv,nounits \
    -l "$STATS_INTERVAL" \
    -i "$CUDA_VISIBLE_DEVICES" \
    -f "${RESULTS_DIR}/gpu_stats.csv" &
NVIDIA_SMI_PID=$!

sleep 20

export DETECTRON2_DATASETS=/nesi/nobackup/nesi99999/riom/detectron2_datasets
cd $RESULTS_DIR
{ time python ../../detectron2/tools/train_net.py \
    --config-file ../../detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml \
    --num-gpus 2 ; } > fit.log 2>&1

kill $NVIDIA_SMI_PID
