#!/usr/bin/env bash
#SBATCH --account=nesi99999
#SBATCH --partition=hgx
#SBATCH --time=00-10:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-task=A100:2
#SBATCH --mem=60GB
#SBATCH --output=logs/%A_%a-%x.out
#SBATCH --error=logs/%A_%a-%x.out
#SBATCH --array=0-4

# exit on errors, undefined variables and errors in pipes
set -euo pipefail

module purge
module load Python/3.9.9-gimkl-2020a CUDA/11.3.1
. venv/bin/activate

RESULTS_DIR="$(pwd)/results/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_NAME}"
mkdir -p $RESULTS_DIR
cd $RESULTS_DIR

export DETECTRON2_DATASETS=/nesi/nobackup/nesi99999/riom/detectron2_datasets
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO

MASTER_ADDR=$HOSTNAME
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export MASTER_URL="tcp://${MASTER_ADDR}:${MASTER_PORT}"

srun bash -c 'python ../../detectron2/tools/train_net.py \
    --config-file ../../detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml \
    --num-gpus 2 \
    --num-machines 2 \
    --machine-rank $SLURM_PROCID \
    --dist-url $(printenv MASTER_URL)'
