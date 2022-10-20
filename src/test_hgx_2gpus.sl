#!/usr/bin/env bash
#SBATCH --reservation=MAHUIKAEX-255
#SBATCH --account=nesi99999
#SBATCH --partition=hgx
#SBATCH --time=00-15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-task=A100:2
#SBATCH --mem=20GB
#SBATCH --output=logs/%j-%x.out
#SBATCH --error=logs/%j-%x.out
#SBATCH --export=NONE

export SLURM_EXPORT_ENV=ALL

# for MEX nodes
. /etc/bashrc

# exit on errors, undefined variables and errors in pipes
set -euo pipefail

module purge
module load Python/3.9.9-gimkl-2020a CUDA/11.3.1
. venv/bin/activate

RESULTS_DIR="$(pwd)/results/${SLURM_JOB_ID}-${SLURM_JOB_NAME}"
mkdir -p $RESULTS_DIR
cd $RESULTS_DIR

export DETECTRON2_DATASETS=/nesi/nobackup/nesi99999/riom/detectron2_datasets
export NCCL_DEBUG=INFO

MASTER_ADDR=$HOSTNAME
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

python ../../detectron2/tools/train_net.py \
    --config-file ../../detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml \
    --num-gpus 2 \
    --machine-rank $SLURM_PROCID
