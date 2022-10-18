#!/usr/bin/env bash

. /nesi/project/nesi99999/Dinindu/m-ex_bench/gpu_tmp_lmod_bootstrap.sh

module purge
module load Python/3.9.9-gimkl-2020a CUDA/11.3.1

. venv/bin/activate

# exit on errors, undefined variables and errors in pipes
set -euo pipefail

export DETECTRON2_DATASETS=/nesi/nobackup/nesi99999/riom/detectron2_datasets
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO

mkdir -p $RESULTS_DIR
cd $RESULTS_DIR

python ../../detectron2/tools/train_net.py \
    --config-file ../../detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml \
    $* 
