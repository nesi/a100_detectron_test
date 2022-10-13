#!/usr/bin/env bash

. /nesi/project/nesi99999/Dinindu/m-ex_bench/gpu_tmp_lmod_bootstrap.sh

module purge
module load Python/3.9.9-gimkl-2020a CUDA/11.3.1

. venv/bin/activate

export DETECTRON2_DATASETS=/nesi/nobackup/nesi99999/riom/detectron2_datasets

export RESULTS_DIR=results/test_1gpu
mkdir -p $RESULTS_DIR
cd $RESULTS_DIR

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="3"

STATS_INTERVAL=5

nvidia-smi --query-gpu=timestamp,uuid,clocks_throttle_reasons.sw_thermal_slowdown,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.sm \
    --format=csv,nounits \
    -l "$STATS_INTERVAL" \
    -i "$CUDA_VISIBLE_DEVICES" \
    -f gpu_stats.csv &
NVIDIA_SMI_PID=$!

sleep 20

{ time python ../../detectron2/tools/train_net.py \
    --config-file ../../detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml \
    --num-gpus 1 ; } > fit.log 2>&1

kill $NVIDIA_SMI_PID
