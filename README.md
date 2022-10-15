# HGX A100 nodes test

Notes about tests on the HGX A100 nodes.


## Detectron2

Let's try to train a large enough model from Detectron2.

Install Detectron2 (for CUDA 11.3 and PyTorch 1.10)

```
module purge
module load Python/3.9.9-gimkl-2020a
python3 -m venv venv
venv/bin/python -m pip install detectron2 \
  -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
venv/bin/python -m pip install torch==1.10 torchvision \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

Get COCO 2017 train/val data

```
export DETECTRON2_DATASETS=/nesi/nobackup/nesi99999/riom/detectron2_datasets
mkdir -p $DETECTRON2_DATASETS
wget -P $DETECTRON2_DATASETS http://images.cocodataset.org/zips/val2017.zip
unzip $DETECTRON2_DATASETS/val2017.zip -d $DETECTRON2_DATASETS/coco/
rm $DETECTRON2_DATASETS/val2017.zip
wget -P $DETECTRON2_DATASETS http://images.cocodataset.org/zips/train2017.zip
unzip $DETECTRON2_DATASETS/train2017.zip -d $DETECTRON2_DATASETS/coco/
rm $DETECTRON2_DATASETS/train2017.zip
wget -P $DETECTRON2_DATASETS http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip $DETECTRON2_DATASETS/annotations_trainval2017.zip -d $DETECTRON2_DATASETS/coco/
rm $DETECTRON2_DATASETS/annotations_trainval2017.zip
```

Get detectron source code

```
git clone https://github.com/facebookresearch/detectron2
```

Log in a HGX node:

```
ssh wmg003.ib.hpcf.nesi.org.nz
```

Fit a model using 4 GPUs

```
src/test_4gpus.bash
```

Scripts to train on 1 GPU and 2 GPUs are also available.


## First results (2022/10/14)

First results (on wmg002), training R50-FPN for object detection on COCO 2017 using Detectron2

- 4 GPUs: AP 37.94, 4:05:27, 0.1636 s/it
- 2 GPUs: AP 37.88, 7:21:13, 0.2942 s/it
- 1 GPU: AP 38.34, 15:07:58, 0.6053 s/it

Reported performances in model zoo is box AP 37.9, so we get something very close. I am not sure if the differences are due to the random seed and/or some non-deterministic operations on GPU.

Reported train time in model zoo is 0.210 s/it, it was trained on 8 NVIDIA V100 GPUs & NVLink.

Potential next steps:

- compare with training on 2 PCIe A100s
- check if it is actually using NVlink (logs don’t say anything about it), try nccl backend maybe (if it is not the default)?
- try 8 GPUs training using 2 nodes (once slurm is in place?)

(2022/10/16) 2 PCIe A100s test: AP 37.92, 9:46:32, 0.3910 s)


## Todos

- plot GPU usage curves
- load NCCL module, but seems already in pytorch https://discuss.pytorch.org/t/pytorch-cuda-and-nccl/132427
- check/profile communication (nvlink, nccl?)
  see https://pytorch.org/docs/stable/distributed.html#profiling-collective-communication
- try AMP?
- compare with 4 PCIe A100 using MPI/NCCL?
- test 8 GPUs (2 HGX) using MPI/NCCL?


## Reference

- datasets https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md
- model training https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html
- model zoo https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
- benchmark https://detectron2.readthedocs.io/en/latest/notes/benchmarks.html
