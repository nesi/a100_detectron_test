# Detectron2-based A100 tests

Notes about tests on the A100 nodes using Detectron2.


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

- (2022/10/16, job 30961630) 2 PCIe A100s (1 task / 1 node): AP 37.92, 9:46:32, 0.3910 s/it
- (2022/10/19, job 30982571) 2 PCIe A100s (1 task / 1 node): AP 38.01, 10:27:39, 0.4184 s/it
- (2022/10/18, job 30975009) 2 PCIe A100s (2 tasks / 2 nodes): AP 37.88, 9:32:56, 0.3820 s/it
- (2022/10/18, job 30975012) 4 PCIe A100s (2 tasks / 2 nodes): AP 38.09, 5:47:41, 0.2318 s/it


## Test on 8 GPUs

Log in wmg002 and run

```
export RESULTS_DIR=results/test_8gpus
src/test_gpus.bash --num-gpus 4 --num-machines 2 --machine-rank 0 \
    --dist-url tcp://wmg002:25000 2>&1 | tee $RESULTS_DIR/fit_wmg002.log
```

Then log in wmg004 and run

```
export RESULTS_DIR=results/test_8gpus
src/test_gpus.bash --num-gpus 4 --num-machines 2 --machine-rank 1 \
    --dist-url tcp://wmg002:25000 2>&1 | tee $RESULTS_DIR/fit_wmg004.log
```

(2022/19/10) AP 37.97, 2:43:43, 0.1092 s/it


# Test HGX via Slurm

Some tests of HGX A100 using a Slurm job

- 31002046 - 2 GPUs (1 node): AP 37.94, 8:04:50 (0.3232 s/it)
- 31002077 - 3 GPUs (1 node): crashed with error
  ```
  AssertionError: Total batch size (16) must be divisible by the number of gpus (3).
  ```
- 31002070 - 4 GPUs (1 node): AP 38.07, 5:08:29 (0.2057 s/it)

Make sure to delete checkpointed models (once jobs are completed) to save space:

```
find results/ -name '*.pth' -exec rm {} \;
```


## Todos

- (test 4x1 HGX GPUs on 4 nodes)
- (test 2x1 HGX GPUs on 2 nodes)
- plot GPU usage curves
- load NCCL module, but seems already in pytorch https://discuss.pytorch.org/t/pytorch-cuda-and-nccl/132427
- check/profile communication (nvlink, nccl?)
  - https://pytorch.org/docs/stable/distributed.html#profiling-collective-communication
  - https://docs.rapids.ai/api/dask-cuda/nightly/examples/best-practices.html#accelerated-networking
- run dl-prof to check inter-GPU data transfer?
- try AMP?


## Reference

- datasets https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md
- model training https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html
- model zoo https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
- benchmark https://detectron2.readthedocs.io/en/latest/notes/benchmarks.html
