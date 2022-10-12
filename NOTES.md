# Notes

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

and using 1 GPU

```
TODO --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```


## Todos

- load NCCL module, but seems already in pytorch https://discuss.pytorch.org/t/pytorch-cuda-and-nccl/132427
- check/profile communication (nvlink, nccl?)
  see https://pytorch.org/docs/stable/distributed.html#profiling-collective-communication
- try AMP?
- compare with 2 PCIe A100
- compare with 1 A100 (PCIe or HGX?)
- compare with 4 cards using MPI/NCCL?


## Reference

- datasets https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md
- model training https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html
- model zoo https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
- benchmark https://detectron2.readthedocs.io/en/latest/notes/benchmarks.html
