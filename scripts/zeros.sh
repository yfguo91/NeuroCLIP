#!/bin/bash
cd ..

# Path to dataset
#DATA=data/esimagenet
#DATASET=esimagenet
DATA=data/nmnist
DATASET=nmnist
#DATA=data/cifar10dvs
#DATASET=cifar10dvs

TRAINER=NeuroCLIP_ZS
# Trainer configs: rn50, rn101, vit_b32 or vit_b16
CFG=rn50

export CUDA_VISIBLE_DEVICES=6
python3 train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--no-train \
--zero-shot \