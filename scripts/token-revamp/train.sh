#!/bin/bash

REPO=$PWD
MODEL=${1:-xlm-roberta-base}
TASK=${2:-mlm_sft}
GPU=${3:-0}
DATA_DIR=${4:-"$REPO/download/wangchanberta_dataset"}
OUT_DIR=${5:-"$REPO/outputs/"}
echo "Fine-tuning $MODEL on $TASK using GPU $GPU"
echo "Load data from $DATA_DIR, and save models to $OUT_DIR"

if [ $TASK == 'mlm_sft' ]; then
  bash $REPO/scripts/token-revamp/train_mlm_sft.sh $MODEL $GPU $DATA_DIR $OUT_DIR
fi