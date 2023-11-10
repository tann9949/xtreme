#!/bin/bash

REPO=$PWD
MODEL=${1:-airesearch/wangchanberta-base-att-spm-uncased}
GPU=${2:-0}
DATASET_DIR=${3:-"$REPO/download/wangchanberta_dataset"}
OUTPUT_DIR=${4:-"$REPO/outputs/"}

TOKENIZER=$REPO/scripts/token-revamp/tokenizer
SEED=42
NUM_WORKERS=8
CACHE_DIR=$REPO/download/.cache
DATA_CACHE_DIR=$DATASET_DIR/cache

export CUDA_VISIBLE_DEVICES=$GPU

task="mlm_sft"
LR=1e-5
EPOCH=5
MAXL=412
BATCH_SIZE=8
GRAD_ACC=4
LANGS="th"
LC=""

SAVE_DIR="$OUTPUT_DIR/token_revamp/$task/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}/"
mkdir -p $SAVE_DIR

python $REPO/scripts/token-revamp/training/run_mlm_pt_with_sft.py \
    --output_dir=$OUTPUT_DIR \
    --do_train \
    --seed=$SEED \
    --do_eval \
    --cache_dir=$CACHE_DIR \
    --model_revision=main \
    --model_name_or_path=$MODEL \
    --config_name=$MODEL \
    --tokenizer_name_or_path=$TOKENIZER \
    --use_fast_tokenizer \
    --dataset_dir=$DATASET_DIR \
    --data_cache_dir=$DATA_CACHE_DIR \
    --max_seq_length=412 \
    --preprocessing_num_workers=$NUM_WORKERS \
    --validation_split_percentage=0.1

