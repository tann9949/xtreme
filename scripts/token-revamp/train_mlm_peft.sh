#!/bin/bash

REPO=$PWD
MODEL=${1:-xlm-roberta-base}
GPU=${2:-0}
DATASET_DIR=${3:-"$REPO/download/wangchanberta_dataset"}
OUTPUT_DIR=${4:-"$REPO/outputs/"}

TOKENIZER=$REPO/scripts/token-revamp/tokenizer
SEED=42
NUM_WORKERS=16
CACHE_DIR=$REPO/download/.cache
DATA_CACHE_DIR=$DATASET_DIR/cache

export CUDA_VISIBLE_DEVICES=$GPU

task="mlm_peft"
LR=1e-5
EPOCH=5
MAXL=512
BATCH_SIZE=8
GRAD_ACC=4
LANGS="th"
LC=""

# wandb
wandb login
export WANDB_PROJECT="token_revamp-XLMR-base+wangchanberta-base-att-spm-uncased"
export WANDB_ENTITY=chompk
export WANDB_NAME="${task}_${MODEL}_LR${LR}_epoch${EPOCH}_MaxLen${MAXL}_Batch${BATCH_SIZE}_Acc${GRAD_ACC}_Seed${SEED}"

SAVE_DIR="$OUTPUT_DIR/token_revamp/$task/${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}/"
mkdir -p $SAVE_DIR

echo "Fine-tuning $MODEL on $TASK using GPU $GPU"
echo "Load data from $DATA_DIR, and save models to $OUTPUT_DIR"

python $REPO/scripts/token-revamp/training/run_mlm_pt.py \
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
    --block_size=$MAXL \
    --preprocessing_num_workers=$NUM_WORKERS \
    --validation_split_percentage=0.1 \
    --logging_strategy=steps \
    --logging_steps=5 \
    --save_strategy=steps \
    --save_steps=5000 \
    --eval_steps=5000 \
    --evaluation_strategy=steps \
    --learning_rate=$LR \
    --num_train_epochs=$EPOCH \
    --per_device_train_batch_size=$BATCH_SIZE \
    --per_device_eval_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$GRAD_ACC \
    --report_to=wandb \
    --overwrite_output_dir \
    --trainable="query,value" \
    --lora_rank=8 \
    --lora_dropout=0.1 \
    --lora_alpha=32.