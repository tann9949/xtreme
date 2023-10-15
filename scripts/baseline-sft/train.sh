#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
MODEL=${1:-xlm-roberta-base}
TASK=${2:-panx}
GPU=${3:-0}
DATA_DIR=${4:-"$REPO/download/"}
OUT_DIR=${5:-"$REPO/outputs/"}
echo "Fine-tuning $MODEL on $TASK using GPU $GPU"
echo "Load data from $DATA_DIR, and save models to $OUT_DIR"

if [ $TASK == 'xnli' ]; then
  bash $REPO/scripts/baseline-sft/train_xnli.sh $MODEL $GPU $DATA_DIR $OUT_DIR
elif [ $TASK == 'panx' ]; then
  bash $REPO/scripts/preprocess_panx.sh $MODEL $DATA_DIR
  bash $REPO/scripts/baseline-sft/train_panx.sh $MODEL $GPU $DATA_DIR $OUT_DIR
elif [ $TASK == 'tydiqa' ]; then
  bash $REPO/scripts/baseline-sft/train_qa.sh $MODEL tydiqa $TASK $GPU $DATA_DIR $OUT_DIR
fi
