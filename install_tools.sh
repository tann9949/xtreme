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

set -eux  # for easier debugging

REPO=$PWD
LIB=$REPO/third_party
mkdir -p $LIB

# install conda env
conda create --name xtreme --file conda-env.txt
conda init bash

# If 'conda activate' fails below, try uncommenting the following lines,
# based on https://github.com/conda/conda/issues/7980.
CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh

conda activate xtreme

# install latest transformer
pip install -U transformers
# cd $LIB
# # clone if not exists
# if [ ! -d transformers ]; then
#   git clone https://github.com/huggingface/transformers
# fi

# cd transformers
# git checkout cefd51c50cc08be8146c1151544495968ce8f2ad
# pip install .
# cd $LIB

pip install seqeval
pip install tensorboardx

# install XLM tokenizer
pip install sacremoses
pip install pythainlp
pip install jieba

# clone if not exits
if [ ! -d kytea ]; then
  git clone https://github.com/neubig/kytea.git
fi

cd kytea
autoreconf -i
./configure --prefix=${CONDA_PREFIX}
make && make install
pip install kytea

# install pytorch/seqeval
pip install torch torchvision seqeval datasets peft
pip install protobuf==3.20
pip install wandb

# make sure jupyterlab is usable
pip install -U jupyter jupyterlab ipython jupyter_client traitlets nbformat nbconvert ipywidgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager