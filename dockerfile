FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime


RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
    subversion \
    git \
    autoconf \
    libtool \
    wget \
    unzip \
    curl \
    vim
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# copy repository codes into docker image
COPY . .
# validate panx dataset
RUN bash scripts/validate_panx.sh

# update conda since 19.01 is an old version
RUN conda update -n base -c defaults conda

# install deps
RUN bash install_tools.sh

# set default conda env
RUN echo "conda activate xtreme" >> /root/.bashrc

