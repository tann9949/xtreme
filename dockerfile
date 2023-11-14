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
    vim \
    ca-certificates \ 
    gnupg

# install nodejs
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | \
    gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    NODE_MAJOR=20 && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list  && \
    apt-get update && \
    apt-get install -y nodejs


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

# setup jupyterlab alias
RUN echo "alias jupyter-lab='jupyter-lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser'" >> /root/.bashrc

# set default conda env
RUN echo "conda activate xtreme" >> /root/.bashrc
