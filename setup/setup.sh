#!/bin/bash

set -eux

# get the directory of the script
script_dir=$(realpath "$(dirname "${0}")")

# define required package
REQUIRED_PKG="singularity"

# check if singularity is already installed
if [ -n "$(which $REQUIRED_PKG)" ]; then
  echo "$REQUIRED_PKG already installed"
else
  # if not, install singularity and its dependencies
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  echo "Installing dependencies..."
  sudo apt-get update && sudo apt-get install -y \
    build-essential \
    uuid-dev \
    libgpgme-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git \
    cryptsetup-bin

  # install go and singularity
  export VERSION=1.12 OS=linux ARCH=amd64 && \
    wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz && \
    sudo tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz && \
    rm go$VERSION.$OS-$ARCH.tar.gz
  wget https://github.com/apptainer/singularity/releases/download/v3.8.7/singularity-container_3.8.7_amd64.deb
  sudo dpkg -i singularity-container_3.8.7_amd64.deb
fi

# define model file path
retccl_model="$script_dir/best_ckpt.pth"
ctranspath="$script_dir/ctranspath.pth"
# check if the model file exists
if [ -f "$model" ]; then
  echo "Model file already exists."
else
  # if not, download the model file using gdown
  echo "Downloading model file..."
  pip install -U --no-cache-dir gdown --pre
  gdown https://drive.google.com/uc?id=1EOqdXSkIHg2Pcl3P8S4SGB5elDylw8O2 -O $retccl_model
  gdown "https://drive.google.com/u/0/uc?id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX&export=download" -O $ctranspath
fi

# build singularity container
sudo singularity build e2e_container.sif container.def

