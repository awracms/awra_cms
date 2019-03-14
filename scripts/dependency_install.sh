#!/bin/bash
# Install conda if necessary, and create or update conda environment
set -e

wget --version || ( yum-config-manager && yum install wget -y )
bunzip2 --version || ( yum-config-manager && yum install bzip2 -y)
gcc --version || (yum install gcc -y)
git --version || (yum install git -y)

export PATH=../miniconda3/bin/:$PATH

if [ ! -f conda ]
then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh -b -p ../miniconda3
fi

if [ "$(conda env list | grep -c 'awra-cms')" -eq 0 ]
then
    ./conda_env.sh create
else
    ./conda_env.sh update
fi
