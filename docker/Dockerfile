#
# AWRA CMS container - caches dependencies for the AWRA CMS Project
# 1. Conda 4.8.1
# 2. OpenMPI 3.1.3

FROM centos:centos7

LABEL maintainer="Subhashs Sharma"  email="awracms@bom.gov.au"

WORKDIR /root

ADD linux_conda_install_env.yml /

# ENV Variables
ARG MPI_VER=3.1.3
ENV MPI4PY_VERSION=3.0.0
ENV MPI_DL="https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-${MPI_VER}.tar.bz2"
ENV AWRAMS_BASE_PATH=/home/awracms
ENV INSTALL_PATH=${AWRAMS_BASE_PATH}

# Miniconda
RUN yum -y install sudo \
    && sudo yum -y update \
    && sudo yum -y install curl bzip2 \ 
    && sudo curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && sudo bash /tmp/miniconda.sh -bfp /usr/local/ \
    && sudo rm -rf /tmp/miniconda.sh 
  
# This is required to clone AWRA CMS data
RUN conda install git git-lfs -y && conda install jupyter -y --quiet && yum install which -y

# Install Conda packages
RUN conda env create -f /linux_conda_install_env.yml

# Intall GCC with C++
RUN yum clean all && yum update && yum install which make gcc-c++ curl -y

# Create a new user account without a password
RUN echo '%awracms ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && adduser awracms && passwd -d awracms
USER awracms
WORKDIR /home/awracms

# Install OpenMPI
ENV MPI_PATH="${INSTALL_PATH}/.openmpi"
ENV PATH="${MPI_PATH}/bin:${PATH}" 
ENV LD_LIBRARY_PATH="${MPI_PATH}/lib/:${LD_LIBRARY_PATH}" 

RUN mkdir -p .openmpi

RUN  echo "[AWRACMS] Downloading OpenMPI" && curl ${MPI_DL} -o ${INSTALL_PATH}/openmpi-${MPI_VER}.tar.bz2 \
    && echo "[AWRACMS] Extracting OpenMPI" \
    && tar -xvf ${INSTALL_PATH}/openmpi-${MPI_VER}.tar.bz2 \
    && echo "[AWRACMS] Deleting tarfile" \
    && rm "${INSTALL_PATH}/openmpi-${MPI_VER}.tar.bz2" \
    && echo "[AWRACMS] Changing directory" \
    && cd "${INSTALL_PATH}/openmpi-${MPI_VER}" \
    && echo "[AWRACMS] Configuring OpenMPI installation." \
    && ./configure --prefix="${MPI_PATH}" \
    && echo "[AWRACMS] Installing OpenMPI to ${HOME}/.openmpi" \
    && make install \
    && cd "${INSTALL_PATH}" \
    && echo "[AWRACMS] Deleting OpenMPI folder" \
    && [ -d "${INSTALL_PATH}/openmpi-${MPI_VER}" ] && rm -rf "${INSTALL_PATH}/openmpi-${MPI_VER}" 


############### AWRA specific ENV variables ############### 
# 1. Paths
ENV PATH /opt/conda/bin:$PATH
ENV AWRAMS_BASE_PATH=/home/awracms
ENV CONDA_ENV_NAME=awra-cms
ENV INSTALL_PATH=${AWRAMS_BASE_PATH}
ENV REPO_PATH=${INSTALL_PATH}/awrams_cm
ENV MINICONDA_PATH=/usr/local
ENV DATA_PATH=${INSTALL_PATH}/awrams_cm/awrams
# 2. Settings
ENV mpiInstalled=1
ENV AWRAMS_VERSION="AWRA_CMS_v1.2"
ENV CLONE_DATA=TRUE
# 3. Data source URLs
ENV AWRACMS_REPOSITORY="https://github.com/awracms/awra_cms.git"
ENV AWRACMS_DATA_REPOSITORY="https://github.com/awracms/awracms_data.git"

# awracms user to install AWRA 
USER awracms

# Activate bash
SHELL ["/bin/bash", "--login", "-c"]

# Activate conda environment
RUN echo "source activate ${CONDA_ENV_NAME}" > ~/.bashrc
ENV PATH /opt/conda/envs/${CONDA_ENV_NAME}/bin:$PATH

# Check if conda env in loaded and activated
RUN python -c "import pandas" && conda env list

# Install git lfs and clone AWRACMS project source code
RUN git lfs install && git clone -b "${AWRAMS_VERSION}" "${AWRACMS_REPOSITORY}" "${REPO_PATH}/"

# Clone data to run tests
RUN git clone "$AWRACMS_DATA_REPOSITORY" "${DATA_PATH}/data"

###############    Some cleanup work ############### 
# Remove .git folder from the container
RUN rm -rf "${DATA_PATH}/data/.git"


# ENV variables required to run AWRACMS
ENV PYTHONPATH="${PYTHONPATH}:/home/awracms/awrams_cm/awrams/code/user"
ENV AWRAMS_BASE_PATH="/home/awracms/awrams_cm/awrams"
ENV AWRAMS_DATA_PATH="${REPO_PATH}/awrams/data"

# Create Symlinks for openmpi
RUN for i in *; do alias "${i}"="/home/awracms/.openmpi/bin/${i}"; done

# Install rest of the AWRACMS dependencies from the install script
RUN curl -s https://raw.githubusercontent.com/awracms/awra_cms/%23225-containerise-awracms/docker/docker_install_dependencies.sh | bash
