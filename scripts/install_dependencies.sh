#!/bin/bash

###########################################
# Setup Environment
###########################################
GIT_LFS_ERROR_MSG="${red}[AWRA] Please install git-lfs. Instructions found here: https://github.com/git-lfs/git-lfs/wiki/Installation${reset}"
ERROR_MSG="${red}[AWRA] Please try running with sudo or please install bunzip2, gcc, openmpi, openssh, git and git lfs to proceed${reset}"
red=$(tput setaf 1)
green=$(tput setaf 2)
reset=$(tput sgr0)



##########################################
# Functions
##########################################
# Openmpi setup
mpiSetup()
{
    if mpirun --version >/dev/null 2>&1; then
        echo "${green}[OPENMPI] Openmpi installed, module load does not need to be added.${reset}"
    else
        if grep "module load mpi" < "${HOME}/.bash_profile" >/dev/null 2>&1; then
            echo "${green}[OPENMPI] Module load MPI has already been added to .bash_profile file${reset}"
            source "${HOME}/.bash_profile" || { false && return; }
        else
            echo "${green}[OPENMPI] Adding module load mpi to .bash_profile${reset}"
            echo "module load mpi/openmpi-x86_64" >> "${HOME}/.bash_profile"
            source "${HOME}/.bash_profile" || { false && return; }
        fi
    fi
}

# Install dependencies
installDependencies()
{
    if bunzip2 --help >/dev/null 2>&1 && gcc --version >/dev/null 2>&1 && mpirun --version >/dev/null 2>&1 && git --version >/dev/null 2>&1 && git lfs --version >/dev/null 2>&1; then
        echo "${green}[AWRA] Dependencies have already been installed${reset}"
    else
        if apt-get --version >/dev/null 2>&1; then
            echo "${green}[AWRA] Downloading dependencies${reset}"
            bunzip2 --help >/dev/null 2>&1 || apt-get install bzip2 -y || { echo "${ERROR_MSG}"; exit 1; }
            gcc --version >/dev/null 2>&1 || apt-get install gcc -y || { echo "${ERROR_MSG}"; exit 1; }
            mpirun --version >/dev/null 2>&1 || apt-get install openssh-server openssh-client openmpi-bin openmpi-doc libopenmpi-dev  -y || { echo "${ERROR_MSG}"; exit 1; }
            git --version >/dev/null 2>&1 || apt-get install git -y || { echo "${ERROR_MSG}"; exit 1; }
            git lfs --version >/dev/null 2>&1 || apt-get install git-lfs -y || { echo "$GIT_LFS_ERROR_MSG"; exit 1; }
            source "${HOME}/.bash_profile"
            mpiSetup
        elif yum --version >/dev/null 2>&1; then
            echo "${green}[AWRA] Downloading dependencies${reset}"
            bunzip2 --help >/dev/null 2>&1 || yum install bzip2 -y || { echo "${ERROR_MSG}"; exit 1; }
            gcc --version >/dev/null 2>&1 || yum install gcc -y || { echo "${ERROR_MSG}"; exit 1; }
            mpirun --version >/dev/null 2>&1 || yum install openssh-server openssh-clients openmpi openmpi-devel -y || { echo "${ERROR_MSG}"; exit 1; }
            git --version >/dev/null 2>&1 || yum install git -y || { echo "${ERROR_MSG}"; exit 1; }
            git lfs --version >/dev/null 2>&1 || yum install git-lfs -y || { echo "${GIT_LFS_ERROR_MSG}"; exit 1; }
            source "${HOME}/.bash_profile"
            mpiSetup
        else
            echo "${red}[AWRA] Cannot install required dependencies: apt-get or yum does not exist${reset}"
            echo "${red}${ERROR_MSG}${reset}"
            exit 1
        fi
    fi
}

####################################
# Main
####################################
installDependencies
