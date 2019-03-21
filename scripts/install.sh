#!/bin/bash

# ------------------------------------------------------------
# Setup Environment
# ------------------------------------------------------------
red="$(tput setaf 1)"
green="$(tput setaf 2)"
reset="$(tput sgr0)"

# Check if the  AWRAMS_INSTALL_PATH variable is set
if [[ -z "${AWRAMS_INSTALL_PATH}" ]]; then
    INSTALL_PATH="${PWD}"
    echo "${green}[AWRACMS] Installing to ${INSTALL_PATH}${reset}"
else
    INSTALL_PATH="${AWRAMS_INSTALL_PATH}"
    if [[ -d "${INSTALL_PATH}" ]]; then
        echo "${green}[AWRACMS] Directory found${reset}"
    else
        echo "${green}[AWRACMS] Directory not found${reset}"
        echo "${green}[AWRACMS] Creating directory ${INSTALL_PATH}${reset}"
        if mkdir -p "${INSTALL_PATH}" >/dev/null 2>&1; then
            echo "${green}[AWRACMS] Successfully created directory ${INSTALL_PATH}${reset}"
        else
            echo "${red}[AWRACMS] Creating directory failed. Change folder permissions or install to a different directory${reset}"
            exit 1
        fi
    fi
    echo "${green}[AWRACMS] Installing to ${INSTALL_PATH}${reset}"
fi

# Does CONDA_PATH exist? If not does conda command exist? If one of the two exist source the conda.sh file to load the variables.
if [[ -n "${CONDA_PATH}" ]]; then
    if [[ -d "${CONDA_PATH}/etc/profile.d" ]]; then
        . "${CONDA_PATH}/etc/profile.d/conda.sh"
    else
        echo "${red}[AWRACMS] CONDA_PATH is incorrect${reset}"
        exit 1
    fi
elif command -v conda; then
    CONDA_PATH=$(which conda)
    CONDA_PATH=$(dirname "${CONDA_PATH}")
    CONDA_PATH=$(dirname "${CONDA_PATH}")
    . "${CONDA_PATH}/etc/profile.d/conda.sh"
else
    echo "${green}[AWRACMS] CONDA_PATH was not found or conda was not found in PATH. Miniconda will be installed${reset}"
fi

# Set MPI Install Path
if [[ -z "${OPENMPI_PATH}" ]]; then
    MPI_PATH="${HOME}/.openmpi"
    echo "${green}[AWRACMS] Installing to ${MPI_PATH}${reset}"
else
    MPI_PATH="${OPENMPI_PATH}"
    if [[ -d "${MPI_PATH}" ]]; then
        echo "${green}[AWRACMS] Directory found${reset}"
    else
        echo "${green}[AWRACMS] Directory not found${reset}"
        echo "${green}[AWRACMS] Creating directory ${MPI_PATH}${reset}"
        if mkdir -p "${MPI_PATH}" >/dev/null 2>&1; then
            echo "${green}[AWRACMS] Successfully created directory ${MPI_PATH}${reset}"
        else
            echo "${red}[AWRACMS] Creating directory failed. Change folder permissions or install to a different directory${reset}"
            exit 1
        fi
    fi
    echo "${green}[AWRACMS] Installing to ${MPI_PATH}${reset}"
fi

# Clone Data Path
if [[ -z "${CLONE_DATA_PATH}" ]]; then
    DATA_PATH="${INSTALL_PATH}/awrams_cm/awrams"
    echo "${green}[AWRACMS] Data clone path: ${DATA_PATH}${reset}"
else
    DATA_PATH="${CLONE_DATA_PATH}/data"
    if [[ -d "${CLONE_DATA_PATH}" ]]; then
        echo "${green}[AWRACMS] Directory found${reset}"
    else
        echo "${green}[AWRACMS] Directory not found${reset}"
        echo "${green}[AWRACMS] Creating directory ${CLONE_DATA_PATH}${reset}"
        if mkdir -p "${CLONE_DATA_PATH}" >/dev/null 2>&1; then
            echo "${green}[AWRACMS] Successfully created directory ${CLONE_DATA_PATH}${reset}"
            altDataPath=1
        else
            echo "${red}[AWRACMS] Creating directory failed. Change folder permissions or install to a different directory${reset}"
            exit 1
        fi
    fi
    echo "${green}[AWRACMS] Installing to ${CLONE_DATA_PATH}${reset}"
fi

# Version of AWRACMS
if [[ -z "${AWRAMS_VERSION}" ]]; then
    AWRAMS_VERSION="AWRA_CMS_v1.2"
    echo "${green}[AWRACMS] Version: ${AWRAMS_VERSION}${reset}"
else
    echo "${green}[AWRACMS] Version: ${AWRAMS_VERSION}${reset}"
fi

# Version of MPI4PY
if [[ -z "${MPI4PY_VERSION}" ]]; then
    MPI4PY_VERSION=3.0.0
fi

# Version of OpenMPI
if [[ -n ${OPENMPI_VERSION} ]]; then
    MPI_VER="${OPENMPI_VERSION}"
else
    MPI_VER=3.1.3
fi

REPO_PATH="${INSTALL_PATH}/awrams_cm"
CONDA_ENV="linux_conda_install_env.yml"
MPI_DL="https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-${MPI_VER}.tar.bz2"
MINICONDA_DL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
AWRACMS_REPOSITORY="https://github.com/awracms/awra_cms.git"
AWRACMS_DATA_REPOSITORY="https://github.com/awracms/awracms_data.git"
AWRACMS_REPOSITORY_SSH="git@github.com:awracms/awra_cms.git"
AWRACMS_DATA_REPOSITORY_SSH="git@github.com:awracms/awracms_data.git"
ERROR_MSG="${red}[AWRACMS] Try running with sudo or install bunzip2 and build essentials to proceed${reset}"
GIT_LFS_ERROR_MSG="${red}[AWRACMS] Install git-lfs. Instructions found here: https://github.com/git-lfs/git-lfs/wiki/Installation${reset}"

if [[ "${FROM_SCRATCH}" == [Tt][Rr][Uu][Ee] ]] && [[ "${NONINTERACTIVE}" == [Tt][Rr][Uu][Ee] ]]; then
    rm -rf ${REPO_PATH}
    rm -rf ${INSTALL_PATH}/miniconda3
elif [[ "${FROM_SCRATCH}" == [Tt][Rr][Uu][Ee] ]]; then
    echo "${red}[AWRACMS] It looks like you've set the FROM_SCRATCH variable to true${reset}"
    read -p "${red}[AWRACMS] Would you like to delete the awrams_cm folder?${reset} (y/n): " fromScratchA
    if [[ "${fromScratchA}" == [Yy] ]]; then
       rm -rf ${REPO_PATH}
    fi
    read -p "${red}[AWRACMS] Would you like to delete the miniconda3 folder?${reset} (y/n): " fromScratchB
    if [[ "${fromScratchA}" == [Yy] ]]; then
       rm -rf ${INSTALL_PATH}/miniconda3
    fi
fi

# ------------------------------------------------------------
# Functions
# ------------------------------------------------------------

# Check if user is running with sudo
sudoCheck() {
    if [[ "${HOME}" == "/root" ]]; then
        echo "${red}[SUDO CHECK] Running with sudo/root. Use sudo with 'sudo -E' or change users${reset}"
        exit 1
    elif [[ -z "${SUDO_USER}" ]]; then
        echo "${green}[SUDO CHECK] Not running with sudo. Using ${HOME} as home directory${reset}"
        isSudo=0
    else
        echo "${green}[SUDO CHECK] Running with sudo. Using ${HOME} as home directory${reset}"
        isSudo=1
    fi
}

# Install dependencies
installDependencies() {
    # Checking if environment variable INSTALL_DEPENDENCIES is set to false. Not case sensitive.
    if [[ "${INSTALL_DEPENDENCIES}" == [Ff][Aa][Ll][Ss][Ee] ]]; then
        echo "${green}[AWRACMS] INSTALL_DEPENDENCIES is set to false. Skipping dependency Installation${reset}"
        sleep 5
    elif command -v bzip2 >/dev/null 2>&1 && command -v gcc >/dev/null 2>&1 && command -v cpp >/dev/null 2>&1 && command -v make >/dev/null 2>&1; then
        echo "${green}[AWRACMS] Dependencies have already been installed${reset}"
    else
        if apt-get --version >/dev/null 2>&1; then
            echo "${green}[AWRACMS] Downloading dependencies${reset}"
            command -v bzip2 >/dev/null 2>&1 || apt-get install bzip2 -y || {
                echo "${ERROR_MSG}"
                exit 1
            }
            command -v gfortran >/dev/null 2>&1 || apt-get install build-essential -y || {
                echo "${ERROR_MSG}"
                exit 1
            }
            . "${HOME}/.bashrc"
        elif yum --version >/dev/null 2>&1; then
            echo "${green}[AWRACMS] Downloading dependencies${reset}"
            command -v bzip2 >/dev/null 2>&1 || yum install bzip2 -y || {
                echo "${ERROR_MSG}"
                exit 1
            }
            command -v gfortran >/dev/null 2>&1 || yum install make automake gcc gcc-c++ kernel-devel -y || {
                echo "${ERROR_MSG}"
                exit 1
            }
            . "${HOME}/.bashrc"
        else
            echo "${red}[AWRACMS] Cannot install required dependencies: apt-get or yum does not exist${reset}"
            echo "${red}${ERROR_MSG}${reset}"
            exit 1
        fi
    fi
}

# Install OpenMPI
installMPI() {
    if [[ OPENMPI_INSTALL == [Ff][Aa][Ll][Ss][Ee] ]]; then
        moduleLoading=1
        echo "${green}[AWRACMS] OPENMPI_INSTALL has been set to false. Loading module as an alternative${reset}"
    elif mpirun --version | grep "${MPI_VER}"; then
        echo "${green}[AWRACMS] Correct version of MPI installed${reset}"
        if command -v mpicc; then
            echo "${green}[AWRACMS] MPICC has been found${reset}"
            mpiInstalled=1
        else
            echo "${green}[AWRACMS] MPICC has not been found${reset}"
        fi
    else
        if [[ -d "${MPI_PATH}" ]]; then
            oldMPI=0
            "${MPI_PATH}/bin/mpirun" --version | grep "${MPI_VER}" || { oldMPI=1; }
            mpiInstalled=1
        fi

        if [[ "${oldMPI}" == 0 ]]; then
            echo "${green}[AWRACMS] Correct version of openmpi found${reset}"
        else
            if [[ -d "${MPI_PATH}" ]]; then
                echo "${green}[AWRACMS] .openmpi folder exists but is the wrong version. Moving to .openmpi_old${reset}"
                mv "${MPI_PATH}" "${HOME}/.openmpi_old"
            fi
            echo "${green}[AWRACMS] Downloading OpenMPI${reset}"
            curl "${MPI_DL}" -o "openmpi-${MPI_VER}.tar.bz2" || wget "${MPI_DL}"
            echo "${green}[AWRACMS] Extracting OpenMPI${reset}"
            tar -xvf "${INSTALL_PATH}/openmpi-${MPI_VER}.tar.bz2" || { false && return; }
            echo "${green}[AWRACMS] Deleting tarfile${reset}"
            [ -f "${INSTALL_PATH}/openmpi-${MPI_VER}.tar.bz2" ] && rm "${INSTALL_PATH}/openmpi-${MPI_VER}.tar.bz2" || { false && return; }
            echo "${green}[AWRACMS] Changing directory${reset}"
            cd "${INSTALL_PATH}/openmpi-${MPI_VER}" || { false && return; }
            echo "${green}[AWRACMS] Configuring OpenMPI installation.${reset}"
            ./configure --prefix="${MPI_PATH}" || { false && return; }
            echo "${green}[AWRACMS] Installing OpenMPI to ${HOME}/.openmpi${reset}"
            make install || { false && return; }
            cd "${INSTALL_PATH}"
            echo "${green}[AWRACMS] Deleting OpenMPI folder${reset}"
            [ -d "${INSTALL_PATH}/openmpi-${MPI_VER}" ] && rm -rf "${INSTALL_PATH}/openmpi-${MPI_VER}" || { false && return; }
            export PATH="${MPI_PATH}/bin:${PATH}"
            export LD_LIBRARY_PATH="${MPI_PATH}/lib/:${LD_LIBRARY_PATH}"
            if command -v mpirun && command -v mpicc; then
                mpiInstalled=1
            else
                mpiInstalled=0
            fi
        fi
    fi
}

# Load Modules
moduleLoad() {
    if [[ "${moduleLoading}" == 1 ]]; then
        . "${HOME}/.bashrc"
        if command -v mpirun >/dev/null 2>&1 && command -v mpicc >/dev/null 2>&1; then
            echo "${green}[MODULE] OpenMPI has been installed, module load does not need to be added.${reset}"
            mpiInstalled=1
        else
            echo "${green}[MODULE] Finding the appropriate OpenMPI Module${reset}"
            lines="$(module avail -t |& grep "openmpi")"
            while read line; do
                module load "${line}"
                command -v mpirun && command -v mpicc || {
                    mpiInstalled=0
                    module unload "${line}"
                    continue
                }
                echo "${green}[MODULE] OpenMPI module has been loaded${reset}"
                mpiModule="${line}"
                mpiInstalled=1
            done <<<"${lines}"
        fi
    else
        echo "${green}[MODULE] Module loading is not required${reset}"
    fi
}

# Conda installation
condaInstall() {
    echo "${green}[CONDA] Checking if miniconda is installed${reset}"
    if [[ -n ${CONDA_PATH} ]]; then
        . "${CONDA_PATH}/bin/activate" base || { false && return; }
        echo "${green}[CONDA] Miniconda has been found${reset}"
        MINICONDA_PATH="${CONDA_PATH}"
    elif [[ -e "${INSTALL_PATH}/miniconda3" ]]; then
        echo "${green}[CONDA] Miniconda has already been installed.${reset}"
        . "${INSTALL_PATH}/miniconda3/etc/profile.d/conda.sh"
        . "${INSTALL_PATH}/miniconda3/bin/activate" base || { false && return; }
        MINICONDA_PATH="${INSTALL_PATH}/miniconda3"
    else
        echo "${green}[CONDA] Installing miniconda${reset}"
        curl "$MINICONDA_DL" -o miniconda.sh || wget "$MINICONDA_DL" -o miniconda.sh || { false && return; }
        if [[ -e miniconda.sh ]]; then
            echo "${green}[CONDA] Miniconda.sh has been found${reset}"
        else
            false && return
        fi
        chmod +x "${INSTALL_PATH}/miniconda.sh" || { false && return; }
        "${INSTALL_PATH}/miniconda.sh" -b -p "${INSTALL_PATH}/miniconda3" || { false && return; }
        . "${INSTALL_PATH}/miniconda3/bin/activate" || { false && return; }
        echo "${green}[CONDA] Miniconda has been installed${reset}"
        MINICONDA_PATH="${INSTALL_PATH}/miniconda3"
        MINICONDA_INSTALLED=1
        [ -f "${INSTALL_PATH}/miniconda.sh" ] && rm "${INSTALL_PATH}/miniconda.sh"
    fi
    echo "${green}[CONDA] Installing base dependencies${reset}"
    conda install git git-lfs -y || { false && return; }

}

# Clone awracms
awraClone() {
    echo "${green}[GIT] Cloning AWRACMS to $REPO_PATH${reset}"
    if [ -d "${REPO_PATH}/awrams" ] && [[ "${NONINTERACTIVE}" == [Tt][Rr][Uu][Ee] ]]; then
        git reset --hard "origin/${AWRAMS_VERSION}" || echo "${red}Failed to clone, proceeding anyway${reset}"
        git pull origin "${AWRAMS_VERSION}" || echo "${red}Failed to clone, proceeding anyway${reset}"
    elif [ -d "${REPO_PATH}" ]; then
        echo "${red}WARNING! ANY CHANGES MADE TO THE FILES WITHIN AWRAMS_CM WILL BE RESET${reset}"
        read -p "${green}[AWRACMS] You have already cloned AWRACMS. Would you like to download the latest changes? [Yes/No]: ${reset} " awraExists
        if [[ "${awraExists}" == [Yy][Ee][Ss] ]]; then
            echo "${green}[AWRACMS] Resetting head and pulling latest changes${reset}"
            git -C "${REPO_PATH}" reset --hard "origin/${AWRAMS_VERSION}" || echo "${red}Failed to clone, proceeding anyway${reset}"
            git -C "${REPO_PATH}" pull origin "${AWRAMS_VERSION}" || echo "${red}Failed to clone, proceeding anyway${reset}"
        fi
    fi
    if git clone -b "${AWRAMS_VERSION}" "${AWRACMS_REPOSITORY}" "${REPO_PATH}/"; then
        echo "${green}[GIT] Cloned awracms${reset}"
    elif git clone -b "${AWRAMS_VERSION}" "${AWRACMS_REPOSITORY_SSH}" "${REPO_PATH}/"; then
        echo "${green}[GIT] Cloned awracms${reset}"
    elif git -C "${REPO_PATH}" pull origin "${AWRAMS_VERSION}"; then
        echo "${green}[GIT] Updated awracms repository${reset}"
    elif [ -d "${REPO_PATH}/awrams" ]; then
        echo "${green}[GIT] Repository found${reset}"
    else
        echo "${red}[GIT] Cannot clone awracms due to unexpected error${reset}"
        false
    fi
}

# Clone awracms data
dataClone() {
    # Checking if environment variable CLONE_DATA is set to true. Not case sensitive.
    if [[ "${CLONE_DATA}" == [Tt][Rr][Uu][Ee] ]]; then
        echo "${green}[GIT] Checking if GIT LFS is installed, if it isn't GIT LFS will be installed${reset}"
        git lfs --version >/dev/null 2>&1 || yum install git-lfs -y || {
            exit 1
            echo "${GIT_LFS_ERROR_MSG}"
        }
        git lfs install >/dev/null 2>&1
        echo "${green}[GIT] Cloning AWRACMS data to $DATA_PATH${reset}"
        if git lfs --version >/dev/null 2>&1; then
            echo "${green}[GIT] GIT LFS has been found${reset}"
        else
            echo "${red}${GIT_LFS_ERROR_MSG}${reset}"
            exit 1
        fi
        echo "${green}[AWRACMS] DATA is set to true. Data will be cloned${reset}"
        if git clone "$AWRACMS_DATA_REPOSITORY" "${DATA_PATH}/data"; then
            echo "${green}[GIT] Cloned awracms data${reset}"
        elif git clone -b "${AWRAMS_VERSION}" "${AWRACMS_DATA_REPOSITORY_SSH}" "${DATA_PATH}/data"; then
            echo "${green}[GIT] Cloned awracms${reset}"
        elif git -C "${DATA_PATH}/data" pull origin master; then
            echo "${green}[GIT] Updated awracms data repository${reset}"
        elif git -C "${DATA_PATH}/data" pull origin master |& grep "option: -C"; then
            echo "${red}[GIT] Cannot find -C option${reset}"
        else
            echo "${red}[GIT] Cannot clone awracms data due to unexpected error${reset}"
            false
        fi
    else
        echo "${green}[AWRACMS] DATA is not set or is not true. Data will not be cloned${reset}"
        sleep 5
    fi
}

# Create/Check symlink for awrams and altdatapath
createSymlink() {
    echo "${green}[AWRACMS] Creating symlink${reset}"

    if [[ -L "${HOME}/.awrams" ]]; then
        echo "${green}[AWRACMS] Symlink found${reset}"
        echo "${green}[AWRACMS] Checking if symlink leads to correct path${reset}"
        if ls -la "${HOME}" | grep "\-> ${REPO_PATH}/awrams" --quiet; then
            echo "${green}[AWRACMS] Correct symlink path found${reset}"
        else
            if [[ -h "${HOME}/.awrams" ]]; then
                mv "${HOME}/.awrams" "${HOME}/.awrams_old"
            fi
            echo "${green}[AWRACMS] Incorrect Symlink path found. Moving old symlink to ${HOME}/.awrams_old${reset}"
            ln -s "${REPO_PATH}/awrams" "${HOME}/.awrams" || { false && return; }
        fi
    else
        ln -s "${REPO_PATH}/awrams" "${HOME}/.awrams" || { false && return; }
        echo "${green}[AWRACMS] Symlink has been created${reset}"
    fi


    if [ "${altDataPath}" == 1 ]; then
        if [[ "${AWRAMS_DATA_PATH}" != "${REPO_PATH}/awrams/data" ]]; then
            echo "${green}[AWRACMS] Setting up alternate data path symlink${reset}"
            if [[ -d "${AWRAMS_DATA_PATH}" ]]; then
                if [[ -L "${REPO_PATH}/awrams/data" ]]; then
                    echo "${green}[AWRACMS] Symlink found${reset}"
                    echo "${green}[AWRACMS] Checking if symlink leads to correct path${reset}"
                    if ls -la "${REPO_PATH}/awrams" | grep "\-> ${AWRAMS_DATA_PATH}" --quiet; then
                        echo "${green}[AWRACMS] Correct symlink path found${reset}"
                    else
                        if [[ -h "${REPO_PATH}/awrams/data" ]]; then
                            mv "${REPO_PATH}/awrams/data" "${REPO_PATH}/awrams/data_old"
                        fi
                        echo "${green}[AWRACMS] Incorrect Symlink path found, deleting and recreating symlink${reset}"
                        rm -rf "${REPO_PATH}/awrams/data" || { false && return; }
                        ln -s "${AWRAMS_DATA_PATH}" "${REPO_PATH}/awrams/data" || { false && return; }
                    fi
                else
                    ln -s "${AWRAMS_DATA_PATH}" "${REPO_PATH}/awrams/data" || { false && return; }
                    echo "${green}[AWRACMS] Symlink has been created${reset}"
                fi
            else
                echo "${red}[AWRACMS] Alternate data path was not found${reset}"
            fi
        fi
    fi
}

# Conda create environment
condaCreateEnv() {
    echo "${green}[CONDA] Creating miniconda environment${reset}"
    if [[ -d "${MINICONDA_PATH}/envs/awra-cms" ]]; then
        conda env update -f "${REPO_PATH}/${CONDA_ENV}"
        echo "${green}[CONDA] Conda environment has been updated${reset}"
        # Activate environment
        . "${MINICONDA_PATH}/bin/activate" awra-cms || { false && return; }
        conda clean --all -y
    else
        conda env create -f "${REPO_PATH}/${CONDA_ENV}"
        echo "${green}[CONDA] Conda environment has been created${reset}"
        # Activate environment
        . "${MINICONDA_PATH}/bin/activate" awra-cms || { false && return; }
        conda clean --all -y
    fi
}

# Pip installing awracms and mpi4py
pipInstalls() {
    echo "${green}[PIP] Installing AWRACMS${reset}"
    pip uninstall -y awrams awrams.benchmarking awrams.utils awrams.cluster awrams.models awrams.simulation awrams.visualisation awrams.calibration awrams.cluster
    [ -d "${MINICONDA_PATH}/envs/awra-cms/lib/python3.6/site-packages" ] && rm -rf "${MINICONDA_PATH}/envs/awra-cms/lib/python3.6/site-packages/awrams*"
    if [[ -d "${REPO_PATH}/packages" ]]; then
        "${MINICONDA_PATH}/envs/awra-cms/bin/pip" install -I -e "${REPO_PATH}/packages" || { false && return; }
        echo "${green}[PIP] AWRACMS has been installed${reset}"
    else
        echo "${red}[AWRACMS] Could not find ${REPO_PATH}/packages${reset}"
        exit 1
    fi

    if [[ "${mpiInstalled}" == 1 ]]; then
        echo "${green}[PIP] Installing mpi4py${reset}"
        if "${MINICONDA_PATH}/envs/awra-cms/bin/pip" freeze | grep "mpi4py==${MPI4PY_VERSION}" --quiet; then
            echo "${green}[PIP] mpi4py has already been installed${reset}"
        else
            if [ -n "${MPI_PATH}" ]; then
                "${MINICONDA_PATH}/envs/awra-cms/bin/pip" download mpi4py==${MPI4PY_VERSION} || { false && return; }
                tar -vxf "${INSTALL_PATH}/mpi4py-${MPI4PY_VERSION}.tar.gz" || { false && return; }
                cd "${INSTALL_PATH}/mpi4py-${MPI4PY_VERSION}"
                "${MINICONDA_PATH}/envs/awra-cms/bin/python" setup.py build --mpicc="${MPI_PATH}/bin/mpicc" || { false && return; }
                "${MINICONDA_PATH}/envs/awra-cms/bin/python" setup.py install || mpiInstalled=0
                cd "${INSTALL_PATH}"
                [ -d "${INSTALL_PATH}/mpi4py-${MPI4PY_VERSION}" ] && rm -rf "${INSTALL_PATH}/mpi4py-${MPI4PY_VERSION}"
                [ -f "${INSTALL_PATH}/mpi4py-${MPI4PY_VERSION}.tar.gz" ] && rm "${INSTALL_PATH}/mpi4py-${MPI4PY_VERSION}.tar.gz"
            else
                "${MINICONDA_PATH}/envs/awra-cms/bin/pip" install mpi4py==${MPI4PY_VERSION} || { false && return; }
            fi
            echo "${green}[PIP] mpi4py has been installed${reset}"
        fi
    else
        echo "${green}[PIP] MPI has not been loaded or is not found. Install mpi4py manually${reset}"
        sleep 5
    fi
}

# Setting up environment variable: AWRAMS_BASE_PATH, AWRAMS_DATA_PATH and Module Loads
createActivationScript() {
    if [ -n ${MPI_PATH} ] && [[ "${mpiInstalled}" == 1 ]] && [ -d "${REPO_PATH}" ] && [ -d "${MINICONDA_PATH}" ] && [ -d "${REPO_PATH}/awrams" ]; then
        echo "${green}[AWRACMS] Creating activation script with installed MPI paths${reset}"

        [ -f "${REPO_PATH}/activation" ] && rm "${REPO_PATH}/activation" && echo "${green} [AWRACMS] Deleted old activation script${reset}"
        echo "green=\"\$(tput setaf 2)\"">> ${REPO_PATH}/activation
        echo "reset=\"\$(tput sgr0)\"">> ${REPO_PATH}/activation
        echo "WD=\"\${PWD}"\">> ${REPO_PATH}/activation
        echo "">> ${REPO_PATH}/activation
        echo "export LD_LIBRARY_PATH=\"${MPI_PATH}/lib/:\${LD_LIBRARY_PATH}\"">> ${REPO_PATH}/activation
        echo "export PATH=\"${MPI_PATH}/bin:\${PATH}\"">> ${REPO_PATH}/activation
        echo "cd "${MPI_PATH}"">> ${REPO_PATH}/activation
        echo "for i in *; do">> ${REPO_PATH}/activation
        echo "    alias \"\${i}\"=\"${MPI_PATH}/bin/\${i}\"">> ${REPO_PATH}/activation
        echo "done">> ${REPO_PATH}/activation
        echo "cd \"\${WD}"\">> ${REPO_PATH}/activation
        echo "export PYTHONPATH=\"\${PYTHONPATH}:${REPO_PATH}/awrams/code/user\"">> ${REPO_PATH}/activation
        echo "export AWRAMS_BASE_PATH=\"${REPO_PATH}/awrams"\">> ${REPO_PATH}/activation
        echo "[ -d \"${MINICONDA_PATH}\" ] && . \"${MINICONDA_PATH}/bin/activate\" awra-cms">> ${REPO_PATH}/activation
        echo "echo \"\${green}[AWRAMS] AWRACMS has been activated\${reset}\"">> ${REPO_PATH}/activation
    elif [[ "${moduleLoading}" == 1 ]] && [ -d "${REPO_PATH}" ] && [ -d "${MINICONDA_PATH}" ] && [ -d "${REPO_PATH}/awrams" ]; then
        echo "${green}[AWRACMS] Creating activation script with MPI module loading${reset}"

        [ -f "${REPO_PATH}/activation" ] && rm "${REPO_PATH}/activation" && echo "${green} [AWRACMS] Deleted old activation script${reset}"
        echo "green=\"\$(tput setaf 2)\"">> ${REPO_PATH}/activation
        echo "reset=\"\$(tput sgr0)\"">> ${REPO_PATH}/activation
        echo "">> ${REPO_PATH}/activation
        echo "module load ${mpiModule}">> ${REPO_PATH}/activation
        echo "export PYTHONPATH=\"\${PYTHONPATH}:${REPO_PATH}/awrams/code/user\"">> ${REPO_PATH}/activation
        echo "export AWRAMS_BASE_PATH=\"${REPO_PATH}/awrams"\">> ${REPO_PATH}/activation
        echo "[ -d \"${MINICONDA_PATH}\" ] && . \"${MINICONDA_PATH}/bin/activate\" awra-cms">> ${REPO_PATH}/activation
        echo "echo \"\${green}[AWRAMS] AWRACMS has been activated\${reset}\"">> ${REPO_PATH}/activation
    else
        echo "${green}[AWRACMS] Creating activation script with no MPI loading${reset}"

        [ -f "${REPO_PATH}/activation" ] && rm "${REPO_PATH}/activation" && echo "${green} [AWRACMS] Deleted old activation script${reset}"
        echo "green=\"\$(tput setaf 2)\"">> ${REPO_PATH}/activation
        echo "reset=\"\$(tput sgr0)\"">> ${REPO_PATH}/activation
        echo "">> ${REPO_PATH}/activation
        echo "export PYTHONPATH=\"\${PYTHONPATH}:${REPO_PATH}/awrams/code/user\"">> ${REPO_PATH}/activation
        echo "export AWRAMS_BASE_PATH=\"${REPO_PATH}/awrams"\">> ${REPO_PATH}/activation
        echo "[ -d \"${MINICONDA_PATH}\" ] && . \"${MINICONDA_PATH}/bin/activate\" awra-cms">> ${REPO_PATH}/activation
        echo "echo \"\${green}[AWRAMS] AWRACMS has been activated\${reset}\"">> ${REPO_PATH}/activation
    fi
}

# Run python nosetests to ensure installation is successful
runTests() {
    if [[ "${CLONE_DATA}" == [Tt][Rr][Uu][Ee] ]]; then
        . "${REPO_PATH}/activation"
        echo "${green}[AWRACMS] Running tests${reset}"
        cd "${REPO_PATH}/packages"
        python setup.py nosetests || { echo "${red}[AWRACMS] Tests failed.${reset}" && return; }
        echo "${green}[AWRACMS] Tests have passed${reset}"
    else
        echo "${green}[AWRACMS] Not running tests. Data hasn't been cloned${reset}"
    fi
}

# Fail script
fail() {
    echo "$1" >&2
    exit 1
}

# Retry loop (5x)
retry() {
    local n=1
    local max=5
    local delay=5
    while true; do
        "$@" && break || {
            if [[ "$n" -lt "$max" ]]; then
                ((n++))
                echo "${red}$@ failed. Attempt $n/$max:${reset}"
                sleep $delay
            else
                fail "${red}The $@ has failed after $n attempts.${reset}"
            fi
        }
    done
}

# Installation job
install() {
    retry installDependencies
    retry condaInstall
    retry installMPI
    retry moduleLoad
    retry awraClone
    retry dataClone
    retry createSymlink
    retry condaCreateEnv
    retry pipInstalls
    retry createActivationScript
    retry runTests
    # delete cache of pip
    [ -d "${HOME}/.cache/pip" ] && rm -rf "${HOME}/.cache/pip"
}

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

# Exit cleanup
trap 'fail "${red}The execution was aborted because a command exited with an error status code.${reset}"' ERR

if sudoCheck; then
    if install; then
        if [[ "${isSudo}" == 1 ]]; then
            echo "${green}[SUDO CHECK] Changing directory ownership to ${SUDO_USER}${reset}"
            chown -R "${SUDO_USER}:${SUDO_USER}" "${REPO_PATH}"
            if [[ "${MINICONDA_INSTALLED}" == 1 ]]; then
                chown -R "${SUDO_USER}:${SUDO_USER}" "${MINICONDA_PATH}"
            fi
        fi
        echo "${green}[AWRACMS] Successfully Installed AWRACMS${reset}"
        echo "${green}Installation Path:${reset} ${REPO_PATH}"
        echo "${green}Miniconda Path:${reset} ${INSTALL_PATH}/miniconda3"
        echo "${green}Data Path:${reset} ${DATA_PATH}"
        echo "${green}Activate awra-cms:${reset} In the installation path: ${REPO_PATH} use the command 'source activation'${reset}"
        if [[ "${mpiInstalled}" == 0 ]]; then
            echo "${red}[AWRACMS] MPI was not found or the incorrect module has been loaded. Install MPI and mpi4py manually
            Use 'module avail' and 'module load' to load the OpenMPI module
            Use 'mpicc --version' to verify OpenMPI has been loaded
            Run 'pip install mpi4py' to install mpi4py. ${reset}"
        fi
        echo "${green}Launch jupyter notebook with the command 'jupyter notebook' once the awra-cms environment has been activated${reset}"
    else
        echo "${red}Failed to install AWRACMS${reset}"
    fi
else
    echo "${red}[SUDO CHECK] Failed to check if user is running with sudo${reset}"
fi