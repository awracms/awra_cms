#!/bin/bash

# ------------------------------------------------------------
# Setup Environment
# ------------------------------------------------------------
red="$(tput setaf 1)"
green="$(tput setaf 2)"
reset="$(tput sgr0)"

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

# Pip installing awracms and mpi4py
pipInstalls() {
    echo "${green}[PIP] Installing ${reset}"
    sudo pip uninstall -y awrams awrams.benchmarking awrams.utils awrams.cluster awrams.models awrams.simulation awrams.visualisation awrams.calibration awrams.cluster
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

# Run python nosetests to ensure installation is successful
runTests() {
    if [[ "${CLONE_DATA}" == [Tt][Rr][Uu][Ee] ]]; then
        # . "${REPO_PATH}/activation"
        echo "${green}[AWRACMS] Running tests${reset}"
        cd "${REPO_PATH}/packages"
        python setup.py nosetests || { echo "${red}[AWRACMS] Tests failed.${reset}" && return; }
        echo "${green}[AWRACMS] Tests have passed${reset}"
    else
        echo "${green}[AWRACMS] Not running tests. Data hasn't been cloned${reset}"
    fi
}

# Installation job
install() {
    mpiInstalled=1
    createSymlink
    pipInstalls
    runTests
    [ -d "${HOME}/.cache/pip" ] && rm -rf "${HOME}/.cache/pip"
}

if install; then
    echo "${green}[AWRACMS] Successfully Installed AWRACMS${reset}"
    echo "${green}Installation Path:${reset} ${REPO_PATH}"
    echo "${green}Data Path:${reset} ${DATA_PATH}"
else
    echo "${red}Failed to install AWRACMS${reset}"
fi
