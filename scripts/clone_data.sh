#!/bin/bash

###########################################
# Setup Environment
###########################################
red=$(tput setaf 1)
green=$(tput setaf 2)
reset=$(tput sgr0)
AWRACMS_DATA_REPOSITORY="https://github.com/awracms/awracms_data.git"
AWRACMS_DATA_WGET_REPO="https://github.com/awracms/awracms_data/archive/master.zip"
AWRACMS_DATA_REPOSITORY_SSH="git@github.com:awracms/awracms_data.git"


# Clone Data Path
if [[ -z "${CLONE_DATA_PATH}" ]]; then
    if [ -z "${AWRA_BASE_PATH}" ]; then
        echo "${red}[AWRA] AWRA_BASE_PATH or CLONE_DATA_PATH is not found, please set the variable${reset}"
        exit 1
    else
        DATA_PATH="${AWRA_BASE_PATH}/data"
    fi
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


##########################################
# Functions
##########################################
# Clone awra_cms data
dataClone()
{
    echo "${green}[GIT] Checking if GIT LFS is installed${reset}"
    if git lfs --version >/dev/null 2>&1; then
        echo "${green}[GIT] GIT LFS has been found${reset}"
    elif yum install git-lfs -y; then
        echo "$green}[GIT] GIT LFS has been installed${reset}"
    elif apt-get install git-lfs -y; then
        echo "${green}[GIT] Git LFS has been installed${reset}"
    elif [[ "${WGET_CLONE}" = [Tt][Rr][Uu][Ee] ]]; then
        echo "${green}[GIT] WGET_CLONE has been set to true${reset}"
    else
        echo "${red}[AWRA] Please install git-lfs or set the environment variable WGET_CLONE to true. Instructions found here: https://github.com/git-lfs/git-lfs/wiki/Installation${reset}"
         exit 1
    fi

    if [[ "${WGET_CLONE}" = [Tt][Rr][Uu][Ee] ]]; then
        echo "${green}[GIT] Downloading awracms data....${reset}"
        wget "${AWRACMS_DATA_WGET_REPO}" --directory-prefix="${DATA_PATH}" || exit 1
        echo "${green}[GIT] Unzipping AWRACMS${reset}"
        unzip "${DATA_PATH}/master.zip" || exit 1
        echo "${green}[GIT] Deleting zip file${reset}"
        rm "${DATA_PATH}/master.zip" || exit 1
        echo "${green}[GIT] Renaming folder name${reset}"
        mv "${DATA_PATH}/awracms_data-master" "${DATA_PATH}/data" || exit 1
        DATA_PATH="${DATA_PATH}/data"
    else
        echo "${green}[GIT] Cloning AWRACMS data to ${DATA_PATH}${reset}"
        if git clone "$AWRACMS_DATA_REPOSITORY" "${DATA_PATH}/"; then
            echo "${green}[GIT] Cloned awracms data${reset}"
        elif git clone -b "${AWRA_VERSION}" "${AWRACMS_DATA_REPOSITORY_SSH}" "${DATA_PATH}/"; then
            echo "${green}[GIT] Cloned awracms${reset}"
        elif git -C "${DATA_PATH}" pull origin master; then
            echo "${green}[GIT] Updated awracms data repository${reset}"
        
        else
            echo "${red}[GIT] Cannot clone awracms data due to unexpected error${reset}"
            exit 1
        fi
        
    fi
    
    echo "${green}[AWRACMS] Setting environment variable: AWRA_DATA_PATH${reset}"
    if grep "AWRA_DATA_PATH$" < "${HOME}/.bashrc" >/dev/null 2>&1; then
        echo "${green}[AWRACMS] Environment variable has already been set${reset}"
    elif [[ "${altDataPath}" = 1 ]]; then
        echo "${green}[AWRACMS] Adding environment varaible to .bashrc file${reset}"
        echo "export AWRA_DATA_PATH$=${DATA_PATH}" >> "${HOME}/.bashrc" || exit 1
        . "${HOME}/.bashrc" || exit 1
    else
        echo "${green}[AWRACMS] Data path has not been set. Variable has been skipped{$reset}"
    fi

}

####################################
# Main
####################################
dataClone
