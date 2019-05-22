# AWRA Community Modelling System

# Installation Guide
[Linux](#Linux)

[Windows](#Windows)

# Linux

## Automated Installation

### With Sudo
Run the following command in terminal:
```
curl -s https://raw.githubusercontent.com/awracms/awra_cms/AWRA_CMS_v1.2/scripts/install.sh | sudo -E bash
```

Alternative Method:

```
wget -O - https://raw.githubusercontent.com/awracms/awra_cms/AWRA_CMS_v1.2/scripts/install.sh | sudo -E bash
```

### Without Sudo
Run the following command in terminal:

```
curl -s https://raw.githubusercontent.com/awracms/awra_cms/AWRA_CMS_v1.2/scripts/install.sh | bash
```

Alternative Method:

```
wget -O - https://raw.githubusercontent.com/awracms/awra_cms/AWRA_CMS_v1.2/scripts/install.sh | bash
```
### Optional Environment Variables:

```
# By default the dependencies are installed. If you do not choose to install the dependencies set INSTALL_DEPENDENCIES to FALSE
export INSTALL_DEPENDENCIES=FALSE                    # (default: True)

# By default the version downloaded is master branch.
export AWRAMS_VERSION=dev                            # (default: master)

# By default data is not cloned. If you choose to clone data set CLONE_DATA to TRUE
export CLONE_DATA=TRUE                               # (default: False)

# By default AWRAMS_INSTALL_PATH is your current directory.
export AWRAMS_INSTALL_PATH="/installation/path/"     # (default: Current directory)

# By default CLONE_DATA_PATH is set to "/install/path/awrams/data". If this variable is set an environment variable will be created.
export CLONE_DATA_PATH="/clone/data/to"              # (default: AWRAMS_INSTALL_PATH/awrams/data)

# By default CONDA_PATH is set to none if you have a conda environment already installed point CONDA_PATH to it.
export CONDA_PATH="/home/miniconda3/"                # (default: none)

# By default OPENMPI_VERSION is set to 3.1.3.
export OPENMPI_VERSION="1.8.0"                       # (default: 3.1.3)

# By default OPENMPI_PATH is /home/user/. If you choose to install it elsewhere specify the path or if you already have openmpi installed then point the variable to where it has been installed.
export OPENMPI_PATH="/usr/bin/"              # (default: /home/user)

# By default OPENMPI_INSTALL is set to install by default. If you do not choose to install OPENMPI and wish to module load MPI set this variable to true.
export OPENMPI_INSTALL="false"                        # (default: true)

# By default MPI4PY_VERSION is set to 3.0.0. This variable changes the python package version.
export MPI4PY_VERSION=3.0.0                           # (default: 3.0.0)

# By default this is set to false. This variable deletes the folders created from a previous install and re-installs everything.
export FROM_SCRATCH=true                              # (default: false)
```
### Notes:
Git and Git-LFS are installed in the base environment of conda.

#

## Manual Installation

The *AWRAMS_INSTALL_PATH* environment variable is required to make the installation easier to follow. Run the following command in the terminal with the correct path specified:

```
export AWRAMS_INSTALL_PATH="/path/to/install/awra/"
```

*AWRAMS_DATA_PATH* is an optional environment that could be set if you choose to clone the data as well. Run the following command in the terminal with the correct path specified:

```
export AWRAMS_DATA_PATH="/path/to/store/data"
```


### Table of Contents

1. [Install Dependencies](###-1.-Install-dependencies)
>1.1 [Install/Load OpenMPI](###-1.1-Install\Load-OpenMPI)
2. [Clone AWRACMS](###-2.-Clone-AWRACMS)
>2.1 [[OPTIONAL] Clone AWRACMS Data](###-2.1-[OPTIONAL]-Clone-AWRACMS-Data)
3. [Create Symbolic Link](###-3.-Create-Symbolic-Link)
4. [Install Miniconda3](###-4.-Install-Miniconda3)
5. [Create Miniconda3 Environment](###-5.-Create-Miniconda3-Environment)
6. [Install AWRACMS Python Packages](###-6.-Install-AWRACMS-Python-Packages)
7. [Install mpi4py](###-7.-Install-mpi4py)
8. [Activate Environment](###-8.-Activate-Environment)
9. [Launch Notebook](###-9.-Launch-Notebook)

### 1. Install dependencies

#### Fedora/RHEL/CentOS

```
yum install bzip2 make automake gcc gcc-c++ kernel-devel 
```
#### Ubuntu/Debian

```
apt-get install bzip2 gcc build-essential
```

### 1.1 Install OpenMPI

```
curl https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.3.tar.bz2 -o openmpi-3.1.3.tar.bz2
tar -xvf openmpi-3.1.3.tar.bz2
cd openmpi-3.1.3
./configure --prefix="${HOME}/.openmpi"
make install
export PATH="${HOME}/.openmpi/bin:$PATH"
```

#### Verification

`mpirun --version` If the following command fails it's because the module hasn't been loaded.

`mpicc --version` If the following command fails it's because GCC/G++ is not installed, GCC/G++ is lower than version 4.4 or the incorrect OpenMPI module has been loaded.


### 2. Install Miniconda3

Miniconda3 is required to create the required python environment. Run the following commands in terminal to install miniconda3:

```
cd ${AWRAMS_INSTALL_PATH}
curl "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p "${AWRAMS_INSTALL_PATH}/miniconda3"
source "${AWRAMS_INSTALL_PATH}/miniconda3/bin/activate" base
conda install git git-lfs -y
```

Alternative method: https://www.anaconda.com/rpm-and-debian-repositories-for-miniconda/

### 3. Clone AWRACMS

Run the following command in terminal:

```
# Master can be changed to specify a particular version such as AWRA_CMS_v1.2
git clone -b master https://github.com/awracms/awra_cms.git "${AWRAMS_INSTALL_PATH}"
```

Alternative Method #1:

```
# Master can be changed to specify a particular version such as AWRA_CMS_v1.2
git clone -b master git@github.com:awracms/awra_cms.git "${AWRAMS_INSTALL_PATH}"
```

Alternative Method #2:

```
# Master can be changed to specify a particular version such as AWRA_CMS_v1.2
wget "https://github.com/awracms/awra_cms/archive/master.zip" --directory-prefix="${AWRAMS_INSTALL_PATH}"
unzip "${AWRAMS_INSTALL_PATH}/master.zip"
mv "${AWRAMS_INSTALL_PATH}/master" "${AWRAMS_INSTALL_PATH}/awrams_cm"
export AWRAMS_INSTALL_PATH=${AWRAMS_INSTALL_PATH}/awrams_cm
```


### 3.1 [OPTIONAL] Clone AWRACMS Data

Run the following command in terminal:
```
git clone https://github.com/awracms/awracms_data.git "${AWRAMS_DATA_PATH}"
```

Alternative Method #1:

```
git clone git@github.com:awracms/awracms_data.git "${AWRAMS_DATA_PATH}"
```

Alternative Method #2:

```
wget "https://github.com/awracms/awracms_data/archive/master.zip" --directory-prefix="${AWRAMS_DATA_PATH}"
unzip "${AWRAMS_DATA_PATH}/master.zip"
mv "${DATA_PATH}/master" "${AWRAMS_DATA_PATH}/data"
export AWRAMS_DATA_PATH="${AWRAMS_DATA_PATH}/data"
```

### 4. Create Symbolic Link

Symbolic link are required to use the default path in the configuration files. Run the following commands in terminal to create the symbolic link required:

```
ln -s "${AWRAMS_INSTALL_PATH}/awrams" "~/.awrams"
```


### 5. Create Miniconda3 Environment

Run the following command to create the *awra-cms* miniconda environment:

```
conda env create -f "${AWRAMS_INSTALL_PATH}/linux_conda_install_env.yml"
```

Miniconda has to be activated whenever a new terminal is launched with the following command where *${AWRAMS_INSTALL_PATH}* is the installation path:

```
source "${AWRAMS_INSTALL_PATH}/miniconda3/bin/activate awra-cms"
```

### 6. Install AWRACMS Python Packages

Run the following command to install AWRACMS python packages.
```
cd ${AWRAMS_INSTALL_PATH}/packages
pip install -e .
```

### 7. Install mpi4py
Installing mpi4py requires MPI compiler. MPI compiler should be installed by default if *gcc, openmpi and openmpi-devel* have been installed. If you're using Intel MPI follow the following guide: https://software.intel.com/en-us/distribution-for-python

To install mpi4py run the following command (where mpicc="PATH" can be changed to where mpicc is found):

```
source "${AWRAMS_INSTALL_PATH}/miniconda3/bin/activate" awra-cms
pip download mpi4py==3.0.0
tar -vxf mpi4py-3.0.0.tar.gz
cd mpi4py-3.0.0
python setup.py build --mpicc="${HOME}/.openmpi/bin/mpicc"
python setup.py install
```

### 8. Activate Environment

Activate the environment by running the command in the project/repository path:

```
./activate
```

### 9. Launch Notebook

To launch jupyter notebook run the following command in terminal:

```
jupyter notebook
```

# Windows

## Automated Installation

### Download Batch Script

[Click here](https://raw.githubusercontent.com/awracms/awra_cms/AWRA_CMS_v1.2/scripts/install.bat) to download the install.bat script. Double click to run the script and follow the prompts.

## Manual Installation

The *AWRAMS_INSTALL_PATH* environment variable is required to make the installation easier to follow. Run the following command in the terminal with the correct path specified:

```
set AWRAMS_INSTALL_PATH="C:\path\to\install\awra\"
```

*AWRAMS_DATA_PATH* is an optional environment that could be set if you choose to clone the data as well. Run the following command in the terminal with the correct path specified:

```
set AWRAMS_DATA_PATH="C:\path\to\store\data"
```


### Table of Contents

1. [Dependencies](###-1.-Dependencies)
2. [Clone AWRACMS](###-2.-Clone-AWRACMS-for-Windows)
>2.1 [[OPTIONAL] Clone AWRACMS Data](###-2.1-[OPTIONAL]-Clone-AWRACMS-Data-for-Windows)
3. [Create Symbolic Links](###-3.-Create-Symbolic-Links-for-Windows)
4. [Create Miniconda3 Environment](###-4.-Create-Miniconda3-Environment-for-Windows)
5. [Install AWRACMS Python Packages](###-5.-Install-AWRACMS-Python-Packages-for-Windows)
6. [Install mpi4py](###-6.-Install-mpi4py-for-Windows)
7. [Set Environment Variables](###-7.-Set-Environment-Variables-for-Windows)
8. [Launch Notebook](###-8.-Launch-Notebook-for-Windows)


### 1. Dependencies

Required dependencies:

```
Microsoft MPI
Anaconda/Miniconda3
GIT
```

Optional dependencies:

```
GIT LFS
```

### 2. Clone AWRACMS for Windows

Run the following command in cmd:

```
# Master can be changed to specify a particular version
git clone -b master https://github.com/awracms/AWRACMS %AWRAMS_INSTALL_PATH%
```

### 2.1 [OPTIONAL] Clone AWRACMS Data for Windows

Follow the following guide to install GIT LFS: https://github.com/git-lfs/git-lfs/wiki/Installation

Once GIT LFS has been installed run the following command in command prompt:

```
git clone https://github.com/awracms/AWRACMS %AWRAMS_DATA_PATH%
```

### 3. Create Symbolic Link for Windows

Symbolic link are required to use the default path in the configuration files. Run the following commands in command prompt to create the symbolic link required:

```
mklink /J %USERPROFILE%\.awrams %AWRA_INSTALL_PATH%\awrams_cm\awrams
```

### 4. Create Miniconda3 Environment for Windows

Run the following command to create the *awra-cms* miniconda environment:

```
conda env create -f "%AWRAMS_INSTALL_PATH%/conda_install_env.yml"
```

Activate the AWRACMS conda environment with the following command ran in command prompt:

```
conda activate awra-cms
```


### 5. Install AWRACMS Python Packages for Windows

Run the following command to install AWRACMS python packages.
```
cd %AWRAMS_INSTALL_PATH%\packages
pip install -e .
```

### 6. Install mpi4py for Windows

To install mpi4py run the following command:

```
source "%AWRAMS_INSTALL_PATH%/miniconda3/bin/activate" awra-cms
pip install mpi4py
```

### 7. Set Environment Variables for Windows


**REQUIRED**

*AWRAMS_BASE_PATH* is a required environment variable that should point to the *awrams* folder within the *config* folder of the cloned repository.

Run the command below in PowerShell with the correct path set to load the environment variable as default when a new command prompt instance is started:

```
[Environment]::SetEnvironmentVariable("AWRAMS_BASE_PATH", "%AWRAMS_INSTALL_PATH%/awrams", "User")
```

**OPTIONAL**

*AWRAMS_DATA_PATH* is an optional environment variable that can be pointed to the data path specified.

Run the command below in PowerShell with the correct path set to load the environment variable as default when a new command prompt instance is started:

```
[Environment]::SetEnvironmentVariable("AWRAMS_DATA_PATH", "%AWRAMS_DATA_PATH%", "User")
```

### 8. Launch Notebook for Windows

To launch jupyter notebook run the following command in command prompt once the environment has been activated:

```
jupyter notebook
```



