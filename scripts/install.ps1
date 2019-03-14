# Only run if running as administrator
#Requires -RunAsAdministrator

# On error stop
$ErrorActionPreference = "Stop"

# Command to get exception type name
## $error[0].exception.gettype().fullname ##

#############################################
# Functions
#############################################

# Create Directory
function createDirectory {
    If (!$args[0]) {
        $args[1]=$args[2]
        Write-Host "[AWRACMS] Installing to $args[1]"
    } else {
        $args[1]=$args[0]
        try {
            cd "$args[1]"
            Write-Host "[AWRACMS] Directory found"
        } catch [System.Management.Automation.ItemNotFoundException] {
            Write-Host "[AWRACMS] Directory not found"
            Write-Host "[AWRACMS] Creating directory $args[1]"
            try {
                mkdir -p "$args[1]"
                Write-Host "[AWRACMS] Successfully created directory $args[1]"
            } catch [System.UnauthorizedAccessException] {
                Write-Host "[AWRACMS] Creating directory failed. Change folder permissions or install to a different directory"
                exit 1
            } catch {
                Write-Host "$catchException" -fore white -back red
                $error[0]
                exit 1
            }
        } catch {
            Write-Host "$catchException" -fore white -back red
            $error[0]
            exit 1
        }
    }
}

# Setting parameters INSTALL_PATH, VERSION and DATA_PATH
function setParams* {
    createDirectory $AWRA_INSTALL_PATH $INSTALL_PATH $PWD.PATH
    createDirectory $CLONE_DATA_PATH $DATA_PATH $INSTALL_PATH\awrams_cm\config\.awrams\data

    If (!$AWRA_VERSION) {
        $AWRA_VERSION="master"
        Write-Host "[AWRACMS] Version: $AWRA_VERSION"
    } else {
        Write-Host "[AWRACMS] Version: $AWRA_VERSION"
    }


    $REPO_PATH="$INSTALL_PATH\awrams_cm"
    $CONDA_ENV="win_conda_install_env.yml"
    $AWRACMS_REPOSITORY="https://github.com/awracms/awra_cms.git"
    $AWRACMS_TEST_DATA_REPOSITORY="https://github.com/awracms/awracms_data.git"
    $successfulInstall="Successfully Installed"
    $checkInstall="Checking if the following depedency is installed:"
    $installing="Installing"
    $alreadyInstall="The following dependency has already been installed:"
    $installed="The following dependency has been installed:"
    $catchException="Other Exception, exiting installation script"
    $chocoInstalls = @("miniconda3","git")
    $installCommands = @("conda","git")
    $MPIDOWNLOAD = "https://download.microsoft.com/download/A/E/0/AE002626-9D9D-448D-8197-1EA510E297CE/msmpisetup.exe"
}


# MPI installation
function setupMPI {
    try {
        Write-Host "$checkInstall MPI" -fore black -back white
        mpiexec --version | Out-Null
        Write-Host "$alreadyInstall MPI" -fore black -back white
    } catch [System.Management.Automation.CommandNotFoundException] { 
        Write-Host "Installing MPI" -fore black -back white
        try {
            wget "$MPIDOWNLOAD" -o install.exe
            .\install.exe -unattend|more
            mpiexec --version | Out-Null
            Write-Host "Installation complete" -fore black -back white
        } catch [System.Net.WebException] {
            Write-Host "Unable to download MPISETUP: $MPIDOWNLOAD" -fore black -back white
        } catch {
            Write-Host "$catchException" -fore white -back red
            $error[0]
        }
    } catch {
        Write-Host "$catchException" -fore white -back red
        $error[0]
    }
}


# Chocolatey installation
function setupChocolatey {
    try {
        Write-Host "$checkInstall Chocolatey" -fore black -back white
        choco --version | Out-Null
        Write-Host "Chocolatey Found" -fore black -back white
    } catch [System.Management.Automation.CommandNotFoundException] { 
        Write-Host "Chocolatey Not found" -fore black -back white
        try {
            Write-Host "Installing Chocolatey" -fore black -back white
            Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
        } catch [System.Net.WebException],[System.IO.IOException] {
            Write-Host "Unable to download and install Chocolatey" -fore white -back red
            exit 1
        } catch {
            Write-Host "$catchException" -fore white -back red
            $error[0]
            exit 1
        }
    } catch {
        Write-Host "$catchException" -fore white -back red
        $error[0]
        exit 1
    }
}

# Miniconda and GIT installation
function installChocoDeps {
    try {
        If ($chocoInstalls.Length -match $installCommands.Length) {
            For($i=0;$i -lt $chocoInstalls.Length; $i++) {
                try {
                    Write-Host "$checkInstall $($chocoInstalls[$i])" -fore black -back white
                    & "$($installCommands[$i])" --version | Out-Null
                    Write-Host "$alreadyInstall $($chocoInstalls[$i])" -fore black -back white
                } catch [System.Management.Automation.CommandNotFoundException] { 
                    Write-Host "$installing $($chocoInstalls[$i])" -fore black -back white
                    choco install "$($chocoInstalls[$i])"
                    Write-Host "$installed $($chocoInstalls[$i])" -fore black -back white
                    refreshenv
                } catch {
                    Write-Host "$catchException" -fore white -back red
                    $error[0]
                    exit 1
                }
            }
        }
    } catch {
        Write-Host "$catchException" -fore white -back red
        $error[0]
        exit 1
    }
}


# Clone AWRACMS
function cloneAWRA {
    try {
        Write-Host "[GIT] Cloning AWRACMS to $REPO_PATH" -fore black -back white
        git lfs install
        git clone -b "$AWRA_VERSION" "$AWRACMS_REPOSITORY" "$REPO_PATH\"
        Write-Host "[GIT] Cloned awracms" -fore black -back white
    } catch  {
        try {
            git -C "$REPO_PATH" pull origin "$AWRA_VERSION"
            Write-Host "[GIT] Updated awracms repository" -fore black -back white
        } catch {
            Write-Host "[GIT] Cannot clone awracms due to unexpected error" -fore white -back red
            exit 1
        }
    }
}

# Clone Data
function cloneData {
    if ($CLONE_DATA.ToLower() -eq "true") {
        try {
            Write-Host "[GIT] Cloning AWRACMS to $DATA_PATH" -fore black -back white
            git lfs install
            git clone "$AWRACMS_DATA_REPOSITORY" "$DATA_PATH\"
            Write-Host "[GIT] Cloned awracms" -fore black -back white
        } catch {
            try {
                git -C "$DATA_PATH" pull origin master
                Write-Host "[GIT] Updated awracms repository" -fore black -back white
            } catch {
                Write-Host "[GIT] Cannot clone awracms due to unexpected error" -fore white -back red
                exit 1
            }
        }
    } else {
        Write-Host "[GIT] Not cloning data" -fore black -back white
    }
}

function createSymLink {
    try {
        dir "$HOME" | findstr "\.awrams"
        Write-Host "[AWRACMS] Symbolic link found"
    } catch {
        mklink /J %USERPROFILE%\.awrams config\.awrams
        Write-Host "[AWRACMS] Symbolic Link has been created"
    }
}


# Setup Miniconda Environment
function setupCondaEnv {
    try {
        Write-Host "[CONDA] Creating miniconda environment" -fore black -back white
        conda env list | findstr "awra" --quiet
        conda env update -f "$REPO_PATH\$CONDA_ENV"
        Write-Host "[CONDA] Conda environment has been updated" -fore black -back white
        # Activate environment
        conda activate awra-cms
    } catch {
        try {
            conda env create -f "$REPO_PATH\$CONDA_ENV"
            Write-Host "[CONDA] Conda environment has been created" -fore black -back white
            # Activate environment
            conda activate awra-cms
        } catch {
            Write-Host "[CONDA] Cannot create conda environment due to unexpected error" -fore white -back red
            exit 1
        } cd 
    }
}

# Installs AWRA and mpi4py
function pipInstalls {
    try {
        Write-Host "[PIP] Installing AWRACMS" -fore black -back white
        pip freeze | findstr "awra" --quiet
        Write-Host "[PIP] AWRACMS has already been installed" -fore black -back white
    } catch [System.Management.Automation.CommandNotFoundException] {
        Write-Host "[PIP] PIP command not found"  -fore white -back red
    } catch {

        try {
            pip install -e "$REPO_PATH\utils" "$REPO_PATH\benchmarking" "$REPO_PATH\models" "$REPO_PATH\simulation" "$REPO_PATH\visualisation" "$REPO_PATH\calibration"
            Write-Host "[PIP] AWRACMS has been installed" -fore black -back white
        } catch {
            Write-Host "[PIP] Failed to install AWRACMS" -fore white -back red
            exit 1
        }
    }
    }

    try {
        Write-Host "[PIP] Installing mpi4py" -fore black -back white
        pip freeze | findstr "mpi4py" --quiet
        Write-Host "[PIP] mpi4py has already been installed" -fore black -back white
    } catch [System.Management.Automation.CommandNotFoundException] {
        Write-Host "[PIP] PIP command not found" -fore white -back red
    } catch {
        try {
            pip install mpi4py
            Write-Host "[PIP] mpi4py has been installed" -fore black -back white
        } catch {
            Write-Host "[PIP] Failed to install mpi4py" -fore white -back red
            exit 1
        }
    }

}

# Setup environment variables
function setupEnvVars {
    try {
        Write-Host "[AWRACMS] Setting environment variable: AWRA_BASE_PATH" -fore black -back white
        if ($AWRA_BASE_PATH) {
            Write-Host "[AWRACMS] Environment variable has already been set" -fore black -back white
        } else {
            Write-Host "[AWRACMS] Adding environment variable to user profile" -fore black -back white
            [Environment]::SetEnvironmentVariable("AWRA_BASE_PATH", "$REPO_PATH\config\.awrams", "User")
        }
    } catch {
        Write-Host "[AWRACMS] Cannot set environment variable due to unexpected error" -fore white -back red
        exit 1
    }

    try {
        Write-Host "[AWRACMS] Setting environment variable: AWRA_DATA_PATH" -fore black -back white
        if ($AWRA_DATA_PATH) {
            Write-Host "[AWRACMS] Environment variable has already been set" -fore black -back white
        } else {
            Write-Host "[AWRACMS] Adding environment variable to user profile" -fore black -back white
            [Environment]::SetEnvironmentVariable("AWRA_DATA_PATH", "$DATA_PATH", "User")
        }
    } catch {
        Write-Host "[AWRACMS] Cannot set environment variable due to unexpected error" -fore white -back red
        exit 1
    }

}


setParams
setupChocolatey
installChocoDeps
cloneAWRA
cloneData
createSymLink
setupCondaEnv
pipInstalls
setupEnvVars
