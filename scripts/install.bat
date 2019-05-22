@echo off
:: BatchGotAdmin
::-------------------------------------
REM  --> Check for permissions
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"

REM --> If error flag set, we do not have admin.
if '%errorlevel%' NEQ '0' (
    echo Requesting administrative privileges...
    goto UACPrompt
) else ( goto gotAdmin )

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    set params = %*:"="
    echo UAC.ShellExecute "cmd.exe", "/c %~s0 %params%", "", "runas", 1 >> "%temp%\getadmin.vbs"

    "%temp%\getadmin.vbs"
    del "%temp%\getadmin.vbs"
    exit /B

:gotAdmin
    pushd "%CD%"
    CD /D "%~dp0"
::--------------------------------------

REM Environment Variables
set AWRACMS_URL="https://github.com/awracms/awra_cms.git"
set DATA_URL="https://github.com/awracms/awracms_data.git"
set MSMPI_DL="https://download.microsoft.com/download/A/E/0/AE002626-9D9D-448D-8197-1EA510E297CE/msmpisetup.exe"
set MINICONDA_DL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
set CONDA_ENV=win_conda_install_env.yml
set MPI4PY_VERSION=3.0.0
set AWRA_VERSION=master

:AWRAversion
    if '%AWRA_VERSION%' == 'master' (
        set /p VERSION_A="What version of AWRACMS would you like to download? Press enter for default [master]: "
    )
    if '%VERSION_A%' == '' (
        echo [AWRACMS] Using default version: master
        goto InstallPath
    ) else (
        set AWRA_VERSION=%VERSION_A%
    )

:InstallPath
    if '%INSTALL_PATH%' == '' (
        goto InstallPathPrompt
    )

    if exist %INSTALL_PATH%\NUL (
        echo [AWRACMS] Install directory found
    ) else (
        mkdir %INSTALL_PATH%
        if exist %INSTALL_PATH%\NUL (
            echo [AWRACMS] Installation directory has been created
        ) else (
            echo [AWRACMS] You have entered an invalid path or you do not have permission to create a directory in the path given. [Example: C:\Users\Guest\Desktop\AWRA]
            goto InstallPathPrompt
        )
    )
    goto DataPath

:InstallPathPrompt
    @echo off
    set /p INSTALL_PATH="Where would you like to install AWRACMS? [C:\Users\Guest\Desktop]: "
    goto InstallPath

:DataPath
    if '%DATA_A%' == '' (
        goto DataPathPrompt
    )
    if /i '%DATA_A%' == 'n' (
        IF NOT "%DATA_PATH%"=="" (
            if exist %DATA_PATH%\NUL (
                echo [AWRACMS] Data directory found
            ) else (
                echo [AWRACMS] You have entered an invalid path or you do not have permission to create a directory in the path given. Example: C:\Users\Guest\Data
                goto DataPathPrompt
            )
        )
        goto GetPaths
    )
    if /i '%DATA_A%' == 'y' (
        if exist %DATA_PATH%\NUL (
            echo [AWRACMS] Data directory found
        ) else (
            mkdir %DATA_PATH%
            if exist %DATA_PATH%\NUL (
                echo [AWRACMS] Data directory has been created
            ) else (
                echo [AWRACMS] You have entered an invalid path or you do not have permission to create a directory in the path given. Example: C:\Users\Guest\Data
                goto DataPathPrompt
            )
        )
        goto GetPaths
    ) else (
        echo [AWRACMS] You did not enter Y or N as a value. Try again.
        goto DataPathPrompt
    )

:DataPathPrompt
    @echo off

    set /p DATA_A="Would you like to download AWRACMS data? [Y/N]: "
    if /i '%DATA_A%' == 'y' (
        @echo off
        set /p DATA_B="Would you like to override the default installation data path INSTALL_PATH\awrams_cm\awrams\data? [Y/N]: "
    )
    if /i '%DATA_A%' == 'n' (
        @echo off
        set /p DATA_PATH="Where is the AWRACMS data stored?"
        goto DataPath
    )
    if /i '%DATA_B%' == 'n' (
        set DATA_PATH="%INSTALL_PATH%\awrams_cm\awrams"
        goto GetPaths
    )
    if /i '%DATA_B%' == 'y' (
        @echo off
        set /p DATA_PATH="Where would you like the AWRACMS data cloned to? [C:\Users\Guest\Data]: "
        if '%DATA_PATH%' == '%INSTALL_PATH%\awrams_cm\*' (
            goto GetPaths
        )
        if '%DATA_PATH%' == 'awrams_cm\*' (
            goto GetPaths
        )
        goto DataPath
    ) else (
        echo [AWRACMS] You did not enter Y or N as a value. Try again.
        goto DataPathPrompt
    )
    goto DataPath


:GetPaths
    @echo off
    set REL_PATH=%INSTALL_PATH%
    set ABS_PATH=
    rem // Save current directory and change to target directory
    pushd %INSTALL_PATH%
    rem // Save value of CD variable (current directory)
    set ABS_PATH=%CD%
    rem // Restore original directory
    popd
    set INSTALL_PATH=%ABS_PATH%
    if /i '%DATA_B%' == 'n' (
        goto InstallChoco
    )
    if /i '%DATA_A%' == 'n' (
        goto InstallChoco
    )
    @echo off
    set REL_PATH=%DATA_PATH%
    set ABS_PATH=
    rem // Save current directory and change to target directory
    pushd %DATA_PATH%
    rem // Save value of CD variable (current directory)
    set ABS_PATH=%CD%
    rem // Restore original directory
    popd
    set DATA_PATH=%ABS_PATH%
    goto InstallChoco

REM Install Choco
:InstallChoco
    choco --version
    if '%errorlevel%' == '0' (
        echo [AWRACMS] Chocolatey has already been installed
        goto InstallWget
    ) else (
        @"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
    )
    goto InstallChoco

REM Install Wget
:InstallWget
    call wget --version
    if '%errorlevel%' == '0' (
        echo [AWRACMS] WGET has already been installed
        goto InstallClang
    ) else (
        choco install wget -y && call RefreshEnv.cmd
    )
    goto InstallWget


REM Install Wget
:InstallClang
    call clang --version
    if '%errorlevel%' == '0' (
        echo [AWRACMS] Clang has already been installed
        goto InstallMiniconda
    ) else (
        choco install visualstudio2017buildtools --package-parameters "--allWorkloads --includeRecommended --includeOptional --passive --locale en-US" && call RefreshEnv.cmd
        choco install llvm -y && call RefreshEnv.cmd
    )
    goto InstallClang


REM Install Miniconda3
:InstallMiniconda
    if exist %INSTALL_PATH%\Miniconda3\Library\bin\NUL (
        echo [AWRACMS] Miniconda3 has been found
    ) else (
        wget %MINICONDA_DL% -O miniconda.exe
        if '%errorlevel%' NEQ '0' (
            echo [AWRACMS] Failed to download miniconda3
            goto END
        )
        start /wait "" miniconda.exe /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=%INSTALL_PATH%\Miniconda3
        del %INSTALL_PATH%\miniconda.exe
    )
    goto InstallMPI

REM Download and install MPI
:InstallMPI
    mpiexec
    if '%errorlevel%' == '0' (
        echo [AWRACMS] Microsoft MPI has been installed
        goto InstallGit
    )
    wget %MSMPI_DL%
    if '%errorlevel%' NEQ '0' (
        echo [AWRACMS] Failed to download Microsoft MPI
        goto END
    )
    msmpisetup.exe -unattend | more
    del %INSTALL_PATH%\msmpisetup.exe
    call RefreshEnv.cmd
    mpiexec

REM Install GIT and GIT-LFS
:InstallGit
    git lfs install
    if '%errorlevel%' == '0' (
        echo [AWRACMS] GIT LFS has already been installed
        goto CloneAWRAMS
    ) else (
        choco install git -y && call RefreshEnv.cmd && git lfs install
    )
    goto InstallGit

REM Clone AWRACMS Repository
:CloneAWRAMS
    git clone -b "%AWRA_VERSION%" "%AWRACMS_URL%" "%INSTALL_PATH%\awrams_cm"
    if '%errorlevel%' == '0' (
        echo [AWRACMS] AWRACMS has been cloned
        goto CloneData
    )
    if exist %INSTALL_PATH%\awrams_cm\awrams\NUL (
        echo [AWRACMS] AWRACMS has been cloned
    ) else (
        RD /S /Q %INSTALL_PATH%\awrams_cm
        git clone -b "%AWRA_VERSION%" "%AWRACMS_URL%" "%INSTALL_PATH%\awrams_cm"
        if '%errorlevel%' == '0' (
            goto CloneData
        ) else (
        echo [AWRACMS] Failed to clone AWRACMS
        goto END
        )
    )

REM Clone Data Repository
:CloneData
    if /i '%DATA_A%' == 'n' (
        goto CreateCondaEnv
    )
    if exist '%DATA_PATH%\data\training\NUL' (
        goto CreateCondaEnv
    )
    git clone "%DATA_URL%" "%DATA_PATH%\data"
    if '%errorlevel%' == '0' (
        echo [AWRACMS] AWRACMS data has been cloned
    ) else (
        RD /S /Q "%DATA_PATH%\data"
        git lfs uninstall
        git lfs install --skip-smudge
        git clone "%DATA_URL%" "%DATA_PATH%\data"
        git -C "%DATA_PATH%\data" lfs pull
        git lfs install --force
    )
    goto CreateCondaEnv

REM Create miniconda environment
:CreateCondaEnv
    echo [AWRACMS] Checking if awra-cms conda environment exists
    call %INSTALL_PATH%\Miniconda3\Scripts\activate.bat base
    echo [AWRACMS] Creating awra-cms conda environment
    call conda env create -f "%INSTALL_PATH%\awrams_cm\%CONDA_ENV%"
    if '%errorlevel%' == '0' (
        echo [AWRACMS] Updating awra-cms conda environment
        call conda env update -f "%INSTALL_PATH%\awrams_cm\%CONDA_ENV%"
    )
    call conda activate awra-cms
    goto PipInstalls


REM Pip install awra-cms and mpi4py
:PipInstalls
    pip freeze | findstr "awra"
    if '%errorlevel%' NEQ '0' (
        echo [AWRACMS] Installing AWRACMS
        pip install -e "%INSTALL_PATH%\awrams_cm\utils" "%INSTALL_PATH%\awrams_cm\benchmarking" "%INSTALL_PATH%\awrams_cm\models" "%INSTALL_PATH%\awrams_cm\simulation" "%INSTALL_PATH%\awrams_cm\visualisation" "%INSTALL_PATH%\awrams_cm\calibration" "%INSTALL_PATH%\awrams_cm\cluster"
    )
    pip freeze | findstr "mpi4py"
    if '%errorlevel%' NEQ '0' (
        echo [AWRACMS] Installing MPI4PY
        pip install mpi4py=="%MPI4PY_VERSION%"
    )
    goto CreateLinks


REM Create symbolic links
:CreateLinks
    if /i '%DATA_B%' == 'y' (
        mklink /J %INSTALL_PATH%\awrams_cm\awrams\data %DATA_PATH%\data
        echo [AWRACMS] Data path symbolic link has been created
    )
    mklink /J %USERPROFILE%\.awrams %INSTALL_PATH%\awrams_cm\awrams
    dir %USERPROFILE% | findstr ".awrams [%INSTALL_PATH%\awrams_cm\awrams]"
    if '%errorlevel%' == '0' (
        echo [AWRACMS] Symbolic link has been created
        goto Activation
    ) else (
        goto CreateLinks
    )

REM Create activation script
:Activation
    (
    echo call %INSTALL_PATH%\Miniconda3\Scripts\activate.bat awra-cms
    IF "%PYTHONPATH%"=="" (
        echo set PYTHONPATH=%INSTALL_PATH%\awrams_cm\awra\code\user
    ) ELSE (
        echo set PYTHONPATH=%INSTALL_PATH%\awrams_cm\awra\code\user;%PYTHONPATH%
    )
    echo set AWRAMS_BASE_PATH=%INSTALL_PATH%\awrams_cm\awrams
    IF NOT "%DATA_PATH%"=="" (
        echo set AWRAMS_DATA_PATH=%DATA_PATH%
    )
    echo cmd /k echo [AWRACMS] AWRACMS has been activated
    )>"%INSTALL_PATH%\awrams_cm\activation.bat"
    goto RunTests


:RunTests
    if /i '%DATA_A%' == y (
        call %INSTALL_PATH%\Miniconda3\Scripts\activate.bat awra-cms
        cd %INSTALL_PATH%\awrams_cm\packages
        python setup.py nosetests
        if '%errorlevel%' == '0' (
            echo [AWRACMS] Tests passed
            goto Finish
        ) else (
            echo [AWRACMS] Tests failed
            goto Finish
        )
    )
    echo [AWRACMS] Not running tests. Data has not been cloned.
    goto Finish


:Finish
    echo [AWRACMS] AWRACMS has been installed go to %INSTALL_PATH%\awrams_cm and click on activation.bat
    start %INSTALL_PATH%\awrams_cm
    goto END


:END
pause

