#!/bin/bash
# Usage:
# conda_env.sh command [--non-int]
# positional arguments:
#   command
#     create
#     update
#
# optional arguments:
#   --non-int     Only install packages for non interactive use

set -e

if [ "$1" = '--help' ] || [ "$1" = '-h' ]
then
  cat << EOF
Usage:
  conda_env.sh command [--non-int]
  positional arguments:
    command
      create
      update

  optional arguments:
    --non-int     Only install packages for non interactive use
    --help | -h   Display this text
EOF
  exit 0
fi


if [ -z "$2" ]
then
  export FILE=../conda_install_env.yml
  export ENV_NAME=awra-cms
elif [ "$2" = '--non-int' ]
then
  export FILE=../conda_install_env_non_interactive.yml
  export ENV_NAME=awra-cms-non-interactive
else
  echo "Optional second argument must be '--non-int' or not provided"
  exit 1
fi


conda env "$1" -f "$FILE"
source activate "$ENV_NAME"

