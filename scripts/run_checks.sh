#!/bin/bash
failedtests=0
succeededtests=0
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "${DIR}"
cd ../packages

if python setup.py nosetests --with-coverage --cover-package=awrams; then
    (( succeededtests=succeededtests+1 ))
else
    (( failedtests=failedtests+1 ))
fi


if [[ "$failedtests" -gt 0 ]]; then
    echo "Failed"
    exit 1
else
    echo "Passed"
fi
