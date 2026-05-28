#!/bin/bash

DIR=$(dirname "$0")
PROJECT_DIR=${DIR}/..
CLANG_FORMAT=${PROJECT_DIR}/tools/host/clang-format/clang-format

pushd ${PROJECT_DIR} > /dev/null
python ${DIR}/run_clang_format.py --clang-format-executable=${CLANG_FORMAT} -i -r src
popd > /dev/null
