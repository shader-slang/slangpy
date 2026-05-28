#!/usr/bin/env bash

# This script is fetching all dependencies.

BASE_DIR=$(dirname "$0")

echo "Updating git submodules ..."

if ! [ -x "$(command -v git)" ]; then
    echo "Cannot find git on PATH! Please initialize submodules manually and rerun."
    exit 1
else
    git submodule sync --recursive
    git submodule update --init --recursive
fi

if [ ! -d ${BASE_DIR}/.vscode ]; then
    echo "Setting up VS Code workspace ..."
    cp -rp ${BASE_DIR}/.vscode-default ${BASE_DIR}/.vscode
fi

exit 0
