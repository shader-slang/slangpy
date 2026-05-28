@echo off

set DIR=%~dp0
set PROJECT_DIR=%DIR%..\
set CLANG_FORMAT=%PROJECT_DIR%tools\host\clang-format\clang-format.exe

pushd %PROJECT_DIR%
call python %DIR%run_clang_format.py --clang-format-executable=%CLANG_FORMAT% -i -r src
popd
