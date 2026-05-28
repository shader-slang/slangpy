: This script is fetching all dependencies.

@echo off
setlocal

echo Updating git submodules ...

where /q git
if errorlevel 1 (
    echo Cannot find git on PATH! Please initialize submodules manually and rerun.
    exit /b 1
) ELSE (
    git submodule sync --recursive
    git submodule update --init --recursive
)

if not exist %~dp0\.vscode\ (
    echo Setting up VS Code workspace ...
    xcopy %~dp0\.vscode-default\ %~dp0\.vscode\ /y
)

exit /b 0
