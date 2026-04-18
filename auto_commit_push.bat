@echo off
setlocal EnableExtensions

REM Auto-stage, commit, and push changes for the current repository.
REM Usage:
REM   auto_commit_push.bat "your commit message"
REM   auto_commit_push.bat "your commit message" origin

if "%~1"=="" (
  echo Usage: %~nx0 "commit message" [remote]
  exit /b 1
)

if not "%~3"=="" (
  echo Usage: %~nx0 "commit message" [remote]
  exit /b 1
)

set "commit_message=%~1"
set "remote=%~2"
if "%remote%"=="" set "remote=origin"

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
  echo Error: this script must run inside a git repository.
  exit /b 1
)

for /f "delims=" %%I in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "branch=%%I"
if not defined branch (
  echo Error: unable to determine the current branch.
  exit /b 1
)

if /i "%branch%"=="HEAD" (
  echo Error: detached HEAD state detected. Checkout a branch first.
  exit /b 1
)

REM Stage all tracked and untracked changes.
git add -A
if errorlevel 1 exit /b 1

REM Exit early if there is nothing staged.
git diff --cached --quiet >nul 2>&1
if not errorlevel 1 (
  echo No staged changes to commit.
  exit /b 0
)

git commit -m "%commit_message%"
if errorlevel 1 exit /b 1

git push "%remote%" "%branch%"
if errorlevel 1 exit /b 1

echo Done: committed and pushed to %remote%/%branch%
exit /b 0
