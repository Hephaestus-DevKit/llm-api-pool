@echo off
setlocal
echo ================================================
echo Building portable Windows app: LLM API Pool
echo ================================================
cd /d "%~dp0"

set "PYTHON_CMD=python"
set "CONDA_EXE="
set "CONDA_ENV="

where conda.exe >nul 2>nul
if not errorlevel 1 set "CONDA_EXE=conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_EXE if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"

if defined CONDA_EXE (
  "%CONDA_EXE%" env list | findstr /R /C:"^happy[ ][ ]*" >nul 2>nul
  if not errorlevel 1 (
    set "CONDA_ENV=happy"
    set "PYTHON_CMD="%CONDA_EXE%" run -n happy python"
    echo Using Conda env: happy
  ) else (
    set "CONDA_ENV=base"
    set "PYTHON_CMD="%CONDA_EXE%" run -n base python"
    echo Conda env happy not found; using base.
  )
) else (
  echo Conda not found on PATH; using python from PATH.
)

set "PY_ARCH=unknown"
for /f "delims=" %%A in ('call %PYTHON_CMD% -c "import platform; print(platform.machine())"') do set "PY_ARCH=%%A"
set "ASSET_NAME=llm-pool-windows-x64"
if /I "%PY_ARCH%"=="ARM64" set "ASSET_NAME=llm-pool-windows-arm64-experimental"
echo Python architecture: %PY_ARCH%
if /I not "%PY_ARCH%"=="ARM64" if /I "%PROCESSOR_ARCHITEW6432%"=="ARM64" echo Windows on Arm detected; building x64 package under emulation.

if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

%PYTHON_CMD% -m pip install --upgrade pip
if errorlevel 1 exit /b 1
if exist requirements-lock.txt (
  %PYTHON_CMD% -m pip install -r requirements-lock.txt
) else (
  %PYTHON_CMD% -m pip install -r requirements.txt
)
if errorlevel 1 exit /b 1
%PYTHON_CMD% -m pip install pyinstaller pytest httpx
if errorlevel 1 exit /b 1

echo.
echo Running test suite before packaging...
%PYTHON_CMD% -m pytest
if errorlevel 1 (
  echo Tests failed; refusing to build.
  exit /b 1
)
%PYTHON_CMD% scripts\validate_web_contracts.py
if errorlevel 1 exit /b 1

%PYTHON_CMD% -m PyInstaller ^
  --onedir ^
  --name llm-pool ^
  --add-data "requirements.txt;." ^
  --add-data ".env.example;." ^
  --add-data "examples;examples" ^
  --add-data "dashboard.html;." ^
  --add-data "README.md;." ^
  --console ^
  --noconfirm ^
  main.py
if errorlevel 1 exit /b 1

echo.
echo Built: dist\llm-pool\llm-pool.exe
echo.
echo Creating launcher: llm-pool-launch.bat
(
echo @echo off
echo cd /d "%%~dp0\dist\llm-pool"
echo llm-pool.exe %%*
) > llm-pool-launch.bat

echo.
echo Portable app folder: dist\llm-pool
echo Creating zip and checksum: dist\%ASSET_NAME%.zip
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path '.\dist\llm-pool' -DestinationPath '.\dist\%ASSET_NAME%.zip' -Force; $h = Get-FileHash '.\dist\%ASSET_NAME%.zip' -Algorithm SHA256; ($h.Hash.ToLowerInvariant() + '  %ASSET_NAME%.zip') | Set-Content '.\dist\%ASSET_NAME%.zip.sha256' -Encoding ascii"
if errorlevel 1 exit /b 1
echo Double-click llm-pool-launch.bat or dist\llm-pool\llm-pool.exe.
echo API base URL: http://localhost:8080/v1
echo Dashboard: http://localhost:8080/
echo Build complete.
