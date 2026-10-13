@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title Zoom UA Subtitles - НЕ ЗАКРИВАЙТЕ це вікно під час лекції

set "ROOT=%~dp0"
set "RT=%ROOT%.runtime"
set "UV=%RT%\tools\uv\uv.exe"
set "PY=%RT%\venv\Scripts\python.exe"
set "UV_CACHE_DIR=%RT%\uv-cache"
set "UV_PYTHON_INSTALL_DIR=%RT%\python"
set "UV_LINK_MODE=copy"
set "HF_HOME=%RT%\models\hf"
set "PYTHONUTF8=1"
if not exist "%RT%\tools" mkdir "%RT%\tools"

rem ---- fast path: already installed and requirements unchanged
if exist "%PY%" if exist "%RT%\installed.txt" (
  fc /b "%ROOT%requirements.txt" "%RT%\installed.txt" >nul 2>&1 && goto :models
)

echo.
echo  ================================================================
echo    Перший запуск: встановлення (5-15 хвилин, лише один раз)
echo  ================================================================
echo.

rem ---- 1. uv (tiny tool that installs Python + packages; no admin rights needed)
if not exist "%UV%" (
  echo  [1/4] Завантаження інсталятора Python...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile '%RT%\tools\uv.zip' -UseBasicParsing; Expand-Archive -Force '%RT%\tools\uv.zip' '%RT%\tools\uv'; exit 0 } catch { exit 1 }"
  if errorlevel 1 goto :err_net
  if not exist "%UV%" goto :err_net
)

rem ---- 2. Python 3.12 + virtual environment (inside this folder)
if not exist "%PY%" (
  echo  [2/4] Встановлення Python 3.12...
  "%UV%" venv --python 3.12 "%RT%\venv"
  if errorlevel 1 goto :err_py
)

rem ---- 3. packages
echo  [3/4] Встановлення компонентів (може тривати кілька хвилин)...
"%UV%" pip install --python "%PY%" -r "%ROOT%requirements.txt"
if errorlevel 1 goto :err_pip
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
  echo        Знайдено відеокарту NVIDIA - встановлюю бібліотеки CUDA...
  "%UV%" pip install --python "%PY%" -r "%ROOT%requirements-gpu.txt"
  if errorlevel 1 echo        [!] CUDA-бібліотеки не встановились - працюватиме на процесорі.
)
copy /y "%ROOT%requirements.txt" "%RT%\installed.txt" >nul

:models
rem ---- 4. speech model + cloudflared (skipped instantly when present)
"%PY%" -m app.setup_models
if errorlevel 1 goto :err_model

echo.
echo  Запуск програми... Панель керування відкриється в браузері.
echo.
"%PY%" -m app %*
if errorlevel 1 (
  echo.
  echo  [!] Програма завершилась з помилкою. Подробиці - у папці logs.
  pause
)
exit /b 0

:err_net
echo.
echo  [X] Не вдалося завантажити файли. Перевірте підключення до інтернету
echo      і запустіть START.bat ще раз.
pause & exit /b 1
:err_py
echo.
echo  [X] Не вдалося встановити Python. Перевірте інтернет і запустіть START.bat ще раз.
pause & exit /b 1
:err_pip
echo.
echo  [X] Не вдалося встановити компоненти. Перевірте інтернет і запустіть START.bat ще раз.
echo      Якщо не допомагає - видаліть папку .runtime\venv і спробуйте знову.
pause & exit /b 1
:err_model
echo.
echo  [X] Не вдалося завантажити модель розпізнавання мовлення.
echo      Перевірте інтернет і місце на диску (потрібно ~3 ГБ) і запустіть START.bat ще раз.
pause & exit /b 1
