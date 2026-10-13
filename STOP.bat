@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Зупинка програми...
set "PORT=8765"
for /f "usebackq delims=" %%p in (`powershell -NoProfile -Command "try { (Get-Content -Raw 'config.json' | ConvertFrom-Json).port } catch { 8765 }"`) do set "PORT=%%p"
curl -s -m 5 -X POST -H "X-Control: 1" http://127.0.0.1:%PORT%/api/shutdown >nul 2>&1
timeout /t 3 /nobreak >nul
echo Готово. Програму зупинено.
timeout /t 2 >nul
