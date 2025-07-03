@echo off
REM ------------------------------------------------------------
REM  install_dependencies_windows.bat
REM  One-shot installer for backend (Python) and frontend (Node)
REM ------------------------------------------------------------

REM Remember original path
set ROOT_DIR=%~dp0

REM ---------- Backend ----------
pushd "%ROOT_DIR%backend"

REM Create or reuse virtualenv (folder "venv")
if not exist venv (
    echo [Backend] Creating virtual environment...
    python -m venv venv
)

echo [Backend] Activating virtual environment & installing requirements...
call venv\Scripts\activate
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
call deactivate

popd

REM ---------- Frontend ----------
pushd "%ROOT_DIR%frontend"

echo [Frontend] Installing npm packages...
npm install --silent

popd

echo.
echo ===============================
echo  All dependencies installed!
echo ===============================
pause 