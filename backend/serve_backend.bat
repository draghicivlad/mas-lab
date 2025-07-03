@echo off
REM --------------------------------------------
REM  Serve Backend - launches Flask server
REM --------------------------------------------
REM Store backend directory and project root
set "BACKEND_DIR=%~dp0"
pushd "%BACKEND_DIR%.."   REM project root

if not exist "%BACKEND_DIR%venv" (
    echo [Backend] Creating virtual environment...
    python -m venv "%BACKEND_DIR%venv"
)

call "%BACKEND_DIR%venv\Scripts\activate"

REM Ensure requirements are installed (silent)
pip install -q -r "%BACKEND_DIR%requirements.txt"

set FLASK_APP=backend.app
python -m flask run --host 0.0.0.0 --port 5000

popd 