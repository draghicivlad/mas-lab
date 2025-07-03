@echo off
REM ------------------------------------------------------------
REM  run_simulation_windows.bat
REM  Opens two separate CMD windows: backend & frontend
REM ------------------------------------------------------------

set "ROOT_DIR=%~dp0"

start "MAPF Backend" "%ROOT_DIR%backend\serve_backend.bat"
start "MAPF Frontend" "%ROOT_DIR%frontend\serve_frontend.bat"

echo Backend and Frontend are launching... 