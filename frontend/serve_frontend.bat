@echo off
REM --------------------------------------------
REM  Serve Frontend - launches Vite dev server
REM --------------------------------------------
cd /d "%~dp0"
echo [Frontend] Working dir %cd%

call npm install --silent
call npm run dev

echo Frontend exited. Press any key to close.
pause> 