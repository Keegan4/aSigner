@echo off
rem Double-click to start the training site on this PC and show the link to share.
rem Runs start-on-windows.ps1 in this folder; see that file for what it does.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-on-windows.ps1" %*
echo.
pause
