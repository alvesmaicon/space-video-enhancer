@echo off
rem Double-click launcher: runs run.ps1 without touching the machine-wide execution policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
