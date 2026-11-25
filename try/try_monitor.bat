@echo off
rem Score the example sessions, or drag your own session .txt file onto this .bat.
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if "%DRIFT_PYTHON%"=="" set "DRIFT_PYTHON=python"
"%DRIFT_PYTHON%" -X utf8 try_monitor.py %*
if not "%~1"=="" pause
