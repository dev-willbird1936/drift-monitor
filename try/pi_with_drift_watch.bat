@echo off
rem Start Pi in the current folder with the drift monitor watching every turn.
rem The scoring server starts hidden on port 8777 if it is not already running (about 30 seconds the first time).
rem Override before running if you like:  set DRIFT_THRESHOLD=40   set DRIFT_ADVISOR=none
set "DRIFT_SCORE_URL=http://127.0.0.1:8777"
if "%DRIFT_THRESHOLD%"=="" set "DRIFT_THRESHOLD=30"
if "%DRIFT_ADVISOR%"=="" set "DRIFT_ADVISOR=claude-bridge-1/claude-haiku-5-5"
if /i "%DRIFT_ADVISOR%"=="none" set "DRIFT_ADVISOR="
if "%DRIFT_LOG%"=="" set "DRIFT_LOG=%TEMP%\drift_watch.jsonl"
curl -s -f %DRIFT_SCORE_URL%/health >nul 2>&1
if errorlevel 1 (
  echo Starting the drift scoring server...
  set "DRIFT_PORT=8777"
  if "%DRIFT_PYTHON%"=="" set "DRIFT_PYTHON=python"
  call "%%DRIFT_PYTHON%%" -X utf8 "%~dp0..\scratch\exposure_start_server.py"
)
pi -e "%~dp0..\harness\pi_drift_watch.ts" %*
