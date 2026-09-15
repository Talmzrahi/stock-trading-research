@echo off
rem Task Scheduler entry point: run from the repo root, append output to a log.
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m trader.run_daily >> "data\run_daily.log" 2>&1
