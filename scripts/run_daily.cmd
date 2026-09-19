@echo off
rem Task Scheduler entry point: run from the repo root, append output to a log.
rem Start and exit lines make a killed or failed run visible in the log; the
rem 2026-09-16 run died silently and only its first output line survived.
cd /d "%~dp0.."
echo ==== %date% %time% start >> "data\run_daily.log"
".venv\Scripts\python.exe" -m trader.run_daily >> "data\run_daily.log" 2>&1
echo ==== %date% %time% exit %errorlevel% >> "data\run_daily.log"
