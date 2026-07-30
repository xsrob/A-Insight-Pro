@echo off
chcp 65001 >nul
title A-Insight Pro Dashboard
cd /d D:\DProjectsA-Insight-Pro
echo Launching A-Insight Pro Dashboard...
call venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.port 8501
pause
