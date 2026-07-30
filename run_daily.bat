@echo off
chcp 65001 >nul
title A-Insight Pro Daily Pipeline V4.0
cd /d D:\DProjectsA-Insight-Pro

set LOGFILE=logs\daily_%date:~0,10%.log
set PYTHON=venv\Scripts\python.exe

echo ============================================== > %LOGFILE%
echo A-Insight Pro Daily Pipeline V4.0 >> %LOGFILE%
echo Date: %date% %time% >> %LOGFILE%
echo ============================================== >> %LOGFILE%

echo.
echo ========================================
echo  A-Insight Pro V4.0 Daily Pipeline
echo  %date% %time%
echo ========================================
echo.

echo [1/10] Updating market data...
%PYTHON% -c "exec(open('ai/update_data.py', encoding='utf-8').read())" >> %LOGFILE% 2>&1
if %errorlevel% neq 0 echo   WARNING: Data update failed, using existing data
echo   Done.

echo [2/10] Feature engineering...
%PYTHON% -c "from data_center.feature_engine import run; run()" >> %LOGFILE% 2>&1
echo   Done.

echo [3/10] Market emotion V4.0 + Smart Money...
%PYTHON% -m ai.emotion >> %LOGFILE% 2>&1
echo   Done.

echo [4/10] AI Prediction (RF model)...
%PYTHON% -m ai.predict >> %LOGFILE% 2>&1
echo   Done.

echo [5/10] AI Scoring + Ranking...
%PYTHON% -m ai.scoring >> %LOGFILE% 2>&1
echo   Done.

echo [6/10] Historical Review (bootstrap samples)...
%PYTHON% -c "import warnings; warnings.filterwarnings('ignore'); from ai.historical_review import run; run(30, None)" >> %LOGFILE% 2>&1
echo   Done.

echo [7/10] Self-learning feedback...
%PYTHON% -m ai.self_learning >> %LOGFILE% 2>&1
echo   Done.

echo [8/10] Weight learning (auto-adjust)...
%PYTHON% -m ai.weight_learning >> %LOGFILE% 2>&1
echo   Done.

echo [9/10] Daily Report to Desktop...
%PYTHON% -m ai.daily_report >> %LOGFILE% 2>&1
echo   Done.

echo [10/10] Push notification...
%PYTHON% -m ai.notify >> %LOGFILE% 2>&1
echo   Done.

echo. >> %LOGFILE%
echo ============================================== >> %LOGFILE%
echo Pipeline Complete: %date% %time% >> %LOGFILE%
echo ============================================== >> %LOGFILE%

echo.
echo ========================================
echo  Pipeline Complete!
echo  Report on Desktop + Email sent
echo  %date% %time%
echo ========================================

if "%1"=="-q" goto end
pause
:end
