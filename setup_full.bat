@echo off
chcp 65001 >nul
title A-Insight Pro — Full Auto Setup
echo ==============================================
echo   A-Insight Pro 全自动推送设置
echo ==============================================
echo.

REM Check admin
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 请右键此文件 → 以管理员身份运行!
    echo.
    pause
    exit /b 1
)

echo [1/3] 删除旧任务...
schtasks /Delete /TN "A-Insight-Daily" /F >nul 2>&1

echo [2/3] 创建SYSTEM级别计划任务...
schtasks /Create /SC DAILY /TN "A-Insight-Daily" /TR "wscript.exe D:\DProjectsA-Insight-Pro\run_silent.vbs" /ST 08:30 /RU SYSTEM /RL HIGHEST /F

if %errorlevel% neq 0 (
    echo [错误] 任务创建失败!
    pause
    exit /b 1
)

echo [3/3] 验证任务...
schtasks /Query /TN "A-Insight-Daily" /FO LIST | findstr /C:"TaskName" /C:"Next Run" /C:"Schedule" /C:"Run as"

echo.
echo ==============================================
echo   设置完成!
echo   每天 8:30 自动运行 — 关机/休眠/未登录都能推送
echo.
echo   [可选] BIOS设置每天8:25自动开机
echo   进BIOS → Power → Auto Power On → 08:25
echo ==============================================
echo.
echo 按任意键退出...
pause >nul
