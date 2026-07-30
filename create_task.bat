@echo off
schtasks /Create /SC DAILY /TN "A-Insight-Daily" /TR "wscript.exe D:\DProjectsA-Insight-Pro\run_silent.vbs" /ST 08:30 /F
echo Done. Check: schtasks /Query /TN "A-Insight-Daily"
pause
