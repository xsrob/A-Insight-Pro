# A-Insight Pro — Windows Task Scheduler Setup
# Run this in PowerShell as Administrator

$taskName = "A-Insight-Pro-Daily"
$batPath = "D:\DProjectsA-Insight-Pro\run_daily.bat"

# Remove existing task if any
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

# Create action: run the daily batch script
$action = New-ScheduledTaskAction -Execute $batPath -Argument "-q"

# Trigger: every day at 8:30 AM
$trigger = New-ScheduledTaskTrigger -Daily -At "08:30AM"

# Run as SYSTEM with highest privileges
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest

# Settings: wake to run, start if missed
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -WakeToRun `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

# Register
Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Force

Write-Host "Task '$taskName' created — runs daily at 08:30"
Write-Host "Check: taskschd.msc → Task Scheduler Library"
