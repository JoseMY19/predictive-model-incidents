<#
.SYNOPSIS
  Registers (or removes) a Windows scheduled task that runs the retraining job weekly.

.EXAMPLE
  .\scripts\schedule_retrain.ps1                       # Sundays 03:00
  .\scripts\schedule_retrain.ps1 -Day Wednesday -Time 02:30
  .\scripts\schedule_retrain.ps1 -Remove

.NOTES
  The job (python -m app.retrain) is safe to run often: it exits early unless there are >= 7 new complete days, trains in a
  staging folder and only swaps the live model if the quality gate passes. PostgreSQL must be reachable when it runs
  (scripts\db_start.ps1 starts the dedicated local cluster). Logs: logs\retrain.log
#>
param(
    [ValidateSet("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")][string]$Day = "Sunday",
    [string]$Time = "03:00",
    [string]$TaskName = "PredictiveIncidenceRetrain",
    [switch]$Remove
)

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Task '$TaskName' removed."
    return
}

$backend = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python -ErrorAction Stop).Source

$action = New-ScheduledTaskAction -Execute $python -Argument "-m app.retrain" -WorkingDirectory $backend
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Day -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Force `
    -Description "Retrains the incident forecasting models if enough new data exists (see backend/README.md)." | Out-Null

Write-Host "Task '$TaskName' registered: every $Day at $Time -> $python -m app.retrain (cwd: $backend)"
Write-Host "Run it now with:  Start-ScheduledTask -TaskName $TaskName      Logs: $backend\logs\retrain.log"
