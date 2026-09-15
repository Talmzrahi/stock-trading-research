# Registers the daily paper-trading run in Windows Task Scheduler.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1
#
# Runs weekdays at 14:30 New York time, converted to this PC's local time
# as of today. That leaves 80 minutes before the 15:50 ET market-on-close
# cutoff, so the one-hour drift when the US and your local clocks change
# for daylight saving on different weekends still lands before it —
# re-run this script after a DST change anyway to keep it exact.
#
# StartWhenAvailable: if the PC was asleep or off, the run happens when it
# comes back. Late runs are safe — the runner won't submit after the cutoff
# and catches up the next session.

param(
    [string]$EtTime = "14:30",
    [string]$TaskName = "PEAD paper trading daily run"
)

$root = Split-Path -Parent $PSScriptRoot
$cmd = Join-Path $root "scripts\run_daily.cmd"

$eastern = [System.TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
$etToday = [datetime]::ParseExact((Get-Date -Format "yyyy-MM-dd") + " " + $EtTime, "yyyy-MM-dd HH:mm", $null)
$local = [System.TimeZoneInfo]::ConvertTimeToUtc($etToday, $eastern).ToLocalTime()

$action = New-ScheduledTaskAction -Execute $cmd -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $local
# 3 hours: with TRADER_BROKER=alpaca the run waits until 15:45 ET to send orders.
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 3)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Refresh data, settle fills, submit market-on-close orders, write reports/YYYY-MM-DD.md" -Force | Out-Null

Write-Host "Registered '$TaskName': weekdays at $($local.ToString('HH:mm')) local time (= $EtTime ET today)."
Write-Host "Log: $root\data\run_daily.log   Reports: $root\reports\"
