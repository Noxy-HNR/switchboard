<#
.SYNOPSIS
    Stop the Switchboard service.

.DESCRIPTION
    Stops only Switchboard's own process. llama-server, the dashboard and cptr
    are left alone: stopping Switchboard must never take the stack down with it.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $Root 'logs\switchboard.pid'

$configPath = Join-Path $Root 'config\switchboard.toml'
$port = 8002
if (Test-Path $configPath) {
    $content = Get-Content $configPath -Raw
    if ($content -match '(?m)^\s*port\s*=\s*(\d+)') { $port = [int]$Matches[1] }
}

$stopped = $false

function Test-IsSwitchboard([int]$ProcessId) {
    $info = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    return ($info -and $info.CommandLine -match 'switchboard')
}

# The pid file first: it names the process we started, not whoever holds the port.
# It holds "PID|start-ticks"; both must match, because Windows reuses PIDs and a
# stale file could otherwise name - and this script would kill - an unrelated
# process tree.
if (Test-Path $PidFile) {
    $recorded = (Get-Content $PidFile -Raw).Trim()
    if ($recorded -match '^(\d+)\|(\d+)$') {
        $recordedPid = [int]$Matches[1]
        $recordedTicks = [long]$Matches[2]
        $process = Get-Process -Id $recordedPid -ErrorAction SilentlyContinue
        $sameProcess = $process -and ($process.StartTime.ToUniversalTime().Ticks -eq $recordedTicks)
        if ($sameProcess -and (Test-IsSwitchboard $recordedPid)) {
            # Kill the tree: `uv run` launches python as a child.
            & taskkill /PID $recordedPid /T /F 2>&1 | Out-Null
            Write-Host "[OK] Stopped Switchboard (PID $recordedPid)" -ForegroundColor Green
            $stopped = $true
        } elseif ($process) {
            Write-Host "[!] PID $recordedPid now belongs to another process; left it alone." -ForegroundColor Yellow
        }
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

if (-not $stopped) {
    $connection = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($connection -and $connection.OwningProcess -ne 4) {
        $process = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
        # Only stop it if it really is ours.
        if (Test-IsSwitchboard $connection.OwningProcess) {
            & taskkill /PID $connection.OwningProcess /T /F 2>&1 | Out-Null
            Write-Host "[OK] Stopped Switchboard on port $port (PID $($connection.OwningProcess))" -ForegroundColor Green
            $stopped = $true
        } else {
            Write-Host "[!] Port $port is held by $($process.ProcessName) (PID $($connection.OwningProcess)), which is not Switchboard. Left running." -ForegroundColor Yellow
        }
    }
}

if (-not $stopped) {
    Write-Host "[!] Switchboard was not running." -ForegroundColor Yellow
}
