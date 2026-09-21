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

# The pid file first: it names the process we started, not whoever holds the port.
if (Test-Path $PidFile) {
    $recorded = (Get-Content $PidFile -Raw).Trim()
    if ($recorded -match '^\d+$') {
        $process = Get-Process -Id ([int]$recorded) -ErrorAction SilentlyContinue
        if ($process) {
            # Kill the tree: `uv run` launches python as a child.
            & taskkill /PID $recorded /T /F 2>&1 | Out-Null
            Write-Host "[OK] Stopped Switchboard (PID $recorded)" -ForegroundColor Green
            $stopped = $true
        }
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

if (-not $stopped) {
    $connection = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($connection -and $connection.OwningProcess -ne 4) {
        $process = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
        # Only stop it if it really is ours.
        $commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($connection.OwningProcess)" -ErrorAction SilentlyContinue).CommandLine
        if ($commandLine -match 'switchboard') {
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
