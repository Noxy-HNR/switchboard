<#
.SYNOPSIS
    Start the Switchboard service.

.DESCRIPTION
    Launches uvicorn from the project-local environment, waits for /health, and
    prints the URLs. Switchboard does not start llama-server or the dashboard;
    it routes to whatever C:\AI\start.ps1 has running, and reports honestly when
    they are down.

.PARAMETER Foreground
    Run in this window instead of detaching (Ctrl+C to stop).

.PARAMETER TimeoutSeconds
    How long to wait for the service to answer. Default 45.
#>

[CmdletBinding()]
param(
    [switch]$Foreground,
    [int]$TimeoutSeconds = 45
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LogDir = Join-Path $Root 'logs'
$PidFile = Join-Path $Root 'logs\switchboard.pid'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

# Read host and port from the config so this script never disagrees with it.
$configPath = Join-Path $Root 'config\switchboard.toml'
$listenHost = '127.0.0.1'
$port = 8002
if (Test-Path $configPath) {
    $content = Get-Content $configPath -Raw
    if ($content -match '(?m)^\s*host\s*=\s*"([^"]+)"') { $listenHost = $Matches[1] }
    if ($content -match '(?m)^\s*port\s*=\s*(\d+)')      { $port = [int]$Matches[1] }
}
$baseUrl = "http://${listenHost}:${port}"

function Test-Health {
    try {
        $response = Invoke-WebRequest -Uri "$baseUrl/health" -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -eq 200
    } catch { return $false }
}

if (Test-Health) {
    Write-Host "[OK] Switchboard is already running at $baseUrl" -ForegroundColor Green
    exit 0
}

# Something else on the port is a configuration problem, not a restart.
$existing = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($existing) {
    $owner = (Get-Process -Id $existing.OwningProcess -ErrorAction SilentlyContinue).ProcessName
    throw "Port $port is already in use by $owner (PID $($existing.OwningProcess)). Change server.port in config\switchboard.toml."
}

if ($Foreground) {
    Write-Host "Starting Switchboard at $baseUrl (Ctrl+C to stop)" -ForegroundColor Cyan
    & uv run switchboard serve
    exit $LASTEXITCODE
}

$stdout = Join-Path $LogDir 'switchboard.out.log'
$stderr = Join-Path $LogDir 'switchboard.err.log'

$process = Start-Process -FilePath 'uv' -ArgumentList @('run', 'switchboard', 'serve') `
    -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr

# Record the start time with the PID. Windows reuses PIDs, so a bare number in
# a stale file could name an unrelated process by the time stop.ps1 reads it;
# the pair (PID, start time) identifies exactly one process, ever.
$started = (Get-Process -Id $process.Id).StartTime.ToUniversalTime().Ticks
Set-Content -Path $PidFile -Value "$($process.Id)|$started" -Encoding UTF8
Write-Host "[*] Switchboard starting (PID $($process.Id))" -ForegroundColor Cyan

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    if (Test-Health) {
        Write-Host "[OK] Switchboard is ready" -ForegroundColor Green
        Write-Host ""
        Write-Host "  OpenAI endpoint : $baseUrl/v1        (model: switchboard-auto)" -ForegroundColor Cyan
        Write-Host "  Dashboard       : $baseUrl/dashboard" -ForegroundColor Cyan
        Write-Host "  API docs        : $baseUrl/docs" -ForegroundColor Cyan
        Write-Host "  MCP             : $baseUrl/mcp" -ForegroundColor Cyan
        Write-Host ""
        exit 0
    }
    if ($process.HasExited) {
        Write-Host "[X] Switchboard exited immediately (code $($process.ExitCode))." -ForegroundColor Red
        Write-Host "--- last lines of $stderr ---" -ForegroundColor Yellow
        Get-Content $stderr -Tail 25 -ErrorAction SilentlyContinue
        exit 1
    }
    Start-Sleep -Milliseconds 500
}

Write-Host "[X] Switchboard did not answer within $TimeoutSeconds seconds." -ForegroundColor Red
Get-Content $stderr -Tail 25 -ErrorAction SilentlyContinue
exit 1
