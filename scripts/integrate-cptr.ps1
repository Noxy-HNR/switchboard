<#
.SYNOPSIS
    Add Switchboard to cptr as a chat connection and an MCP tool server.

.DESCRIPTION
    Additive: cptr's existing "Llama.cpp" connection and its existing tool
    servers are left exactly as they are, so the direct path to llama-server
    keeps working. The file is backed up first, and cptr must be stopped -
    it rewrites config.toml from its database whenever a setting changes, so an
    edit made while it runs would be lost.

.PARAMETER Remove
    Undo the integration.

.PARAMETER DryRun
    Show what would change without writing anything.

.PARAMETER Force
    Proceed even if cptr appears to be running (not recommended).
#>

[CmdletBinding()]
param(
    [switch]$Remove,
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$configPath = Join-Path $Root 'config\switchboard.toml'
$listenHost = '127.0.0.1'
$port = 8002
if (Test-Path $configPath) {
    $content = Get-Content $configPath -Raw
    if ($content -match '(?m)^\s*host\s*=\s*"([^"]+)"') { $listenHost = $Matches[1] }
    if ($content -match '(?m)^\s*port\s*=\s*(\d+)')      { $port = [int]$Matches[1] }
}
$baseUrl = "http://${listenHost}:${port}"

$cptrRunning = $null -ne (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)
if ($cptrRunning -and -not $Force -and -not $DryRun) {
    Write-Host "[X] cptr is running on port 8000." -ForegroundColor Red
    Write-Host "    cptr rewrites config.toml from its database when settings change, so" -ForegroundColor Yellow
    Write-Host "    an edit made now would be overwritten. Stop cptr first:" -ForegroundColor Yellow
    Write-Host "      Get-CimInstance Win32_Process | Where-Object { `$_.CommandLine -match 'cptr(\.exe)?\s+run' } | ForEach-Object { Stop-Process -Id `$_.ProcessId -Force }" -ForegroundColor Cyan
    Write-Host "    ...then re-run this script and start cptr again." -ForegroundColor Yellow
    exit 1
}

$arguments = @('run', 'python', (Join-Path $Root 'scripts\integrate_cptr.py'),
               '--base-url', $baseUrl,
               '--backup-dir', (Join-Path $Root 'backups'))
if ($Remove) { $arguments += '--remove' }
if ($DryRun) { $arguments += '--dry-run' }

& uv @arguments
$code = $LASTEXITCODE

if ($code -eq 0 -and -not $DryRun -and -not $Remove) {
    Write-Host ""
    Write-Host "In cptr, pick the model 'switchboard-auto' on the Switchboard connection" -ForegroundColor Cyan
    Write-Host "to get automatic routing. The Llama.cpp connection still talks to" -ForegroundColor Cyan
    Write-Host "llama-server directly, exactly as before." -ForegroundColor Cyan
}
exit $code
