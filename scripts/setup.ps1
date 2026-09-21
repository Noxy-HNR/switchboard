<#
.SYNOPSIS
    Create Switchboard's project-local environment and check the machine.

.DESCRIPTION
    Installs into C:\AI\Switchboard\.venv only. Nothing global is touched, and
    no other project's dependencies are changed. Safe to re-run.
#>

[CmdletBinding()]
param(
    [switch]$SkipDoctor
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step([string]$Text) { Write-Host "`n[*] $Text" -ForegroundColor Cyan }
function OK([string]$Text)   { Write-Host "[OK] $Text" -ForegroundColor Green }
function Warn([string]$Text) { Write-Host "[!] $Text" -ForegroundColor Yellow }

Step "Checking uv"
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    throw "uv was not found on PATH. Install it from https://docs.astral.sh/uv/ and re-run."
}
OK "uv $(& uv --version)"

Step "Syncing the project environment (.venv)"
& uv sync
if ($LASTEXITCODE -ne 0) { throw "uv sync failed with exit code $LASTEXITCODE" }
OK "Dependencies installed into $Root\.venv"

Step "Preparing directories"
foreach ($dir in @('data', 'logs', 'reports', 'backups')) {
    $path = Join-Path $Root $dir
    if (-not (Test-Path $path)) { New-Item -ItemType Directory -Path $path -Force | Out-Null }
}
OK "data, logs, reports and backups are present"

Step "Checking for a .env"
$envFile = Join-Path $Root '.env'
$examplePath = Join-Path $Root '.env.example'
if (-not (Test-Path $envFile)) {
    if (Test-Path $examplePath) {
        Copy-Item $examplePath $envFile
        OK "Created .env from .env.example (no API key set: Switchboard runs local-only)"
    }
} else {
    OK ".env already exists (left untouched)"
}

Step "Verifying the package imports"
& uv run python -c "import switchboard, switchboard.app; print('switchboard', switchboard.__version__ if hasattr(switchboard,'__version__') else '0.1.0')"
if ($LASTEXITCODE -ne 0) { throw "The package failed to import." }

if (-not $SkipDoctor) {
    Step "Running diagnostics"
    & uv run switchboard doctor
    if ($LASTEXITCODE -ne 0) {
        Warn "doctor reported failures (above). That is expected if the AI stack is not running."
    }
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  Start Switchboard :  .\scripts\start.ps1" -ForegroundColor Cyan
Write-Host "  Diagnose          :  .\scripts\doctor.ps1" -ForegroundColor Cyan
Write-Host ""
