<#
.SYNOPSIS
    Add an optional Switchboard step to C:\AI\start.ps1.

.DESCRIPTION
    Inserts a small, clearly marked block just before the stack's final health
    check. The block:
      * does nothing if C:\AI\Switchboard is missing;
      * catches its own errors, so a Switchboard problem can never stop
        llama-server, the dashboard or cptr from starting;
      * is bracketed by marker comments so -Remove takes it out exactly.

    start.ps1 is backed up to C:\AI\Switchboard\backups before any edit.

.PARAMETER Remove
    Take the block out again.

.PARAMETER DryRun
    Show what would change without writing.
#>

[CmdletBinding()]
param(
    [switch]$Remove,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Target = 'C:\AI\start.ps1'
$BackupDir = Join-Path $Root 'backups'

$BeginMarker = '# --- Switchboard (added by C:\AI\Switchboard\scripts\integrate-start.ps1) ---'
$EndMarker   = '# --- end Switchboard ---'

if (-not (Test-Path $Target)) {
    Write-Host "[X] $Target not found." -ForegroundColor Red
    exit 1
}

$original = Get-Content $Target -Raw
$hasBlock = $original.Contains($BeginMarker)

if ($Remove) {
    if (-not $hasBlock) {
        Write-Host "[!] No Switchboard block found in $Target." -ForegroundColor Yellow
        exit 0
    }
    $pattern = [regex]::Escape($BeginMarker) + '.*?' + [regex]::Escape($EndMarker) + '\r?\n?'
    $updated = [regex]::Replace($original, $pattern, '', 'Singleline')
} else {
    if ($hasBlock) {
        Write-Host "[OK] $Target already starts Switchboard." -ForegroundColor Green
        exit 0
    }

    $anchor = 'Banner "FINAL HEALTH CHECK"'
    if (-not $original.Contains($anchor)) {
        Write-Host "[X] Could not find the anchor ($anchor) in $Target." -ForegroundColor Red
        Write-Host "    start.ps1 has changed shape; add the block by hand instead - see docs\INTEGRATION.md." -ForegroundColor Yellow
        exit 1
    }

    $block = @"
$BeginMarker
# Optional: routes chat requests to the best local model. Remove this block, or
# run C:\AI\Switchboard\scripts\integrate-start.ps1 -Remove, to take it out.
# Nothing below can stop the rest of the stack: it is wrapped in try/catch.
`$SwitchboardStart = "C:\AI\Switchboard\scripts\start.ps1"
if (Test-Path `$SwitchboardStart) {
    Step "Starting Switchboard"
    try {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `$SwitchboardStart | Out-Null
        if (Test-Endpoint -Url "http://127.0.0.1:8002/health" -TimeoutSec 5) {
            OK "Switchboard         READY  :8002"
        } else {
            Warn "Switchboard did not answer on :8002 (the rest of the stack is unaffected)."
        }
    } catch {
        Warn "Switchboard failed to start: `$(`$_.Exception.Message)"
    }
}
$EndMarker

"@

    $updated = $original.Replace($anchor, $block + $anchor)
}

if ($DryRun) {
    Write-Host "[dry run] $Target would be " -NoNewline
    Write-Host $(if ($Remove) { 'stripped of the Switchboard block.' } else { 'given a Switchboard start step before the final health check.' })
    exit 0
}

New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backup = Join-Path $BackupDir "start.ps1.$stamp.bak"
Copy-Item $Target $backup

# Preserve the file's original encoding: start.ps1 is UTF-8 with a BOM, and
# PowerShell 5.1 would otherwise write it back as ANSI.
[System.IO.File]::WriteAllText($Target, $updated, (New-Object System.Text.UTF8Encoding $true))

# Syntax-check the result. A start.ps1 that cannot parse would be a bad day.
$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile($Target, [ref]$null, [ref]$errors)
if ($errors -and $errors.Count -gt 0) {
    Copy-Item $backup $Target -Force
    Write-Host "[X] The edit did not parse; restored from $backup." -ForegroundColor Red
    $errors | ForEach-Object { Write-Host "    $($_.Message)" -ForegroundColor Red }
    exit 1
}

Write-Host "[OK] $Target updated." -ForegroundColor Green
Write-Host "     Backup: $backup" -ForegroundColor DarkGray
if (-not $Remove) {
    Write-Host "     C:\AI\start.ps1 will now start Switchboard after cptr." -ForegroundColor Cyan
}
