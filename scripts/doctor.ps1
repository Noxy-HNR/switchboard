<#
.SYNOPSIS
    Diagnose Switchboard and the stack around it.

.DESCRIPTION
    Works with everything stopped, which is when it matters most. Checks the
    configuration, the model registry, llama-server, the dashboard, Switchboard
    itself, the decision providers, the database, privacy settings, the cptr and
    start.ps1 integrations, and host memory.

.PARAMETER Json
    Emit the machine-readable report instead of the terminal one.
#>

[CmdletBinding()]
param(
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if ($Json) {
    & uv run switchboard doctor --json
} else {
    & uv run switchboard doctor
}
exit $LASTEXITCODE
