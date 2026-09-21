<#
.SYNOPSIS
    Run Switchboard's test suite.

.DESCRIPTION
    Offline by default: every test uses mock transports, so this passes with the
    AI stack stopped and with no network.

.PARAMETER Path
    Limit the run to one file or test id.

.PARAMETER Verbose_
    Show individual test names.
#>

[CmdletBinding()]
param(
    [string]$Path = 'tests',
    [switch]$Verbose_
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$arguments = @('run', 'pytest', $Path)
if ($Verbose_) { $arguments += '-v' } else { $arguments += '-q' }

& uv @arguments
exit $LASTEXITCODE
