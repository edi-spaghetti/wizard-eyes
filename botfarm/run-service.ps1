<#
Runs one of a farm's services on the host (farm.json "services"): the action of
its scheduled task (install.ps1 registers them). Python from the engine's venv,
with the engine and the farm's repo importable; the farm's folder is in
$env:WIZARD_EYES_FARM for anything it starts (e.g. bot-control.ps1).
Log: <engine>\logs\<service>.log
#>
param([Parameter(Mandatory)][string]$Service, [string]$Farm)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'lib\BotFarm.psm1') -Force
$fm = Get-Farm $Farm
$s = @($fm.Def.services | Where-Object name -eq $Service)[0]
if (-not $s) { throw "No service '$Service' in $($fm.Name)" }
$env:PYTHONPATH = "$($fm.Engine)\python;$($fm.Repo)"
$env:PYTHONUNBUFFERED = '1'
$env:WIZARD_EYES_FARM = $fm.Dir
Set-Location $fm.Repo
New-Item -ItemType Directory -Force $fm.LogDir | Out-Null
$argText = (@($s.args) | ForEach-Object {
    $a = Expand-FarmText $fm $_
    if ($a -match '\s') { "`"$a`"" } else { $a }
}) -join ' '
cmd /c "`"$($fm.Python)`" -m $($s.module) $argText >> `"$($fm.LogDir)\$Service.log`" 2>&1"
