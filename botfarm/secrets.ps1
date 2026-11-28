<#
.SYNOPSIS
    Which secrets a farm needs, and whether each is in this PC's Credential Manager (and looks right).
    Never prints a secret.

.DESCRIPTION
    Lists, for the farm: the VMs' Windows password, the Telegram bot token, each role's VPN config
    (Proton WireGuard, per VM), each role's Jagex login (wizard-eyes/jagex/<account>: user = email),
    bank PINs (kept in the VMs), the Gmail app password for email codes (optional), the R2 read config.
    -Online also asks Telegram whether the token works. Exit code 1 if anything needed is missing or bad.

    To store one, in your own terminal (it asks for the value; nothing shows):
      cmdkey /generic:<target> /user:<user> /pass
    A VM's VPN config: botfarm\vpn-config.ps1 -Vm <vm> -Path <file.conf>.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File botfarm\secrets.ps1 -Farm ..\wizard-eyes-experiments\botfarm\herbitrage -Online
#>
param([string]$Farm, [switch]$Online)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'lib\BotFarm.psm1') -Force
$fm = Get-Farm $Farm
$all = @(Get-FarmSecrets $fm -Online:$Online)
"Secrets for $($fm.Name) ($($fm.Dir))"
foreach ($s in $all) {
    $mark = switch ($s.state) { 'ok' { 'OK  ' } 'optional' { '--  ' } default { 'TODO' } }
    $line = "  $mark $($s.for): $($s.target)"
    if ($s.note) { $line += " ($($s.note))" }
    $line
    if ($s.state -ne 'ok' -and $s.howto) { "         how: $($s.howto)" }
}
$todo = @($all | Where-Object { $_.state -in 'missing', 'bad' })
if ($todo) { "`n$($todo.Count) to do"; exit 1 } else { "`nall needed secrets are there" }
