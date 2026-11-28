<#
.SYNOPSIS
    Stores a bot VM's Proton VPN WireGuard config in this PC's Credential Manager. Run it yourself, in
    your own terminal: the config holds the VM's private key.

.DESCRIPTION
    Make the config on account.protonvpn.com > Downloads > WireGuard configuration: platform Windows,
    a name per VM (e.g. the VM's machine name), a server (UK, for the farm), then "Create" and download
    the .conf file. One config per VM: each is a device of its own on the account, and each VM has its
    own key, so a config is never used in two VMs.

    This checks the file (an [Interface] with a PrivateKey and an IPv4 Address, one [Peer] with an
    Endpoint as IPv4:port), stores it as "wizard-eyes/vpn/<VM>" and deletes the file (-Keep keeps it).
    Only the endpoint is printed, never the key. bot-control.ps1 puts it into the VM when the role's
    farm.json has "vpn": {"provider": "proton"} (vm\guest-wireguard.ps1).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File botfarm\vpn-config.ps1 -Vm wizard-eyes-vm5 -Path "$env:USERPROFILE\Downloads\vm5-UK-12.conf"
#>
param(
    [Parameter(Mandatory = $true)][string]$Vm,
    [Parameter(Mandatory = $true)][string]$Path,
    [switch]$Keep
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'lib\BotFarm.psm1') -Force

$text = [IO.File]::ReadAllText((Resolve-Path $Path))
$bad = @()
if ($text -notmatch '(?m)^\s*\[Interface\]') { $bad += 'no [Interface]' }
if ($text -notmatch '(?m)^\s*PrivateKey\s*=\s*\S{40,}') { $bad += 'no PrivateKey' }
if ($text -notmatch '(?m)^\s*Address\s*=\s*\d+\.\d+\.\d+\.\d+') { $bad += 'no IPv4 Address' }
if (([regex]::Matches($text, '(?m)^\s*\[Peer\]')).Count -ne 1) { $bad += 'not exactly one [Peer]' }
if ($text -notmatch '(?m)^\s*Endpoint\s*=\s*(\d+\.\d+\.\d+\.\d+:\d+)\s*$') { $bad += 'no Endpoint = IPv4:port' }
if ($bad) { throw "not a usable WireGuard config: $($bad -join ', ')" }
$endpoint = $Matches[1]

Set-StoredSecret (Get-VpnConfigTarget $Vm) $text.Trim()
$text = $null
if (-not (Get-StoredSecret (Get-VpnConfigTarget $Vm))) { throw 'stored, but it could not be read back' }
"stored for $Vm as $(Get-VpnConfigTarget $Vm) (server $endpoint)"
if (-not $Keep) { Remove-Item $Path -Force; "deleted $Path" }
