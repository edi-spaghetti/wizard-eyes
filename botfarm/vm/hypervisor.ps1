<#
.SYNOPSIS
    Switch the Windows hypervisor off while bot VMs are installed, and back on.

.DESCRIPTION
    Each change needs an admin prompt and a reboot (this script never reboots).

      status  what Windows reports now; no changes, no admin
      off     save the Memory Integrity (HVCI) settings, turn Memory Integrity
              off, hypervisorlaunchtype off
      on      restore the saved Memory Integrity settings,
              hypervisorlaunchtype auto (WSL2 works again)

    Saved settings: %LOCALAPPDATA%\wizard-eyes\hypervisor-state.json. It is
    only written when missing, so a second 'off' can't overwrite the originals,
    and removed once 'on' has restored them.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File vm\hypervisor.ps1 off
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)][ValidateSet('status', 'off', 'on')][string]$Action,
    # internal: passed to the elevated copy
    [string]$StateFile = (Join-Path $env:LOCALAPPDATA 'wizard-eyes\hypervisor-state.json'),
    [switch]$Elevated
)

$ErrorActionPreference = 'Stop'
$Hvci = 'HKLM:\SYSTEM\CurrentControlSet\Control\DeviceGuard\Scenarios\HypervisorEnforcedCodeIntegrity'
$Values = 'Enabled', 'WasEnabledBy'
$LogFile = Join-Path (Split-Path $StateFile) 'hypervisor-last.log'

function Show-Status {
    $dg = Get-CimInstance -Namespace root\Microsoft\Windows\DeviceGuard -ClassName Win32_DeviceGuard
    $key = Get-ItemProperty $Hvci -ErrorAction SilentlyContinue
    "Hypervisor present now       : $((Get-CimInstance Win32_ComputerSystem).HypervisorPresent)"
    "VBS now (2 = running)        : $($dg.VirtualizationBasedSecurityStatus)"
    "Memory Integrity now         : $($dg.SecurityServicesRunning -contains 2)"
    "Memory Integrity after reboot: $(if ($key.Enabled -eq 1) { 'on' } else { 'off' })"
    "Saved settings to restore    : $(Test-Path $StateFile)"
}

function Get-LaunchType {
    $text = & bcdedit.exe /enum | Out-String
    if ($LASTEXITCODE -ne 0) { throw "bcdedit /enum failed: $text" }
    if ($text -match '(?im)^hypervisorlaunchtype\s+(\S+)') { return $Matches[1] }
    return 'not set'
}

function Set-LaunchType([string]$Type) {
    $text = & bcdedit.exe /set hypervisorlaunchtype $Type | Out-String
    if ($LASTEXITCODE -ne 0) { throw "bcdedit /set failed: $text" }
    $now = Get-LaunchType
    if ($now -ne $Type) { throw "hypervisorlaunchtype is '$now', expected '$Type'" }
}

if ($Action -eq 'status') { Show-Status; return }

if (-not $Elevated) {
    New-Item -ItemType Directory -Force (Split-Path $StateFile) | Out-Null
    Remove-Item $LogFile -ErrorAction SilentlyContinue
    $p = Start-Process powershell.exe -Verb RunAs -Wait -PassThru -WindowStyle Hidden -ArgumentList `
        "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" $Action -StateFile `"$StateFile`" -Elevated"
    if (Test-Path $LogFile) { Get-Content $LogFile }
    if ($p.ExitCode -ne 0) { throw "hypervisor.ps1 $Action failed (exit $($p.ExitCode))" }
    ''
    Show-Status
    "`nReboot for this to take effect."
    return
}

# --- elevated from here ---
try {
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole('Administrators')
    if (-not $isAdmin) { throw 'not running as administrator' }
    "hypervisorlaunchtype before: $(Get-LaunchType)" | Out-File $LogFile -Encoding utf8

    if ($Action -eq 'off') {
        if (-not (Test-Path $StateFile)) {
            $key = Get-ItemProperty $Hvci
            $saved = [ordered]@{}
            foreach ($v in $Values) { if ($null -ne $key.$v) { $saved[$v] = [int]$key.$v } }
            if (-not $saved.Contains('Enabled')) { throw 'Memory Integrity has no Enabled value; nothing changed' }
            $saved | ConvertTo-Json | Out-File $StateFile -Encoding utf8
        }
        Set-ItemProperty $Hvci -Name Enabled -Value 0 -Type DWord
        Set-LaunchType 'off'
    } else {
        if (-not (Test-Path $StateFile)) { throw "no saved settings at $StateFile; nothing changed" }
        $saved = Get-Content $StateFile -Raw | ConvertFrom-Json
        foreach ($v in $Values) {
            if ($null -ne $saved.$v) { Set-ItemProperty $Hvci -Name $v -Value $saved.$v -Type DWord }
        }
        Set-LaunchType 'auto'
        $now = Get-ItemProperty $Hvci
        if ($now.Enabled -ne $saved.Enabled) { throw 'Memory Integrity value was not restored' }
        Remove-Item $StateFile
    }
    "hypervisorlaunchtype after : $(Get-LaunchType)" | Out-File $LogFile -Append -Encoding utf8
    exit 0
} catch {
    "ERROR: $_" | Out-File $LogFile -Append -Encoding utf8
    exit 1
}
