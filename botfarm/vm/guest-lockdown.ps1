<#
Runs inside a bot VM, elevated, after the Windows install (new-bot-vm.ps1
copies it in as a plain file). Re-runnable.

User rule: AVOID Windows updates in the bot VMs (their pop-ups interfere with
the bots; an occasional update is fine). Several layers, best effort, because Windows re-enables single ones:
  - Windows Update policy: no automatic updates, no access to WU servers
    (points WU at a dead local server), no UI to start updates
  - update services disabled: wuauserv, UsoSvc, WaaSMedicSvc (the one that
    "repairs" the others), DoSvc (delivery optimisation)
  - update scheduled tasks disabled
  - Microsoft Store auto-updates off
Also: never sleep / turn the screen off / lock (bots need a live screen),
auto-logon so scheduled bot tasks have a desktop, UAC consent prompt off.
#>
param([string]$User = '', [string]$PasswordFile = '')

$Password = ''
if ($PasswordFile -and (Test-Path $PasswordFile)) {
    $Password = [IO.File]::ReadAllText($PasswordFile).Trim()
    Remove-Item $PasswordFile -Force
}

$ErrorActionPreference = 'Continue'
function Set-Reg($path, $name, $value, $type = 'DWord') {
    if (-not (Test-Path $path)) { New-Item -Path $path -Force | Out-Null }
    New-ItemProperty -Path $path -Name $name -Value $value -PropertyType $type -Force | Out-Null
}

# --- Windows Update: off for good ---
$wu = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate'
Set-Reg "$wu\AU" NoAutoUpdate 1
Set-Reg "$wu\AU" AUOptions 1
Set-Reg "$wu\AU" UseWUServer 1
Set-Reg $wu WUServer 'http://127.0.0.1:1' 'String'
Set-Reg $wu WUStatusServer 'http://127.0.0.1:1' 'String'
Set-Reg $wu UpdateServiceUrlAlternate 'http://127.0.0.1:1' 'String'
Set-Reg $wu DoNotConnectToWindowsUpdateInternetLocations 1
Set-Reg $wu SetDisableUXWUAccess 1
Set-Reg $wu ExcludeWUDriversInQualityUpdate 1
# no update / restart pop-ups over the bots, no restarts while logged on
Set-Reg $wu SetUpdateNotificationLevel 1
Set-Reg $wu UpdateNotificationLevel 2
Set-Reg "$wu\AU" NoAutoRebootWithLoggedOnUsers 1
Set-Reg 'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\PushNotifications' ToastEnabled 0
Set-Reg 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\DeliveryOptimization' DODownloadMode 99
Set-Reg 'HKLM:\SOFTWARE\Policies\Microsoft\WindowsStore' AutoDownload 2
Set-Reg 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\CloudContent' DisableWindowsConsumerFeatures 1

foreach ($svc in 'wuauserv', 'UsoSvc', 'WaaSMedicSvc', 'DoSvc') {
    Stop-Service $svc -Force -ErrorAction SilentlyContinue
    # registry, because sc.exe is refused for WaaSMedicSvc
    Set-Reg "HKLM:\SYSTEM\CurrentControlSet\Services\$svc" Start 4
}
foreach ($folder in '\Microsoft\Windows\WindowsUpdate\', '\Microsoft\Windows\UpdateOrchestrator\',
                    '\Microsoft\Windows\WaaSMedic\') {
    Get-ScheduledTask -TaskPath $folder -ErrorAction SilentlyContinue |
        Disable-ScheduledTask -ErrorAction SilentlyContinue | Out-Null
}

# --- firewall: the update services can't reach the network even if Windows
# re-enables them (wuauserv was switched back to demand start, 2026-11-23) ---
Remove-NetFirewallRule -Group 'WizardEyes no updates' -ErrorAction SilentlyContinue
foreach ($svc in 'wuauserv', 'UsoSvc', 'WaaSMedicSvc', 'DoSvc') {
    New-NetFirewallRule -DisplayName "WizardEyes block $svc" -Group 'WizardEyes no updates' -Direction Outbound `
        -Action Block -Service $svc -Profile Any | Out-Null
}
foreach ($exe in 'MoUsoCoreWorker.exe', 'UsoClient.exe', 'wuauclt.exe') {
    New-NetFirewallRule -DisplayName "WizardEyes block $exe" -Group 'WizardEyes no updates' -Direction Outbound `
        -Action Block -Program "%SystemRoot%\System32\$exe" -Profile Any | Out-Null
}
# --- always awake, never locked ---
powercfg /change monitor-timeout-ac 0
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /hibernate off
Set-Reg 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\Personalization' NoLockScreen 1
Set-Reg 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System' InactivityTimeoutSecs 0
Set-Reg 'HKCU:\Control Panel\Desktop' ScreenSaveActive '0' 'String'

# --- admin work without a prompt (user, 2026-11-27): UAC stays on, its consent prompt off, so the host's
# setup can run admin steps with RunAs (Invoke-GuestElevated in lib\BotFarm.psm1) ---
Set-Reg 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System' ConsentPromptBehaviorAdmin 0

# --- auto-logon (bots run in the logged-on desktop) ---
if ($User -and $Password) {
    $wl = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon'
    Set-Reg $wl AutoAdminLogon '1' 'String'
    Set-Reg $wl DefaultUserName $User 'String'
    Set-Reg $wl DefaultPassword $Password 'String'
    Set-Reg $wl DefaultDomainName $env:COMPUTERNAME 'String'
    Remove-ItemProperty $wl -Name AutoLogonCount -ErrorAction SilentlyContinue
}

"guest-lockdown done $(Get-Date -Format s)" | Out-File C:\guest-lockdown.log -Append
