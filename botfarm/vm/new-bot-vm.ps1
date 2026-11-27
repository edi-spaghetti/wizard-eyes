<#
.SYNOPSIS
    Build a bot VM (VirtualBox, Windows 11) from nothing, or finish /
    refresh one that already exists.

.DESCRIPTION
    Runs on the host. Safe to re-run: each stage checks whether it's done.

      1. Checks       VirtualBox >= 7.1 (7.0 can't boot current Windows 11
                      media under UEFI), the Windows ISO, free disk space
      2. Login        VM Windows login: user 'bot', random password kept in
                      the host's Credential Manager (wizard-eyes/vm-user);
                      shared by all bot VMs, never printed
      3. Create       EFI + secure boot + TPM 2.0, VBoxSVGA (no 3D), NAT, USB tablet
      4. Install      unattended Windows install with the network cable
                      unplugged, then one reboot (starts Guest Additions)
      5. Wait         until Windows has logged in (Guest Additions report it)
      6. Lockdown     guest-lockdown.ps1 copied in and run elevated (Windows
                      Update off for good, never sleep/lock, auto-logon, UAC
                      consent prompt off), then checked; the network stays
                      unplugged if anything is missing
      6b. Name        new VMs (and -Fixup): consent prompt off if it isn't
                      (one Alt+Y from the host), unique machine name (restart)
      7. Network      cable plugged in, screen set to -Width x -Height
      8. Provision    the engine repo and the farm's repo (git bundles from
                      this PC: the farm's repo may be private) + read-only R2
                      config copied in, then guest-provision.ps1 run inside

    Then follow the checklist it prints (or botfarm\setup.ps1 walks you
    through it): Jagex Launcher install + login, RuneLite, profile.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File botfarm\vm\new-bot-vm.ps1 -Name wizard-eyes-vm3

.EXAMPLE
    # refresh repos / re-run provisioning on an existing, running VM
    powershell -ExecutionPolicy Bypass -File botfarm\vm\new-bot-vm.ps1 -Name wizard-eyes-vm1
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Name,
    # the farm whose repo goes in too (default: this machine's default farm)
    [string]$Farm,
    [string]$Iso = (Join-Path $env:USERPROFILE 'VirtualBox VMs\iso\Win11_26H2_EnglishUK_x64.iso'),
    # Windows edition index in the ISO (6 = Pro: honours the update policies)
    [int]$ImageIndex = 6,
    [int]$MemoryMB = 0,
    [int]$Cpus = 0,
    [string]$Paravirt = 'default',
    [int]$DiskGB = 0,
    [int]$Width = 0,
    [int]$Height = 0,
    [int]$InstallTimeoutMin = 120,
    # Stop after the verify stage (no network, no provisioning).
    [switch]$NoProvision,
    # Make a linked clone of this VM instead of installing Windows (small
    # disk; the source must be powered off and must be kept).
    [string]$CloneFrom,
    # On a VM that already existed: also turn its consent prompt off and give
    # it its machine name (may restart it: not while a bot runs in it)
    [switch]$Fixup
)

$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot '..\lib\BotFarm.psm1') -Force
$fm = Get-Farm $Farm
$Engine = $fm.Engine
# farm.json's vm_defaults unless given here
$d = $fm.Def.vm_defaults
if (-not $MemoryMB) { $MemoryMB = if ($d.memory_mb) { $d.memory_mb } else { 6144 } }
if (-not $Cpus) { $Cpus = if ($d.cpus) { $d.cpus } else { 6 } }
if (-not $DiskGB) { $DiskGB = if ($d.disk_gb) { $d.disk_gb } else { 64 } }
if (-not $Width) { $Width = if ($d.width) { $d.width } else { 1920 } }
if (-not $Height) { $Height = if ($d.height) { $d.height } else { 1200 } }

function Write-Step([string]$t) { Write-Host "`n== $t" -ForegroundColor Cyan }
function Write-Ok([string]$t) { Write-Host "   OK   $t" -ForegroundColor Green }
function Write-Info([string]$t) { Write-Host "        $t" }
function Guest([string]$Code, [int]$TimeoutSec = 600) { (Invoke-Guest $Name $Code $TimeoutSec) -join "`n" }

# ---------------------------------------------------------------------------
Write-Step '1. Checks'
if (-not (Get-VBoxManage)) { throw 'VirtualBox is not installed' }
$ver = (Invoke-VBox @('--version')).Trim()
if ([version]($ver -replace 'r\d+$', '') -lt [version]'7.1') { throw "VirtualBox $ver is too old (need 7.1+)" }
Write-Ok "VirtualBox $ver"
$vm = Get-VmInfo $Name
$exists = $vm.ContainsKey('VMState')
$built = -not $exists
if (-not $exists -and -not $CloneFrom) {
    # Windows setup hangs under Hyper-V (VM1, 2026-11-23, four installs);
    # switching it off is the user's decision, never done here
    if ((Get-CimInstance Win32_ComputerSystem).HypervisorPresent) {
        throw 'The Windows hypervisor is on: installs hang. Run botfarm\vm\hypervisor.ps1 off and reboot first.'
    }
    Write-Ok 'Windows hypervisor is off'
    if (-not (Test-Path $Iso)) { throw "Windows ISO not found: $Iso" }
    $free = (Get-PSDrive C).Free / 1GB
    if ($free -lt 30) { throw ("Only {0:N0} GB free on C: (a VM needs ~25-30 GB)" -f $free) }
    Write-Ok ("ISO found, {0:N0} GB free" -f $free)
}

Write-Step '2. VM login'
if (-not (Get-VmPassword)) {
    [void](New-VmPassword)
    Write-Ok "new password stored in Credential Manager ($VmCredential)"
} else { Write-Ok "using stored password ($VmCredential)" }

Use-VmPassword {
    if (-not $exists -and $CloneFrom) {
        Write-Step "3. Linked clone of $CloneFrom"
        $srcState = (Get-VmInfo $CloneFrom)['VMState']
        if ($srcState -ne 'poweroff') { throw "$CloneFrom is '$srcState'; shut it down first" }
        $snap = "clone-base-$Name"
        Invoke-VBox @('snapshot', $CloneFrom, 'take', $snap) | Out-Null
        # clonevm gives the clone its own MAC address and disk ids
        Invoke-VBox @('clonevm', $CloneFrom, '--snapshot', $snap, '--options', 'link', '--name', $Name, '--register') -TimeoutSec 600 | Out-Null
        $vm = Get-VmInfo $Name
        $exists = $true
        Write-Ok "$Name cloned"
    }
    if (-not $exists) {
        Write-Step '3. Create VM'
        Invoke-VBox @('createvm', '--name', $Name, '--ostype', 'Windows11_64', '--register') | Out-Null
        Invoke-VBox @('modifyvm', $Name, '--memory', "$MemoryMB", '--cpus', "$Cpus", '--paravirt-provider', $Paravirt, '--firmware', 'efi',
            '--tpm-type', '2.0', '--graphicscontroller', 'vboxsvga', '--accelerate3d', 'off', '--vram', '256',   # the Windows driver only knows vboxsvga
             '--nic1', 'nat',
            # guests reach the host's 127.0.0.1 (coordination service) at 10.0.2.2
            '--nat-localhostreachable1', 'on',
            '--cable-connected1', 'off', '--audio-enabled', 'off', '--clipboard-mode', 'bidirectional',
            '--mouse', 'usbtablet') | Out-Null
        foreach ($sub in 'inituefivarstore', 'enrollmssignatures', 'enrollorclpk') {
            Invoke-VBox @('modifynvram', $Name, $sub) | Out-Null
        }
        $vm = Get-VmInfo $Name
        $dir = Split-Path $vm['CfgFile']
        $disk = Join-Path $dir "$Name.vdi"
        Invoke-VBox @('createmedium', 'disk', '--filename', $disk, '--size', "$($DiskGB * 1024)", '--format', 'VDI') | Out-Null
        # Under Hyper-V (NEM) Windows setup reset the SATA disk 12 min in and
        # hung, twice, with and without the host cache; NVMe isn't seen by
        # setup at all (VM1, 2026-11-23)
        Invoke-VBox @('storagectl', $Name, '--name', 'SATA', '--add', 'sata', '--controller', 'IntelAhci', '--portcount', '3',
            '--hostiocache', 'on') | Out-Null
        Invoke-VBox @('storageattach', $Name, '--storagectl', 'SATA', '--port', '0', '--device', '0', '--type', 'hdd', '--medium', $disk) | Out-Null
        Write-Ok "$Name created ($MemoryMB MB, $Cpus CPUs, $DiskGB GB disk, network unplugged)"

        Write-Step '4. Unattended Windows install'
        # Additions only start after a reboot; the lockdown is copied in as a
        # plain file afterwards (stage 6: Defender blocked an encoded one).
        # The command runs elevated at the first logon (FirstLogonCommands):
        # the UAC consent prompt goes off there, so later admin steps are silent
        $post = 'reg add HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System /v ConsentPromptBehaviorAdmin /t REG_DWORD /d 0 /f & shutdown /r /t 30'
        $short = ($Name -replace '[^A-Za-z0-9-]', '-'); if ($short.Length -gt 15) { $short = $short.Substring($short.Length - 15) }
        Invoke-VBox @('unattended', 'install', $Name, "--iso=$Iso", "--image-index=$ImageIndex",
            "--user=$GuestUser", "--user-password-file=$(Get-VmPasswordFile)", '--full-user-name=bot',
            "--hostname=$short.local", '--locale=en_GB', '--country=GB', '--time-zone=GMT Standard Time',
            '--install-additions', "--post-install-command=$post") -TimeoutSec 300 | Out-Null
        # (installs don't need the fast cores; bots do, so they start through Start-FarmVm)
        Invoke-VBox @('startvm', $Name, '--type', 'headless') -TimeoutSec 120 | Out-Null
        Write-Ok 'install started (headless)'
    } else {
        Write-Ok "$Name exists ($($vm['VMState'])); skipping create / install"
        if (-not (Test-VmRunning $Name)) {
            Start-FarmVm $Name
            Write-Ok 'started (headless, fast cores)'
        }
    }

    Write-Step '5. Wait for Windows to log in'
    $started = Get-Date
    $deadline = $started.AddMinutes($InstallTimeoutMin)
    while (-not (Test-VmLoggedIn $Name)) {
        $state = (Get-VmInfo $Name)['VMState']
        if ($state -in 'poweroff', 'aborted') {
            # (right after startvm the state can still read 'poweroff' for a moment)
            if (((Get-Date) - $started).TotalSeconds -gt 45) { throw "VM stopped ($state) before logging in" }
            Start-Sleep 5; continue
        }
        if ((Get-Date) -gt $deadline) { throw "not logged in after $InstallTimeoutMin min" }
        Write-Info ("{0:HH:mm} waiting..." -f (Get-Date))
        Start-Sleep 60
    }
    Write-Ok 'logged in'
    # install media no longer needed; the generated answer file holds the
    # password in plain text, so remove it
    foreach ($port in 1, 2) {
        Invoke-VBox @('storageattach', $Name, '--storagectl', 'SATA', '--port', "$port", '--device', '0',
            '--type', 'dvddrive', '--medium', 'emptydrive', '--forceunmount') -AllowFail | Out-Null
    }
    Get-ChildItem (Split-Path (Get-VmInfo $Name)['CfgFile']) -Filter 'Unattended-*' | Remove-Item -Force
    Write-Ok 'install media ejected, generated answer files deleted'

    Write-Step '6. Verify the no-updates lockdown'
    $check = @'
$bad = @()
# services are best effort only: Windows re-enables wuauserv once online
$svc = foreach ($s in 'wuauserv','UsoSvc','WaaSMedicSvc','DoSvc') {
  if ((Get-Service $s -ErrorAction SilentlyContinue).Status -eq 'Running') { $s } }
$au = Get-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU' -ErrorAction SilentlyContinue
if ($au.NoAutoUpdate -ne 1) { $bad += 'policy NoAutoUpdate' }
if ($au.UseWUServer -ne 1) { $bad += 'policy UseWUServer' }
$wu = Get-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate' -ErrorAction SilentlyContinue
if ($wu.DoNotConnectToWindowsUpdateInternetLocations -ne 1) { $bad += 'policy DoNotConnect' }
if ((Get-ItemProperty 'HKLM:\SOFTWARE\Policies\Microsoft\WindowsStore' -ErrorAction SilentlyContinue).AutoDownload -ne 2) { $bad += 'store auto-download' }
if ($wu.UpdateNotificationLevel -ne 2) { $bad += 'policy update notifications' }
if ($au.NoAutoRebootWithLoggedOnUsers -ne 1) { $bad += 'policy auto reboot' }
$fw = @(Get-NetFirewallRule -Group 'WizardEyes no updates' -ErrorAction SilentlyContinue | Where-Object { $_.Enabled -eq 'True' -and $_.Action -eq 'Block' }).Count
if ($fw -lt 7) { $bad += "policy firewall rules $fw/7" }
if ($bad) { 'LOCKDOWN-MISSING: ' + ($bad -join '; ') } else { 'LOCKDOWN-OK' }
if ($svc) { 'running services: ' + ($svc -join ', ') }
'@
    $res = Guest $check 120
    if ($res -notmatch 'LOCKDOWN-OK') {
        # (only unplugged while it isn't locked down: a working VM keeps its network)
        Invoke-VBox @('controlvm', $Name, 'setlinkstate1', 'off') | Out-Null
        Write-Info 'not locked down yet; running guest-lockdown.ps1 (plain file)'
        Guest "New-Item -ItemType Directory -Force '$GuestDrop' | Out-Null" 60 | Out-Null
        Copy-ToGuest $Name (Join-Path $PSScriptRoot 'guest-lockdown.ps1') "$GuestDrop\guest-lockdown.ps1"
        Copy-ToGuest $Name (Get-VmPasswordFile) "$GuestDrop\pw.txt"
        # guestcontrol isn't elevated: RunAs from the VM's desktop, silent once
        # the consent prompt is off (else the host answers it with Alt+Y)
        Invoke-GuestElevated $Name -Script "$GuestDrop\guest-lockdown.ps1" -ArgText "-User $GuestUser -PasswordFile $GuestDrop\pw.txt" -TimeoutSec 300 | Out-Null
        $res = Guest $check 120
        if ($res -notmatch 'LOCKDOWN-OK') { throw "Update lockdown not applied, network left unplugged: $($res.Trim())" }
        # disabled services still got started until a reboot (VM1, 2026-11-23)
        Write-Info 'settings applied; rebooting'
        Guest 'shutdown /r /t 5' 60 | Out-Null
        Start-Sleep 150
        $res = Guest $check 120
    }
    if ($res -notmatch 'LOCKDOWN-OK') {
        throw "Update lockdown incomplete, network left unplugged: $($res.Trim())"
    }
    Write-Ok 'automatic updates, notifications and reboots off; network can go on'
    if ($res -match 'running services: ([^\r\n]+)') { Write-Info "note: running: $($Matches[1])" }

    if ($built -or $Fixup) {
        Write-Step '6b. Admin without a prompt + machine name'
        # (VMs built before 2026-11-27 have the prompt on: one Alt+Y from here turns it off)
        Write-Ok "consent prompt off: $(Enable-GuestSilentAdmin $Name)"
        $mname = Get-MachineName $Name
        $r = Set-GuestMachineName $Name $mname
        if ($r -eq 'restarting') {
            # two restarts: the task renames at the first startup, the name holds after the second
            $end = (Get-Date).AddMinutes(10); $now = ''
            while ($now -ne $mname.ToUpper()) {
                if ((Get-Date) -gt $end) { throw "machine name is still '$now', not $mname, 10 min after the restart" }
                Start-Sleep 20
                if (-not (Test-VmLoggedIn $Name)) { continue }
                $now = try { ((Invoke-Guest $Name '$env:COMPUTERNAME' 60 -AllowFail) -join '').Trim() } catch { '' }
            }
            Start-Sleep 20      # (the desktop settles)
        }
        Write-Ok "machine name $mname ($r)"
    }
    if ($NoProvision) { Write-Info 'stopping here (-NoProvision)'; return }

    Write-Step '7. Network + screen'
    Invoke-VBox @('controlvm', $Name, 'setlinkstate1', 'on') | Out-Null
    Invoke-VBox @('controlvm', $Name, 'setvideomodehint', "$Width", "$Height", '32') -AllowFail | Out-Null
    Write-Ok "network on, screen ${Width}x$Height"

    Write-Step '8. Provision'
    $stage = Join-Path $env:TEMP "botfarm-vm-$PID"
    New-Item -ItemType Directory -Force $stage | Out-Null
    $remotes = [ordered]@{}
    # -q and no 2>&1: PS 5.1 turns redirected native stderr into a terminating error
    foreach ($repo in @($Engine, $fm.Repo)) {
        $leaf = Split-Path $repo -Leaf
        git -C $repo bundle create -q (Join-Path $stage "$leaf.bundle") main
        if ($LASTEXITCODE -ne 0) { throw "git bundle of $leaf failed" }
        $remotes[$leaf] = (git -C $repo remote get-url origin)
    }
    Write-Utf8 (Join-Path $stage 'remotes.json') ($remotes | ConvertTo-Json)
    # read-only R2 access only; never the publish keys
    $cfg = Get-Content (Join-Path $Engine 'config\assets.local.json') -Raw | ConvertFrom-Json
    $ro = [ordered]@{ backend = $cfg.backend; local_dir = ''; r2 = $cfg.r2 }
    Write-Utf8 (Join-Path $stage 'assets.local.json') ($ro | ConvertTo-Json -Depth 5)
    Copy-Item (Join-Path $PSScriptRoot 'guest-provision.ps1') $stage
    Guest "New-Item -ItemType Directory -Force '$GuestDrop' | Out-Null" 60 | Out-Null
    foreach ($f in Get-ChildItem $stage) { Copy-ToGuest $Name $f.FullName "$GuestDrop\$($f.Name)" }
    Remove-Item $stage -Recurse -Force
    Write-Ok "repos (main) + R2 read config copied to $GuestDrop"
    $out = Guest "& '$GuestDrop\guest-provision.ps1' -Drop '$GuestDrop'" 3600
    $out.Trim() -split "`r?`n" | ForEach-Object { Write-Info $_ }
}
