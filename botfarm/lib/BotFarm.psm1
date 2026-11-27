<#
Shared helpers for the botfarm scripts (Windows PowerShell 5.1).

A farm is a folder (usually in the bots' own repo) holding:
  farm.json         what the farm runs: roles (one bot each, in its own VM),
                    services on the host, the page, the icon; committed
  farm.local.json   this machine only, git-ignored: account names, ...
The engine's scripts take -Farm <folder>; without it they use
$env:WIZARD_EYES_FARM (set for a farm's services), then the default farm
registered on this machine (%LOCALAPPDATA%\wizard-eyes\botfarm.json).

    Import-Module "$PSScriptRoot\lib\BotFarm.psm1" -Force
    $fm = Get-Farm $Farm     # (not $farm: names ignore case, it would overwrite $Farm)
#>
$ErrorActionPreference = 'Stop'

$script:Engine = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$script:MachineFile = Join-Path $env:LOCALAPPDATA 'wizard-eyes\botfarm.json'
$script:PwFile = $null

# inside every bot VM (vm\guest-provision.ps1 sets them up)
$GuestUser = 'bot'
$GuestDrop = 'C:\wizard-eyes-provision'
$GuestRoot = 'C:\Users\bot\wizard-eyes'
$VmCredential = 'wizard-eyes/vm-user'

function Get-EngineRoot { $script:Engine }

# ---------------------------------------------------------------------------
# farms
# ---------------------------------------------------------------------------

function Get-MachineSettings {
    if (Test-Path $script:MachineFile) { Get-Content $script:MachineFile -Raw | ConvertFrom-Json }
    else { [pscustomobject]@{ default = $null } }
}

function Set-DefaultFarm([string]$Folder) {
    $s = Get-MachineSettings
    $s.default = (Resolve-Path $Folder).Path
    New-Item -ItemType Directory -Force (Split-Path $script:MachineFile) | Out-Null
    Write-Utf8 $script:MachineFile ($s | ConvertTo-Json)
}

function Resolve-FarmFolder([string]$Farm) {
    if (-not $Farm) { $Farm = $env:WIZARD_EYES_FARM }
    if (-not $Farm) { $Farm = (Get-MachineSettings).default }
    if (-not $Farm) { throw 'No farm given (-Farm), and no default farm on this machine (botfarm\setup.ps1 sets one)' }
    if (-not (Test-Path (Join-Path $Farm 'farm.json'))) { throw "No farm.json in $Farm" }
    (Resolve-Path $Farm).Path
}

function Get-RepoRoot([string]$Folder) {
    $d = Get-Item $Folder
    while ($d) {
        if (Test-Path (Join-Path $d.FullName '.git')) { return $d.FullName }
        $d = $d.Parent
    }
    throw "$Folder is not inside a git repo"
}

function Get-Farm([string]$Farm) {
    $dir = Resolve-FarmFolder $Farm
    $def = Get-Content (Join-Path $dir 'farm.json') -Raw | ConvertFrom-Json
    $localPath = Join-Path $dir 'farm.local.json'
    $local = if (Test-Path $localPath) { Get-Content $localPath -Raw | ConvertFrom-Json } else { [pscustomobject]@{} }
    $repo = Get-RepoRoot $dir
    $f = [pscustomobject]@{
        Name = $def.name
        Dir = $dir
        Def = $def
        Local = $local
        LocalPath = $localPath
        Engine = $script:Engine
        Repo = $repo
        RepoName = Split-Path $repo -Leaf
        Python = Join-Path $script:Engine 'venv\Scripts\python.exe'
        StateDir = Join-Path $script:Engine $def.state_dir
        LogDir = Join-Path $script:Engine 'logs'
        Roles = [ordered]@{}
    }
    foreach ($p in $def.roles.PSObject.Properties) {
        $r = $p.Value
        $f.Roles[$p.Name] = [pscustomobject]@{
            Name = $p.Name
            Vm = $r.vm
            Task = if ($r.task) { $r.task } else { "WizardEyes-$($def.name)-$($p.Name)" }
            Script = $r.script
            Vpn = Get-RoleVpnSetting $def $r
            Args = @($r.args)
            Log = "$($p.Name).log"
            Account = Get-Account $f $p.Name -Quiet
        }
    }
    $f
}

function Get-Account($Farm, [string]$Role, [switch]$Quiet) {
    $a = $null
    if ($Farm.Local.accounts) { $a = $Farm.Local.accounts.$Role }
    if (-not $a -and -not $Quiet) { throw "No account for the $Role in $($Farm.LocalPath) (accounts.$Role)" }
    $a
}

# {engine} {farm} {repo} {control} {account:ROLE} in farm.json strings
function Expand-FarmText($Farm, [string]$Text) {
    $Text = $Text.Replace('{engine}', $Farm.Engine).Replace('{farm}', $Farm.Dir).Replace('{repo}', $Farm.Repo)
    $Text = $Text.Replace('{control}', (Join-Path $Farm.Engine 'botfarm\bot-control.ps1'))
    [regex]::Replace($Text, '\{account:([\w-]+)\}', { param($m) Get-Account $Farm $m.Groups[1].Value })
}

# the farm's coordination service as a VM sees it (NAT: the host's 127.0.0.1 is 10.0.2.2)
function Get-GuestCoordination($Farm) {
    $url = if ($Farm.Def.coordination) { $Farm.Def.coordination } else { 'http://127.0.0.1:8765' }
    $url -replace '//(127\.0\.0\.1|localhost)([:/]|$)', '//10.0.2.2$2'
}

# the batch file a role's scheduled task runs in its VM (sent at every start)
function Get-RoleCmd($Farm, $Role) {
    $engine = "$GuestRoot\wizard-eyes"
    $repo = "$GuestRoot\$($Farm.RepoName)"
    $argText = ($Role.Args | ForEach-Object {
        $a = Expand-FarmText $Farm $_
        if ($a -match '[\s&|<>^]') { '"' + $a + '"' } else { $a }
    }) -join ' '
    $log = "$GuestDrop\$($Role.Log)"
    @(
        '@echo off'
        "set PYTHONPATH=$engine\python;$repo"
        'set PYTHONUNBUFFERED=1'
        'rem the engine sends a heartbeat as this role (wizard_eyes.heartbeat)'
        "set WIZARD_EYES_ROLE=$($Role.Name)"
        "set WIZARD_EYES_COORDINATOR=$(Get-GuestCoordination $Farm)"
        'rem bots only run while Num Lock (or Caps Lock) is on'
        'powershell -NoProfile -Command "if(-not [console]::NumberLock){(New-Object -ComObject WScript.Shell).SendKeys(''{NUMLOCK}'')}"'
        "cd /d $repo"
        "$engine\venv\Scripts\python.exe -u $($Role.Script) $argText > $log 2>&1"
        "echo exit %errorlevel% >> $log"
    ) -join "`r`n"
}

function Get-RoleVbs($Role) {
    "CreateObject(""WScript.Shell"").Run ""cmd /c $GuestDrop\$($Role.Name).cmd"", 0, True"
}

# ---------------------------------------------------------------------------
# small things
# ---------------------------------------------------------------------------

function Write-Utf8([string]$Path, [string]$Text) {
    [IO.File]::WriteAllText($Path, $Text, (New-Object Text.UTF8Encoding $false))
}

function Test-Url([string]$Url) {
    try { [void](Invoke-RestMethod $Url -TimeoutSec 5); $true } catch { $false }
}

# ---------------------------------------------------------------------------
# VirtualBox
# ---------------------------------------------------------------------------

function Get-VBoxManage {
    $dir = $env:VBOX_MSI_INSTALL_PATH
    if (-not $dir) { $dir = $env:VBOX_INSTALL_PATH }
    if (-not $dir) { $dir = 'C:\Program Files\Oracle\VirtualBox\' }
    $exe = Join-Path $dir 'VBoxManage.exe'
    if (Test-Path $exe) { $exe } else { $null }
}

# VBoxManage with a timeout (its service has hung before); returns stdout
function Invoke-VBox {
    param([string[]]$VBoxArgs, [int]$TimeoutSec = 120, [switch]$AllowFail)
    $vbox = Get-VBoxManage
    if (-not $vbox) { throw 'VirtualBox is not installed' }
    $out = Join-Path $env:TEMP "vbox-$PID-out.txt"; $err = Join-Path $env:TEMP "vbox-$PID-err.txt"
    $quoted = $VBoxArgs | ForEach-Object { if ($_ -match '[\s"]' -or $_ -eq '') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }
    $p = Start-Process $vbox -ArgumentList $quoted -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $p.Handle   # PS 5.1: without this ExitCode stays empty
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        $p.Kill(); throw "VBoxManage $($VBoxArgs[0]) timed out after ${TimeoutSec}s"
    }
    $text = Get-Content $out -Raw -ErrorAction SilentlyContinue
    $script:VBoxErr = Get-Content $err -Raw -ErrorAction SilentlyContinue
    Remove-Item $out, $err -ErrorAction SilentlyContinue
    if ($p.ExitCode -ne 0 -and -not $AllowFail) { throw "VBoxManage $($VBoxArgs[0]) failed: $text $($script:VBoxErr)" }
    "$text"
}

function Get-VmInfo([string]$Vm) {
    $info = @{}
    foreach ($line in ((Invoke-VBox @('showvminfo', $Vm, '--machinereadable') -AllowFail) -split "`r?`n")) {
        if ($line -match '^"?([^"=]+)"?="?(.*?)"?$') { $info[$Matches[1]] = $Matches[2] }
    }
    $info
}

function Test-VmRunning([string]$Vm) {
    (Invoke-VBox @('list', 'runningvms') -AllowFail) -match [regex]::Escape("`"$Vm`"")
}

function Test-VmLoggedIn([string]$Vm) {
    (Invoke-VBox @('guestproperty', 'get', $Vm, '/VirtualBox/GuestInfo/OS/LoggedInUsers') -AllowFail) -match 'Value: [1-9]'
}

# Logical CPUs of the fastest cores (the P-cores of a hybrid CPU; all CPUs
# otherwise) in processor group 0, as a hex mask for "start /affinity".
# VMs pinned this way ran ~6x faster than unpinned on an i7-12700H.
function Get-FastCpuMask {
    if (-not ('BotFarm.CpuSets' -as [type])) {
        Add-Type -Namespace BotFarm -Name CpuSets -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError=true)]
public static extern bool GetSystemCpuSetInformation(IntPtr info, uint length, out uint returned, IntPtr process, uint flags);
'@
    }
    [uint32]$len = 0
    [void][BotFarm.CpuSets]::GetSystemCpuSetInformation([IntPtr]::Zero, 0, [ref]$len, [IntPtr]::Zero, 0)
    $buf = [Runtime.InteropServices.Marshal]::AllocHGlobal([int]$len)
    try {
        if (-not [BotFarm.CpuSets]::GetSystemCpuSetInformation($buf, $len, [ref]$len, [IntPtr]::Zero, 0)) { throw 'GetSystemCpuSetInformation failed' }
        $sets = @(); $off = 0
        while ($off -lt $len) {
            $size = [Runtime.InteropServices.Marshal]::ReadInt32($buf, $off)
            # SYSTEM_CPU_SET_INFORMATION: Size, Type, Id, Group(2), LogicalProcessorIndex, CoreIndex, LLC, Numa, EfficiencyClass
            $sets += [pscustomobject]@{
                Group = [Runtime.InteropServices.Marshal]::ReadInt16($buf, $off + 12)
                Index = [Runtime.InteropServices.Marshal]::ReadByte($buf, $off + 14)
                Class = [Runtime.InteropServices.Marshal]::ReadByte($buf, $off + 18)
            }
            $off += $size
        }
    } finally { [Runtime.InteropServices.Marshal]::FreeHGlobal($buf) }
    $g0 = @($sets | Where-Object Group -eq 0)
    $best = ($g0 | Measure-Object Class -Maximum).Maximum
    [uint64]$mask = 0
    foreach ($s in $g0 | Where-Object Class -eq $best) { $mask = $mask -bor ([uint64]1 -shl $s.Index) }
    '{0:X}' -f $mask
}

# Start a VM pinned to the fast cores (VirtualBox won't change it later).
# -Window also opens its window (closing that window can offer to stop the VM: detach).
function Start-FarmVm([string]$Vm, [switch]$Window) {
    $dir = Split-Path (Get-VBoxManage)
    $mask = Get-FastCpuMask
    Start-Process cmd -ArgumentList '/c', "start `"`" /min /affinity $mask `"$dir\VBoxHeadless.exe`" --startvm $Vm" -WindowStyle Hidden
    if ($Window) { Start-Sleep 8; Show-VmWindow $Vm }
}

function Show-VmWindow([string]$Vm) {
    $open = Get-Process VirtualBoxVM -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "$Vm*" }
    # the guest keeps its own screen size and the window shows all of it, scaled: a window resized by hand used to
    # shrink the guest, and the maximised RuneLite then stuck out of the screen (user, 2026-11-27)
    Invoke-VBox @('setextradata', $Vm, 'GUI/AutoresizeGuest', 'false') -AllowFail | Out-Null
    Invoke-VBox @('setextradata', $Vm, 'GUI/Scale', 'true') -AllowFail | Out-Null
    if (-not $open) { Start-Process (Get-VBoxManage) -ArgumentList 'startvm', $Vm, '--type', 'separate' -WindowStyle Hidden }
}

function Wait-VmLoggedIn([string]$Vm, [int]$Minutes = 6) {
    $end = (Get-Date).AddMinutes($Minutes)
    while (-not (Test-VmLoggedIn $Vm)) {
        if ((Get-Date) -gt $end) { throw "$Vm did not get to its desktop" }
        Start-Sleep 5
    }
}

# ---------------------------------------------------------------------------
# inside the VMs (guest control as the bot user)
# ---------------------------------------------------------------------------

Add-Type -Namespace BotFarm -Name Cred -MemberDefinition @'
[DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
public static extern bool CredRead(string target, int type, int flags, out IntPtr cred);
[DllImport("advapi32.dll")] public static extern void CredFree(IntPtr cred);
'@ -ErrorAction SilentlyContinue

# A generic credential's secret from Credential Manager, or $null. Never print it.
function Get-StoredSecret([string]$Target) {
    $p = [IntPtr]::Zero
    if (-not [BotFarm.Cred]::CredRead($Target, 1, 0, [ref]$p)) { return $null }
    try {
        # CREDENTIALW: Flags, Type, TargetName*, Comment*, LastWritten(8), BlobSize, Blob*
        $off = 8 + [IntPtr]::Size * 2 + 8
        $size = [Runtime.InteropServices.Marshal]::ReadInt32($p, $off)
        $blob = [Runtime.InteropServices.Marshal]::ReadIntPtr($p, $off + 8)
        [Runtime.InteropServices.Marshal]::PtrToStringUni($blob, $size / 2)
    } finally { [BotFarm.Cred]::CredFree($p) }
}

function Get-VmPassword { Get-StoredSecret $VmCredential }

# The VMs' Windows login (user 'bot'): made once, shared by all bot VMs.
function New-VmPassword {
    $chars = 'abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789'.ToCharArray()
    $bytes = New-Object byte[] 20
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $pw = -join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] })
    cmdkey /generic:$VmCredential /user:$GuestUser /pass:$pw | Out-Null
    $pw
}

# Runs $Block with the VM password in a temp file (for --passwordfile), then deletes it.
function Use-VmPassword([scriptblock]$Block) {
    $pw = Get-VmPassword
    if (-not $pw) { throw "No VM password in Credential Manager ($VmCredential)" }
    $script:PwFile = Join-Path $env:TEMP "botfarm-$PID.pw"
    [IO.File]::WriteAllText($script:PwFile, $pw)
    try { & $Block } finally { Remove-Item $script:PwFile -ErrorAction SilentlyContinue; $script:PwFile = $null }
}

function Get-VmPasswordFile {
    if (-not $script:PwFile) { throw 'Use-VmPassword { ... } around guest calls' }
    $script:PwFile
}

# Run PowerShell code in the VM as the bot user; returns its output lines.
function Invoke-Guest([string]$Vm, [string]$Code, [int]$TimeoutSec = 90, [switch]$AllowFail) {
    $Code = '$ProgressPreference = "SilentlyContinue"; ' + $Code
    $enc = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Code))
    $out = Invoke-VBox @('guestcontrol', $Vm, 'run', '--username', $GuestUser, '--passwordfile', (Get-VmPasswordFile),
        '--exe', 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe', '--wait-stdout', '--wait-stderr',
        "--timeout=$($TimeoutSec * 1000)", '--', 'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-EncodedCommand', $enc) -TimeoutSec ($TimeoutSec + 30) -AllowFail:$AllowFail
    $out -split "`r?`n" | Where-Object { $_ -notmatch 'CLIXML|<Objs' }
}

# $Target: the full path of the file in the VM
function Copy-ToGuest([string]$Vm, [string]$Source, [string]$Target) {
    Invoke-VBox @('guestcontrol', $Vm, 'copyto', '--username', $GuestUser, '--passwordfile', (Get-VmPasswordFile),
        '--target-directory', $Target, $Source) -TimeoutSec 600 | Out-Null
}

function Copy-FromGuest([string]$Vm, [string]$Source, [string]$TargetDir) {
    Invoke-VBox @('guestcontrol', $Vm, 'copyfrom', '--username', $GuestUser, '--passwordfile', (Get-VmPasswordFile),
        '--target-directory', $TargetDir, $Source) -TimeoutSec 600 | Out-Null
}

# Text into a file in the VM (through a temp file on the host)
function Set-GuestText([string]$Vm, [string]$Target, [string]$Text) {
    $tmp = Join-Path $env:TEMP ("botfarm-$PID-" + (Split-Path $Target -Leaf))
    [IO.File]::WriteAllText($tmp, $Text, [Text.Encoding]::ASCII)
    try { Copy-ToGuest $Vm $tmp $Target } finally { Remove-Item $tmp -ErrorAction SilentlyContinue }
}

# --- admin work in a VM, with no one at it ----------------------------------------------------------
# Guest control runs as the bot user without admin rights (UAC). Admin work is started with RunAs from
# the VM's desktop session. With the consent prompt off (ConsentPromptBehaviorAdmin=0: new VMs get it in
# the install's post-install command, older ones from Enable-GuestSilentAdmin) that is silent. Otherwise
# the prompt appears on the VM's secure desktop and the host answers it: VirtualBox's keyboard is virtual
# hardware, so its Alt+Y reaches the secure desktop (a program in the VM, or on motherbear for its own
# prompts, can't). Tested on VM4, 2026-11-27: the prompt opens with "No" focused, Alt+Y picks "Yes".
$UacPolicy = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System'

# Alt+Y as key presses (PS/2 set 1 scancodes: Alt down, Y down, Y up, Alt up)
function Send-VmAltY([string]$Vm) {
    Invoke-VBox @('controlvm', $Vm, 'keyboardputscancode', '38', '15', '95', 'b8') | Out-Null
}

# Runs PowerShell elevated in the VM, as the bot user on its desktop, and waits. Give a script already in
# the VM (-Script, with -ArgText) or -Code. Answers a consent prompt with Alt+Y (at most twice); throws if
# a prompt stays, or nothing finishes in time. Returns the run's output. Call inside Use-VmPassword.
function Invoke-GuestElevated([string]$Vm, [string]$Script, [string]$ArgText = '', [string]$Code, [int]$TimeoutSec = 600) {
    $tag = "elevated-$PID-$(Get-Random)"
    $log = "$GuestDrop\$tag.log"; $done = "$GuestDrop\$tag.done"
    $work = if ($Script) { "& '$Script' $ArgText" } else { $Code }
    $inner = "`$ProgressPreference = 'SilentlyContinue'; try { & { $work } *>&1 | Out-File '$log' -Encoding utf8; 'ok' | Set-Content '$done' } " +
        "catch { ('FAILED: ' + `$_.Exception.Message) | Add-Content '$log'; 'failed' | Set-Content '$done' }"
    $encInner = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($inner))
    $launch = "Start-Process powershell.exe -Verb RunAs -WindowStyle Hidden -ArgumentList '-NoProfile -ExecutionPolicy Bypass -EncodedCommand $encInner'"
    $encLaunch = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($launch))
    Invoke-Guest $Vm "New-Item -ItemType Directory -Force '$GuestDrop' | Out-Null" 60 | Out-Null
    # "start", not "run": RunAs waits for the prompt's answer, and the answer comes from out here
    Invoke-VBox @('guestcontrol', $Vm, 'start', '--username', $GuestUser, '--passwordfile', (Get-VmPasswordFile),
        '--exe', 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe', '--',
        'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', $encLaunch) | Out-Null
    $pressed = 0; $seen = 0
    $end = (Get-Date).AddSeconds($TimeoutSec)
    while ($true) {
        Start-Sleep 2
        $st = (Invoke-Guest $Vm "'DONE=' + (Get-Content '$done' -EA 0) + ' CONSENT=' + @(Get-Process consent -EA 0).Count" 60 -AllowFail) -join ' '
        if ($st -match 'DONE=(ok|failed)') { $result = $Matches[1]; break }
        if ($st -match 'CONSENT=[1-9]') {
            # (give the prompt a moment to draw; it ignores keys before that)
            if (++$seen -ge 2) {
                if ($pressed -ge 2) { throw "an admin prompt in $Vm is still open after Alt+Y twice" }
                Send-VmAltY $Vm; $pressed++; $seen = 0
                Write-Host "        answered an admin prompt in $Vm (Alt+Y)"
            }
        }
        if ((Get-Date) -gt $end) { throw "admin work in $Vm did not finish in ${TimeoutSec}s" }
    }
    $out = Invoke-Guest $Vm "Get-Content '$log' -EA 0; Remove-Item '$log', '$done' -Force -EA 0" 60 -AllowFail
    if ($result -ne 'ok') { throw "admin work in $Vm failed: $(($out | Where-Object { $_ -match 'FAILED' }) -join ' ')" }
    $out
}

# Turns the consent prompt off in the VM, so later admin work is silent (one Alt+Y if it was on).
# Returns 'already' or 'done'. Call inside Use-VmPassword.
function Enable-GuestSilentAdmin([string]$Vm) {
    $get = "'CPBA=' + (Get-ItemProperty '$UacPolicy' -EA 0).ConsentPromptBehaviorAdmin"
    if (((Invoke-Guest $Vm $get 60) -join ' ') -match 'CPBA=0\b') { return 'already' }
    Invoke-GuestElevated $Vm -Code "Set-ItemProperty '$UacPolicy' ConsentPromptBehaviorAdmin 0 -Type DWord" -TimeoutSec 120 | Out-Null
    if (((Invoke-Guest $Vm $get 60) -join ' ') -notmatch 'CPBA=0\b') { throw "the consent prompt in $Vm is still on" }
    'done'
}

# The role's scheduled task in its VM: runs <role>.vbs on the desktop, no time
# limit, on battery too. Created when missing; returns its state.
function Install-RoleTask($Farm, $Role) {
    $task = $Role.Task
    $vbs = "$GuestDrop\$($Role.Name).vbs"
    Set-GuestText $Role.Vm $vbs (Get-RoleVbs $Role)
    $code = "if (-not (Get-ScheduledTask '$task' -ErrorAction SilentlyContinue)) { " +
        "`$a = New-ScheduledTaskAction -Execute wscript.exe -Argument '$vbs'; " +
        "`$p = New-ScheduledTaskPrincipal -UserId `"`$env:USERDOMAIN\`$env:USERNAME`" -LogonType Interactive; " +
        '$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew; ' +
        "Register-ScheduledTask '$task' -Action `$a -Principal `$p -Settings `$s | Out-Null; 'created' }; " +
        "'state=' + (Get-ScheduledTask '$task').State"
    Invoke-Guest $Role.Vm $code 60
}

# --- VPN (ExpressVPN in a VM, kept up by vm\guest-vpn-keeper.ps1) -------------------------
# farm.json: "vpn": {"provider": "expressvpn", "region": "uk-london", "fallback": ["uk-manchester"]}
# on a role, or for the whole farm; "vpn": false on a role leaves it out.
function Get-RoleVpnSetting($Def, $Role) {
    $v = if ($Role.PSObject.Properties['vpn']) { $Role.vpn } else { $Def.vpn }
    if ($v -is [pscustomobject] -and $v.region) { $v } else { $null }
}

$VpnCtl = 'C:\Program Files\ExpressVPN\expressvpnctl.exe'
$VpnTask = 'WizardEyesVpnKeeper'
$VpnStatusFile = "$GuestDrop\vpn.json"
$VpnCredential = 'wizard-eyes/expressvpn'

# What the VPN keeper in the VM last wrote (or $null). Call inside Use-VmPassword.
function Get-GuestVpnStatus([string]$Vm) {
    $out = Invoke-Guest $Vm "if (Test-Path '$VpnStatusFile') { 'VPN' + (Get-Content '$VpnStatusFile' -Raw) }" 60 -AllowFail
    $line = @($out | Where-Object { $_ -like 'VPN{*' })[0]
    if ($line) { $line.Substring(3) | ConvertFrom-Json } else { $null }
}

# Sets the VPN up in the VM and waits until it's connected with an exit address that
# isn't the ISP's. Throws if not: a bot must not start without it. Call inside Use-VmPassword.
function Start-GuestVpn($Farm, $Role, [scriptblock]$Say = { param($t) Write-Host $t }) {
    $v = $Role.Vpn
    if (-not $v) { return $null }
    if ($v.provider -and $v.provider -ne 'expressvpn') { throw "VPN provider '$($v.provider)' is not supported (expressvpn)" }
    $vm = $Role.Vm
    $installed = (Invoke-Guest $vm "Test-Path '$VpnCtl'" 60 -AllowFail) -join ' '
    if ($installed -notmatch 'True') { throw "ExpressVPN is not installed in $vm (install it by hand: it needs an admin prompt)" }

    # signed in? (the status text for "not signed in" is matched loosely: untested)
    $st = (Invoke-Guest $vm "& '$VpnCtl' status 2>&1 | Out-String" 60 -AllowFail) -join ' '
    if ($st -match '(?i)log ?in|sign ?in|activat|not logged') {
        $secret = Get-StoredSecret $VpnCredential
        if (-not $secret) { throw "ExpressVPN in $vm is not signed in and there is no $VpnCredential in Credential Manager (type it in your own terminal)" }
        & $Say "signing the VPN in ($vm)"
        # the login file only lives in the VM for the call (an activation code, or email + password on two lines)
        $lines = ($secret -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ }) -join "`r`n"
        Set-GuestText $vm "$GuestDrop\vpn-login.txt" $lines
        $r = (Invoke-Guest $vm "& '$VpnCtl' login '$GuestDrop\vpn-login.txt' 2>&1 | Out-String; Remove-Item '$GuestDrop\vpn-login.txt' -Force -EA 0" 90 -AllowFail) -join ' '
        $secret = $null
        if ($r -match '(?i)fail|error|invalid|denied') { throw "ExpressVPN login failed in $vm" }
    }

    $fallback = @($v.fallback) -join ','
    Invoke-Guest $vm ("& '$VpnCtl' background enable | Out-Null; & '$VpnCtl' set autoconnect true | Out-Null; & '$VpnCtl' set allowlan true | Out-Null; " +
        "& '$VpnCtl' set networklock true | Out-Null; & '$VpnCtl' set region '$($v.region)' | Out-Null") 90 | Out-Null

    # the keeper: a task at logon that connects and reconnects (made again, started now)
    $guestScript = "$GuestDrop\vpn-keeper.ps1"
    Copy-ToGuest $vm (Join-Path $PSScriptRoot '..\vm\guest-vpn-keeper.ps1') $guestScript
    $arg = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $guestScript -Region $($v.region) -Fallback `"$fallback`""
    Invoke-Guest $vm ("Stop-ScheduledTask '$VpnTask' -EA 0; Get-CimInstance Win32_Process -Filter `"Name='powershell.exe'`" | ? { `$_.CommandLine -match 'vpn-keeper' } | % { Stop-Process -Id `$_.ProcessId -Force -EA 0 }; " +
        "Unregister-ScheduledTask '$VpnTask' -Confirm:`$false -EA 0; " +
        "`$a = New-ScheduledTaskAction -Execute powershell.exe -Argument '$arg'; " +
        "`$t = New-ScheduledTaskTrigger -AtLogOn -User `"`$env:USERDOMAIN\`$env:USERNAME`"; " +
        "`$p = New-ScheduledTaskPrincipal -UserId `"`$env:USERDOMAIN\`$env:USERNAME`" -LogonType Interactive; " +
        '$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew; ' +
        "Register-ScheduledTask '$VpnTask' -Action `$a -Trigger `$t -Principal `$p -Settings `$s | Out-Null; Remove-Item '$VpnStatusFile' -EA 0; Start-ScheduledTask '$VpnTask'") 90 | Out-Null

    $isp = ((Invoke-Guest $vm "& '$VpnCtl' get pubip 2>&1 | Out-String" 60 -AllowFail) -join '').Trim()
    $end = (Get-Date).AddSeconds(120)
    $s = $null
    while ((Get-Date) -lt $end) {
        Start-Sleep 5
        $s = Get-GuestVpnStatus $vm
        if ($s -and $s.state -eq 'Connected' -and $s.exit_ip) {
            if ($isp -and $s.exit_ip -eq $isp) { throw "the VPN is connected but the exit address is still the ISP's ($isp)" }
            return $s
        }
    }
    throw "the VPN in $vm did not connect within 2 minutes (last: $(($s | ConvertTo-Json -Compress)))"
}

# The ONLY way to take a VM's VPN down on purpose. ExpressVPN lifts Network Lock when it is
# disconnected deliberately, so the real address would be visible: the bot, the RuneLite client
# and the Jagex Launcher are closed first (user's rule, 2026-11-26), and the keeper is stopped so
# it doesn't reconnect. Never run `expressvpnctl disconnect` in a VM by hand. Call inside Use-VmPassword.
function Stop-GuestVpn($Farm, $Role, [scriptblock]$Say = { param($t) Write-Host $t }) {
    $vm = $Role.Vm
    $leaf = if ($Role.Script) { [regex]::Escape((Split-Path $Role.Script -Leaf)) } else { 'a^' }
    & $Say "closing the bot, the game client and the Jagex Launcher in $vm"
    $left = (Invoke-Guest $vm ("Stop-ScheduledTask '$($Role.Task)' -EA 0; " +
        "Get-CimInstance Win32_Process | ? { (`$_.Name -match '^python' -and `$_.CommandLine -match '$leaf') -or `$_.Name -match '^(RuneLite|JagexLauncher|Jagex Launcher)' -or (`$_.Name -match '^javaw?\.exe`$' -and `$_.CommandLine -match 'RuneLite|runelite') } | % { Stop-Process -Id `$_.ProcessId -Force -EA 0 }; " +
        "for (`$i = 0; `$i -lt 15; `$i++) { `$n = @(Get-CimInstance Win32_Process | ? { `$_.Name -match '^(RuneLite|JagexLauncher|Jagex Launcher)' -or (`$_.Name -match '^javaw?\.exe`$' -and `$_.CommandLine -match 'RuneLite|runelite') }).Count; if (-not `$n) { break }; Start-Sleep 1 }; " +
        "'LEFT' + `$n") 90) -join ' '
    if ($left -notmatch 'LEFT0\s*$') { throw "the game client or launcher is still running in $vm ($left): not disconnecting the VPN" }
    & $Say "stopping the VPN keeper and disconnecting"
    $out = (Invoke-Guest $vm ("Stop-ScheduledTask '$VpnTask' -EA 0; Get-CimInstance Win32_Process -Filter `"Name='powershell.exe'`" | ? { `$_.CommandLine -match 'vpn-keeper' } | % { Stop-Process -Id `$_.ProcessId -Force -EA 0 }; " +
        "& '$VpnCtl' disconnect | Out-Null; Start-Sleep 3; 'STATE ' + (& '$VpnCtl' get connectionstate)") 90) -join ' '
    if ($out -notmatch 'STATE Disconnected') { throw "the VPN in $vm did not disconnect ($out)" }
}

# --- machine names (user, 2026-11-26): random, unique, with the ring of "motherbear" ------------
# A kin word / title + an animal, made once per machine and kept in
# %LOCALAPPDATA%\wizard-eyes\machine-names.json (key -> name). The key is the VM's VirtualBox name.
$MachineNameFirst = 'mother', 'father', 'sister', 'brother', 'auntie', 'uncle', 'cousin', 'granny', 'captain', 'duchess', 'baron', 'bishop', 'widow', 'hermit', 'pilgrim', 'abbot', 'squire', 'marshal', 'dowager', 'deacon'
$MachineNameSecond = 'bear', 'fox', 'wolf', 'otter', 'badger', 'heron', 'stoat', 'hare', 'owl', 'moth', 'newt', 'crow', 'stag', 'lynx', 'wren', 'mole', 'finch', 'vole', 'toad', 'swan'
$MachineNamesFile = Join-Path $env:LOCALAPPDATA 'wizard-eyes\machine-names.json'

function Get-MachineNames {
    if (Test-Path $MachineNamesFile) { Get-Content $MachineNamesFile -Raw | ConvertFrom-Json } else { [pscustomobject]@{} }
}

# The machine's name; made (and kept) the first time it's asked for. Never the same as another's.
function Get-MachineName([string]$Key, [string[]]$Taken = @('motherbear')) {
    $all = Get-MachineNames
    if ($all.PSObject.Properties[$Key]) { return $all.$Key }
    $used = @(@($all.PSObject.Properties.Value) + $Taken | Where-Object { $_ })
    # no first or second word repeated either, while there are words left (they should look different)
    $firsts = @($MachineNameFirst | Where-Object { $f = $_; -not ($used | Where-Object { $_.StartsWith($f) }) })
    $seconds = @($MachineNameSecond | Where-Object { $x = $_; -not ($used | Where-Object { $_.EndsWith($x) }) })
    if (-not $firsts) { $firsts = $MachineNameFirst }
    if (-not $seconds) { $seconds = $MachineNameSecond }
    do {
        $name = (Get-Random -InputObject $firsts) + (Get-Random -InputObject $seconds)
    } while ($used -contains $name)
    $all | Add-Member -NotePropertyName $Key -NotePropertyValue $name -Force
    New-Item -ItemType Directory -Force (Split-Path $MachineNamesFile) | Out-Null
    Write-Utf8 $MachineNamesFile ($all | ConvertTo-Json)
    $name
}

# --- naming a VM's Windows computer (needs admin: done once by a SYSTEM task, registered elevated) ---
# The task (WizardEyesRename, at startup, as SYSTEM) renames the computer from the first line of
# C:\wizard-eyes-provision\name.txt if it is a plain name (letters / digits, 3-15), deletes the file and
# restarts. The task's code lives in the task, not in a folder the bot user can write; the file is data.
$RenameTask = 'WizardEyesRename'
$RenameFile = "$GuestDrop\name.txt"

# Gives the VM its machine name: with the task in place it restarts into the new name; without it, the
# task is registered elevated (Invoke-GuestElevated) and renames + restarts at once. Returns 'already' or
# 'restarting' (then Wait-VmLoggedIn). Inside Use-VmPassword.
function Set-GuestMachineName([string]$Vm, [string]$Name) {
    $now = ((Invoke-Guest $Vm '$env:COMPUTERNAME' 60) -join '').Trim()
    if ($now -eq $Name.ToUpper()) { return 'already' }
    Set-GuestText $Vm $RenameFile $Name
    # (a SYSTEM task is hidden from the bot user without admin rights: look for it elevated)
    $has = (Invoke-GuestElevated $Vm -Code "'HAS=' + [bool](Get-ScheduledTask '$RenameTask' -EA 0)" -TimeoutSec 120) -join ' '
    if ($has -notmatch 'HAS=True') {
        Copy-ToGuest $Vm (Join-Path $PSScriptRoot '..\vm\guest-register-rename-task.ps1') "$GuestDrop\register-rename-task.ps1"
        $out = (Invoke-GuestElevated $Vm -Script "$GuestDrop\register-rename-task.ps1" -TimeoutSec 120) -join ' '
        if ($out -notmatch "registered $RenameTask") { throw "the $RenameTask task did not get registered in $Vm ($out)" }
    }
    Invoke-Guest $Vm "Restart-Computer -Force" 60 -AllowFail | Out-Null    # (the task renames at startup)
    'restarting'
}

Export-ModuleMember -Function * -Variable GuestUser, GuestDrop, GuestRoot, VmCredential
