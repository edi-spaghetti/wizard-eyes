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

# inside every bot VM (vm\guest-provision.ps1 sets them up). Each VM's Windows user has a human name of
# its own (Get-GuestUser; VMs from before 2026-11-27 are 'bot'); the repos are in its
# profile (Get-GuestRoot)
$GuestDrop = 'C:\wizard-eyes-provision'
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
    $engine = "$(Get-GuestRoot $Role.Vm)\wizard-eyes"
    $repo = "$(Get-GuestRoot $Role.Vm)\$($Farm.RepoName)"
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
    # the last run's "logged in" can outlive a power-off (VM3, 2026-11-27): it would pass the login wait at once
    Invoke-VBox @('guestproperty', 'delete', $Vm, '/VirtualBox/GuestInfo/OS/LoggedInUsers') -AllowFail | Out-Null
    Start-Process cmd -ArgumentList '/c', "start `"`" /min /affinity $mask `"$dir\VBoxHeadless.exe`" --startvm $Vm" -WindowStyle Hidden
    # VBoxHeadless starts in the background: the state reads 'poweroff' for a few seconds
    $end = (Get-Date).AddSeconds(60)
    while (-not (Test-VmRunning $Vm)) {
        if ((Get-Date) -gt $end) { throw "$Vm did not start" }
        Start-Sleep 2
    }
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

if (-not ('BotFarm.CredWriter' -as [type])) {
    Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
namespace BotFarm {
  public static class CredWriter {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    struct CREDENTIAL {
      public int Flags; public int Type; public string TargetName; public string Comment;
      public long LastWritten; public int CredentialBlobSize; public IntPtr CredentialBlob;
      public int Persist; public int AttributeCount; public IntPtr Attributes; public string TargetAlias; public string UserName;
    }
    [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern bool CredWriteW(ref CREDENTIAL cred, int flags);
    // a generic credential, kept for this Windows user on this PC (CRED_PERSIST_LOCAL_MACHINE)
    public static void Write(string target, string user, string secret) {
      byte[] b = System.Text.Encoding.Unicode.GetBytes(secret);
      if (b.Length > 2560) throw new ArgumentException("secret too long for Credential Manager (" + b.Length + " bytes)");
      CREDENTIAL c = new CREDENTIAL(); c.Type = 1; c.TargetName = target; c.UserName = user; c.Persist = 2;
      c.CredentialBlobSize = b.Length; c.CredentialBlob = Marshal.AllocHGlobal(b.Length);
      try { Marshal.Copy(b, 0, c.CredentialBlob, b.Length); if (!CredWriteW(ref c, 0)) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error()); }
      finally { Marshal.FreeHGlobal(c.CredentialBlob); }
    }
  }
}
'@
}

# Stores a generic credential (any text, also several lines, up to 1280 characters). Never print it.
function Set-StoredSecret([string]$Target, [string]$Secret, [string]$User = 'wizard-eyes') {
    [BotFarm.CredWriter]::Write($Target, $User, $Secret)
}

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

# The VMs' Windows password: made once, shared by all bot VMs (each has its own user name).
function New-VmPassword {
    $chars = 'abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789'.ToCharArray()
    $bytes = New-Object byte[] 20
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $pw = -join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] })
    cmdkey /generic:$VmCredential /user:vm-user /pass:$pw | Out-Null
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
    $out = Invoke-VBox @('guestcontrol', $Vm, 'run', '--username', (Get-GuestUser $Vm), '--passwordfile', (Get-VmPasswordFile),
        '--exe', 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe', '--wait-stdout', '--wait-stderr',
        "--timeout=$($TimeoutSec * 1000)", '--', 'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-EncodedCommand', $enc) -TimeoutSec ($TimeoutSec + 30) -AllowFail:$AllowFail
    $out -split "`r?`n" | Where-Object { $_ -notmatch 'CLIXML|<Objs' }
}

# $Target: the full path of the file in the VM
function Copy-ToGuest([string]$Vm, [string]$Source, [string]$Target) {
    Invoke-VBox @('guestcontrol', $Vm, 'copyto', '--username', (Get-GuestUser $Vm), '--passwordfile', (Get-VmPasswordFile),
        '--target-directory', $Target, $Source) -TimeoutSec 600 | Out-Null
}

function Copy-FromGuest([string]$Vm, [string]$Source, [string]$TargetDir) {
    Invoke-VBox @('guestcontrol', $Vm, 'copyfrom', '--username', (Get-GuestUser $Vm), '--passwordfile', (Get-VmPasswordFile),
        # (a folder that exists needs its trailing \, else VBoxManage refuses it)
        '--target-directory', ($TargetDir.TrimEnd('\') + '\'), $Source) -TimeoutSec 600 | Out-Null
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
    Invoke-VBox @('guestcontrol', $Vm, 'start', '--username', (Get-GuestUser $Vm), '--passwordfile', (Get-VmPasswordFile),
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

# --- VPN in a VM ------------------------------------------------------------------------------
# farm.json "vpn", on a role or for the whole farm ("vpn": false on a role leaves it out):
#   {"provider": "proton"}  Proton VPN through WireGuard (user, 2026-11-27): one WireGuard config per VM,
#                           made on the Proton account page and stored with botfarm\vpn-config.ps1;
#                           vm\guest-wireguard.ps1 makes it a tunnel service + the firewall lock
#   (ExpressVPN, the provider before, was dropped 2026-11-28: user)
function Get-RoleVpnSetting($Def, $Role) {
    $v = if ($Role.PSObject.Properties['vpn']) { $Role.vpn } else { $Def.vpn }
    if ($v -is [pscustomobject] -and $v.provider) { $v } else { $null }
}

$WgTunnel = 'wizardeyes'
$WgDoneFile = "$GuestDrop\vpn-config.sha256"

# The Credential Manager target holding a VM's WireGuard config
function Get-VpnConfigTarget([string]$Vm) { "wizard-eyes/vpn/$Vm" }

# The VPN as the VM sees it, for a WireGuard tunnel: {state, region, exit_ip, lock}. Inside Use-VmPassword.
function Get-GuestWireGuardStatus([string]$Vm) {
    $code = "`$s = Get-Service 'WireGuardTunnel`$$WgTunnel' -EA 0; if (-not `$s) { 'WG=none' } else { " +
        "`$ip = ''; if (`$s.Status -eq 'Running') { try { `$ip = (Invoke-RestMethod https://api.ipify.org -TimeoutSec 8).ToString().Trim() } catch { } }; " +
        "`$lock = @(Get-NetFirewallProfile | ? { `$_.DefaultOutboundAction -ne 'Block' }).Count -eq 0; " +
        "'WG=' + (@{ state = [string]`$s.Status; exit_ip = `$ip; lock = `$lock } | ConvertTo-Json -Compress) }"
    $line = @(Invoke-Guest $Vm $code 60 -AllowFail | Where-Object { $_ -like 'WG=*' })[0]
    if (-not $line -or $line -eq 'WG=none') { return $null }
    $o = $line.Substring(3) | ConvertFrom-Json
    $state = if ($o.state -eq 'Running' -and $o.exit_ip) { 'Connected' } elseif ($o.state -eq 'Running') { 'Connecting' } else { 'Disconnected' }
    [pscustomobject]@{ state = $state; region = 'proton'; exit_ip = $o.exit_ip; lock = $o.lock; provider = 'proton' }
}

# The tunnel up in the VM, from its config in Credential Manager (put in again only when it changed),
# and connected with an exit address that isn't the ISP's. Throws if not. Inside Use-VmPassword.
function Start-GuestWireGuard($Role, [scriptblock]$Say) {
    $vm = $Role.Vm
    $conf = Get-StoredSecret (Get-VpnConfigTarget $vm)
    if (-not $conf) {
        throw ("no WireGuard config for $vm in Credential Manager ($(Get-VpnConfigTarget $vm)): make one on " +
            "account.protonvpn.com > Downloads > WireGuard configuration, then in your own terminal: " +
            "botfarm\vpn-config.ps1 -Vm $vm -Path <the .conf file>")
    }
    $hash = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($conf))) -replace '-'
    $done = ((Invoke-Guest $vm "Get-Content '$WgDoneFile' -EA 0" 60 -AllowFail) -join '').Trim()
    $st = Get-GuestWireGuardStatus $vm
    # (put in again also when the tunnel service is stopped: Stop-GuestVpn, or it failed)
    if ($done -ne $hash -or -not $st -or -not $st.lock -or $st.state -eq 'Disconnected') {
        & $Say "putting the WireGuard tunnel + firewall lock in $vm"
        Copy-ToGuest $vm (Join-Path $PSScriptRoot '..\vm\guest-wireguard.ps1') "$GuestDrop\guest-wireguard.ps1"
        # (the config is in the VM's drop folder only until the elevated script moves it)
        Set-GuestText $vm "$GuestDrop\wg-in.conf" $conf
        $conf = $null
        $out = (Invoke-GuestElevated $vm -Script "$GuestDrop\guest-wireguard.ps1" -ArgText "-ConfFile $GuestDrop\wg-in.conf -Tunnel $WgTunnel" -TimeoutSec 180) -join ' '
        Invoke-Guest $vm "Remove-Item '$GuestDrop\wg-in.conf' -Force -EA 0" 60 -AllowFail | Out-Null
        if ($out -notmatch 'WG-OK') { throw "the WireGuard tunnel did not come up in ${vm}: $(($out -replace '.*WG-FAILED: ', '').Trim())" }
        Set-GuestText $vm $WgDoneFile $hash
    }
    $isp = try { (Invoke-RestMethod https://api.ipify.org -TimeoutSec 10).ToString().Trim() } catch { '' }
    $end = (Get-Date).AddSeconds(60)
    while ($true) {
        $st = Get-GuestWireGuardStatus $vm
        if ($st -and $st.state -eq 'Connected') {
            if ($isp -and $st.exit_ip -eq $isp) { throw "the VPN in $vm is up but the exit address is motherbear's ($isp)" }
            if (-not $st.lock) { throw "the VPN in $vm is up but the firewall lock is off" }
            return $st
        }
        if ((Get-Date) -gt $end) { throw "the VPN in $vm did not connect within a minute (last: $(($st | ConvertTo-Json -Compress)))" }
        Start-Sleep 5
    }
}

# The VM's VPN (its WireGuard tunnel), or $null if it has none. Call inside Use-VmPassword.
function Get-GuestVpnStatus([string]$Vm) { Get-GuestWireGuardStatus $Vm }

# Sets the VPN up in the VM and waits until it's connected with an exit address that isn't this PC's.
# Throws if not: a bot must not start without it. Call inside Use-VmPassword.
function Start-GuestVpn($Farm, $Role, [scriptblock]$Say = { param($t) Write-Host $t }) {
    $v = $Role.Vpn
    if (-not $v) { return $null }
    if ($v.provider -ne 'proton') { throw "VPN provider '$($v.provider)' is not supported (proton; ExpressVPN was dropped 2026-11-28)" }
    Start-GuestWireGuard $Role $Say
}

# Takes a VM's VPN down on purpose: the bot, the RuneLite client and the Jagex Launcher are closed
# first (user's rule, 2026-11-26), then the tunnel is stopped. The firewall lock stays on, so the VM
# then has no internet at all (tested on vm5, 2026-11-28). Call inside Use-VmPassword.
function Stop-GuestVpn($Farm, $Role, [scriptblock]$Say = { param($t) Write-Host $t }) {
    $vm = $Role.Vm
    $leaf = if ($Role.Script) { [regex]::Escape((Split-Path $Role.Script -Leaf)) } else { 'a^' }
    & $Say "closing the bot, the game client and the Jagex Launcher in $vm"
    $left = (Invoke-Guest $vm ("Stop-ScheduledTask '$($Role.Task)' -EA 0; " +
        "Get-CimInstance Win32_Process | ? { (`$_.Name -match '^python' -and `$_.CommandLine -match '$leaf') -or `$_.Name -match '^(RuneLite|JagexLauncher|Jagex Launcher)' -or (`$_.Name -match '^javaw?\.exe`$' -and `$_.CommandLine -match 'RuneLite|runelite') } | % { Stop-Process -Id `$_.ProcessId -Force -EA 0 }; " +
        "for (`$i = 0; `$i -lt 15; `$i++) { `$n = @(Get-CimInstance Win32_Process | ? { `$_.Name -match '^(RuneLite|JagexLauncher|Jagex Launcher)' -or (`$_.Name -match '^javaw?\.exe`$' -and `$_.CommandLine -match 'RuneLite|runelite') }).Count; if (-not `$n) { break }; Start-Sleep 1 }; " +
        "'LEFT' + `$n") 90) -join ' '
    if ($left -notmatch 'LEFT0\s*$') { throw "the game client or launcher is still running in $vm ($left): not stopping the VPN" }
    & $Say "stopping the WireGuard tunnel"
    $out = (Invoke-GuestElevated $vm -Code "Stop-Service 'WireGuardTunnel`$$WgTunnel' -Force -EA 0; 'STATE ' + (Get-Service 'WireGuardTunnel`$$WgTunnel' -EA 0).Status" -TimeoutSec 120) -join ' '
    if ($out -notmatch 'STATE (Stopped)?\s*$') { throw "the tunnel in $vm did not stop ($out)" }
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

# --- the VMs' Windows users (user, 2026-11-27): a human name each, never the same twice -----------------
# A new VM is installed with a user of its own, made once and kept in %LOCALAPPDATA%\wizard-eyes\vm-users.json
# (VM name -> user, full name). VMs not in it (built before 2026-11-27) have 'bot'; they are not renamed.
# A linked clone has its source's user (to do: a sysprep'd template, so each clone makes its own).
$PersonFirst = 'dave', 'alan', 'rose', 'petunia', 'gordon', 'maureen', 'keith', 'brenda', 'colin', 'sheila', 'trevor',
    'pauline', 'derek', 'janet', 'nigel', 'beryl', 'clive', 'doris', 'graham', 'hazel', 'barry', 'marion', 'dennis',
    'joyce', 'malcolm', 'glenda', 'stanley', 'irene', 'roger', 'valerie'
$PersonLast = 'Hollis', 'Pryce', 'Atkins', 'Marsh', 'Fenwick', 'Doyle', 'Harker', 'Lomax', 'Briggs', 'Cotton', 'Whitlock',
    'Pennington', 'Sutcliffe', 'Rowe', 'Gill', 'Haines', 'Kemp', 'Ashworth', 'Barlow', 'Tate'
$LegacyGuestUser = 'bot'
$VmUsersFile = Join-Path $env:LOCALAPPDATA 'wizard-eyes\vm-users.json'

function Get-VmUsers {
    if (Test-Path $VmUsersFile) { Get-Content $VmUsersFile -Raw | ConvertFrom-Json } else { [pscustomobject]@{} }
}

function Save-VmUsers($All) {
    New-Item -ItemType Directory -Force (Split-Path $VmUsersFile) | Out-Null
    Write-Utf8 $VmUsersFile ($All | ConvertTo-Json)
}

# Records the VM's person ({user, full})
function Set-VmPerson([string]$Vm, $Person) {
    $all = Get-VmUsers
    $all | Add-Member -NotePropertyName $Vm -NotePropertyValue $Person -Force
    Save-VmUsers $all
}

# The VM's person ({user, full}); made (and kept) the first time it's asked for. -Name picks the
# first name (else a random unused one).
function Get-VmPerson([string]$Vm, [string]$Name) {
    $all = Get-VmUsers
    if ($all.$Vm -and $all.$Vm.user) { return $all.$Vm }
    $used = @($all.PSObject.Properties.Value | ForEach-Object { $_.user }) + $LegacyGuestUser
    $lasts = @($all.PSObject.Properties.Value | ForEach-Object { ("$($_.full)" -split ' ')[-1] })
    if (-not $Name) {
        $left = @($PersonFirst | Where-Object { $used -notcontains $_ })
        if (-not $left) { throw 'no first names left in $PersonFirst (lib\BotFarm.psm1): add some' }
        $Name = Get-Random -InputObject $left
    }
    $Name = $Name.ToLower()
    if ($Name -notmatch '^[a-z]{3,15}$') { throw "user names are 3-15 letters: '$Name'" }
    if ($used -contains $Name) { throw "$Name is another VM's user already" }
    $left = @($PersonLast | Where-Object { $lasts -notcontains $_ })
    if (-not $left) { $left = $PersonLast }
    $p = [pscustomobject]@{ user = $Name; full = "$((Get-Culture).TextInfo.ToTitleCase($Name)) $(Get-Random -InputObject $left)" }
    Set-VmPerson $Vm $p
    $p
}

# The user guest control logs in as
function Get-GuestUser([string]$Vm) {
    $p = (Get-VmUsers).$Vm
    if ($p -and $p.user) { $p.user } else { $LegacyGuestUser }
}

# The folder holding the repos in the VM (the user's profile)
function Get-GuestRoot([string]$Vm) { "C:\Users\$(Get-GuestUser $Vm)\wizard-eyes" }

# --- naming a VM's Windows computer (needs admin: done once by a SYSTEM task, registered elevated) ---
# The task (WizardEyesRename, at startup, as SYSTEM) renames the computer from the first line of
# C:\wizard-eyes-provision\name.txt if it is a plain name (letters / digits, 3-15), deletes the file and
# restarts. The task's code lives in the task, not in a folder the bot user can write; the file is data.
$RenameTask = 'WizardEyesRename'
$RenameFile = "$GuestDrop\name.txt"

# Registers the task, elevated. Inside Use-VmPassword.
function Register-GuestRenameTask([string]$Vm) {
    Copy-ToGuest $Vm (Join-Path $PSScriptRoot '..\vm\guest-register-rename-task.ps1') "$GuestDrop\register-rename-task.ps1"
    $out = (Invoke-GuestElevated $Vm -Script "$GuestDrop\register-rename-task.ps1" -TimeoutSec 120) -join ' '
    if ($out -notmatch "registered $RenameTask") { throw "the $RenameTask task did not get registered in $Vm ($out)" }
}

# Gives the VM its machine name: with the task in place it restarts into the new name; without it, the
# task is registered elevated (Invoke-GuestElevated) and renames + restarts at once. Returns 'already' or
# 'restarting' (then Wait-VmLoggedIn). Inside Use-VmPassword.
function Set-GuestMachineName([string]$Vm, [string]$Name) {
    $now = ((Invoke-Guest $Vm '$env:COMPUTERNAME' 60) -join '').Trim()
    if ($now -eq $Name.ToUpper()) { return 'already' }
    Set-GuestText $Vm $RenameFile $Name
    # (a SYSTEM task is hidden from the bot user without admin rights: look for it elevated)
    $has = (Invoke-GuestElevated $Vm -Code "'HAS=' + [bool](Get-ScheduledTask '$RenameTask' -EA 0)" -TimeoutSec 120) -join ' '
    if ($has -notmatch 'HAS=True') { Register-GuestRenameTask $Vm }
    Invoke-Guest $Vm "Restart-Computer -Force" 60 -AllowFail | Out-Null    # (the task renames at startup)
    'restarting'
}

# --- templates: a sysprep'd VM that clones are made from (new-bot-vm.ps1 -Template / -CloneFrom) ---------
# The template has Windows, the lockdown and the machine-wide software (VC++, WireGuard, Jagex Launcher),
# no logins, and is generalized by sysprep. Each linked clone of it gets, at its first start, its own
# machine SID, computer name and Windows user from an answer file on a small CD made here
# (New-CloneAnswerIso), then goes through guest-provision.ps1 like a fresh build.
$TemplateKey = 'wizard-eyes/template'
$InstallerDir = Join-Path $env:LOCALAPPDATA 'wizard-eyes\installers'

function Test-VmTemplate([string]$Vm) {
    (Invoke-VBox @('getextradata', $Vm, $TemplateKey) -AllowFail) -match 'Value: sysprep'
}

function Set-VmTemplate([string]$Vm) { Invoke-VBox @('setextradata', $Vm, $TemplateKey, 'sysprep') | Out-Null }

function Clear-VmTemplate([string]$Vm) { Invoke-VBox @('setextradata', $Vm, $TemplateKey) | Out-Null }

# An installer kept on motherbear (%LOCALAPPDATA%\wizard-eyes\installers): the newest file matching
# $Pattern; downloaded from $Url when there is none and a URL is known. Throws if neither.
function Get-HostInstaller([string]$Pattern, [string]$Url, [string]$FileName, [string]$Hint) {
    New-Item -ItemType Directory -Force $InstallerDir | Out-Null
    $f = Get-ChildItem $InstallerDir -Filter $Pattern -EA 0 | Sort-Object LastWriteTime | Select-Object -Last 1
    if ($f) { return $f.FullName }
    if (-not $Url) { throw "no installer '$Pattern' in $InstallerDir. $Hint" }
    $path = Join-Path $InstallerDir $FileName
    $old = $ProgressPreference; $ProgressPreference = 'SilentlyContinue'
    try { Invoke-WebRequest $Url -OutFile $path -UseBasicParsing } finally { $ProgressPreference = $old }
    $path
}

# Key presses into the VM (PS/2 set 1 scancodes, make + break)
$VmKeys = @{
    Enter = '1c', '9c'; Space = '39', 'b9'; Tab = '0f', '8f'; ShiftTab = '2a', '0f', '8f', 'aa'
}
function Send-VmKey([string]$Vm, [string]$Key) {
    Invoke-VBox (@('controlvm', $Vm, 'keyboardputscancode') + $VmKeys[$Key]) | Out-Null
}

# A screenshot of the VM's screen into %TEMP% (its path), for error messages
function Save-VmScreenshot([string]$Vm, [string]$Tag) {
    $p = Join-Path $env:TEMP "botfarm-$Vm-$Tag-$(Get-Date -Format HHmmss).png"
    Invoke-VBox @('controlvm', $Vm, 'screenshotpng', $p) -AllowFail | Out-Null
    $p
}

# The Jagex Launcher, installed machine-wide. Its installer (Qt) has no quiet mode that works (it ignores
# /quiet and the Qt-IFW command line, VM3 2026-11-27), so it runs elevated on the VM's desktop and the
# host presses its keys: Enter (Next), Shift+Tab + Space (accept the licence), Enter (Next), Enter
# (Install), then Enter (Finish). The launcher it starts is closed again: no one signs in to a template.
# Returns 'already' or 'installed'. Inside Use-VmPassword.
function Install-GuestJagexLauncher([string]$Vm) {
    $exe = 'C:\Program Files (x86)\Jagex Launcher\JagexLauncher.exe'
    $has = { ((Invoke-Guest $Vm "'HAS=' + (Test-Path '$exe')" 60 -AllowFail) -join ' ') -match 'HAS=True' }
    if (& $has) { return 'already' }
    $inst = Get-HostInstaller 'Jagex Launcher Installer*.exe' 'https://cdn.jagex.com/Jagex%20Launcher%20Installer.exe' 'Jagex Launcher Installer.exe'
    $sig = Get-AuthenticodeSignature $inst
    if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'O=Jagex Limited') { throw "the Jagex Launcher installer ($inst) is not signed by Jagex" }
    $target = "$GuestDrop\jagex-launcher-installer.exe"
    Copy-ToGuest $Vm $inst $target
    Invoke-GuestElevated $Vm -Code "Start-Process '$target'" -TimeoutSec 120 | Out-Null
    Start-Sleep 20      # (it loads)
    foreach ($k in 'Enter', 'ShiftTab', 'Space', 'Enter', 'Enter') { Send-VmKey $Vm $k; Start-Sleep 4 }
    $end = (Get-Date).AddMinutes(6)
    while (-not (& $has)) {
        if ((Get-Date) -gt $end) { throw "the Jagex Launcher did not install in $Vm (screen: $(Save-VmScreenshot $Vm jagex))" }
        Start-Sleep 10
    }
    Start-Sleep 15      # (the last page)
    Send-VmKey $Vm 'Enter'
    Start-Sleep 15
    # (the launcher it started runs elevated: closed elevated)
    Invoke-GuestElevated $Vm -Code ("Get-Process JagexLauncher, 'Jagex Launcher', installer -EA 0 | Stop-Process -Force -EA 0; " +
        "Start-Sleep 3; Remove-Item '$target' -Force -EA 0") -TimeoutSec 120 | Out-Null
    'installed'
}

# An ISO 9660 + Joliet image of a folder (Windows' own IMAPI2, nothing to install)
function New-IsoFile([string]$SourceDir, [string]$IsoPath, [string]$Label = 'CLONE') {
    if (-not ('WizardEyes.IsoWriter' -as [type])) {
        Add-Type -TypeDefinition @'
using System; using System.IO; using System.Runtime.InteropServices; using System.Runtime.InteropServices.ComTypes;
namespace WizardEyes {
  public static class IsoWriter {
    public static void Write(object stream, string path) {
      IStream s = (IStream)stream; byte[] buf = new byte[65536]; IntPtr pRead = Marshal.AllocHGlobal(4);
      try {
        using (FileStream f = File.Create(path)) {
          while (true) { s.Read(buf, buf.Length, pRead); int n = Marshal.ReadInt32(pRead); if (n <= 0) break; f.Write(buf, 0, n); }
        }
      } finally { Marshal.FreeHGlobal(pRead); }
    }
  }
}
'@
    }
    $fs = New-Object -ComObject IMAPI2FS.MsftFileSystemImage
    $fs.FileSystemsToCreate = 3
    $fs.VolumeName = $Label
    $fs.Root.AddTree($SourceDir, $false)
    [WizardEyes.IsoWriter]::Write($fs.CreateResultImage().ImageStream, $IsoPath)
}

# The answer file a clone of a sysprep'd template reads at its first start (Windows looks for
# Unattend.xml at the root of a CD): computer name, the user (an administrator, signed in
# automatically), no setup screens. The password is in it in plain text: the ISO is deleted once the
# clone has logged in (new-bot-vm stage 5), and Windows blanks it in its own cached copy.
function Get-CloneAnswerXml([string]$ComputerName, $Person, [string]$Password) {
    $e = { param($t) [Security.SecurityElement]::Escape($t) }
    $c = 'processorArchitecture="amd64" publicKeyToken="31bf3856ad364e35" language="neutral" versionScope="nonSxS"'
    $pw = "<Password><Value>$(& $e $Password)</Value><PlainText>true</PlainText></Password>"
    @"
<?xml version="1.0" encoding="utf-8"?>
<unattend xmlns="urn:schemas-microsoft-com:unattend" xmlns:wcm="http://schemas.microsoft.com/WMIConfig/2002/State">
  <settings pass="specialize">
    <component name="Microsoft-Windows-Shell-Setup" $c>
      <ComputerName>$(& $e $ComputerName)</ComputerName>
      <TimeZone>GMT Standard Time</TimeZone>
    </component>
  </settings>
  <settings pass="oobeSystem">
    <component name="Microsoft-Windows-International-Core" $c>
      <InputLocale>0809:00000809</InputLocale>
      <SystemLocale>en-GB</SystemLocale>
      <UILanguage>en-GB</UILanguage>
      <UserLocale>en-GB</UserLocale>
    </component>
    <component name="Microsoft-Windows-Shell-Setup" $c>
      <OOBE>
        <HideEULAPage>true</HideEULAPage>
        <HideOEMRegistrationScreen>true</HideOEMRegistrationScreen>
        <HideOnlineAccountScreens>true</HideOnlineAccountScreens>
        <HideWirelessSetupInOOBE>true</HideWirelessSetupInOOBE>
        <HideLocalAccountScreen>true</HideLocalAccountScreen>
        <ProtectYourPC>3</ProtectYourPC>
        <!-- (no first-run screens at all: without these Windows sat on "Just a moment" for good,
             vm5 2026-11-28; old settings, still honoured) -->
        <SkipMachineOOBE>true</SkipMachineOOBE>
        <SkipUserOOBE>true</SkipUserOOBE>
      </OOBE>
      <UserAccounts>
        <LocalAccounts>
          <LocalAccount wcm:action="add">
            $pw
            <DisplayName>$(& $e $Person.full)</DisplayName>
            <Group>Administrators</Group>
            <Name>$(& $e $Person.user)</Name>
          </LocalAccount>
        </LocalAccounts>
      </UserAccounts>
      <AutoLogon>
        $pw
        <Enabled>true</Enabled>
        <LogonCount>999</LogonCount>
        <Username>$(& $e $Person.user)</Username>
      </AutoLogon>
    </component>
  </settings>
</unattend>
"@
}

# The clone's answer-file CD, in the VM's folder (named Unattended-*, so stage 5's clean-up deletes it),
# attached to its first CD drive. Inside Use-VmPassword.
function New-CloneAnswerIso([string]$Vm, [string]$ComputerName, $Person) {
    $dir = Split-Path (Get-VmInfo $Vm)['CfgFile']
    $src = Join-Path $env:TEMP "botfarm-answer-$PID"
    New-Item -ItemType Directory -Force $src | Out-Null
    try {
        $xml = Get-CloneAnswerXml $ComputerName $Person (Get-VmPassword)
        [IO.File]::WriteAllText((Join-Path $src 'Unattend.xml'), $xml, (New-Object Text.UTF8Encoding($false)))
        $iso = Join-Path $dir 'Unattended-clone.iso'
        Remove-Item $iso -Force -EA 0
        New-IsoFile $src $iso 'ANSWER'
    } finally { Remove-Item $src -Recurse -Force -EA 0 }
    Invoke-VBox @('storageattach', $Vm, '--storagectl', 'SATA', '--port', '1', '--device', '0', '--type', 'dvddrive', '--medium', $iso) | Out-Null
    $iso
}

# --- the farm's secrets (M3): what a farm needs, and whether it's there ---------------------------------
# All live in this PC's Credential Manager (typed in by the user, in their own terminal; never in the chat,
# never printed). Get-FarmSecrets lists them for a farm with a state each: ok / missing / bad (+ why).
# A login to a game account: wizard-eyes/jagex/<account> (user = its email, password = its password).
function Get-JagexCredentialTarget([string]$Account) { "wizard-eyes/jagex/$Account" }
$GmailCredential = 'wizard-eyes/gmail'

# The user name stored with a generic credential (e.g. an account's email), or $null. Not a secret.
function Get-StoredUser([string]$Target) {
    $line = cmdkey /list:$Target | Where-Object { $_ -match '^\s*User:\s*(.+)$' } | Select-Object -First 1
    if ($line -and $line -match '^\s*User:\s*(.+)$') { $Matches[1].Trim() } else { $null }
}

# Problems in a WireGuard config's text (empty = fine); the same checks as vpn-config.ps1
function Test-WireGuardConfig([string]$Text) {
    $bad = @()
    if ($Text -notmatch '(?m)^\s*\[Interface\]') { $bad += 'no [Interface]' }
    if ($Text -notmatch '(?m)^\s*PrivateKey\s*=\s*\S{40,}') { $bad += 'no PrivateKey' }
    if ($Text -notmatch '(?m)^\s*Address\s*=\s*\d+\.\d+\.\d+\.\d+') { $bad += 'no IPv4 Address' }
    if (([regex]::Matches($Text, '(?m)^\s*\[Peer\]')).Count -ne 1) { $bad += 'not exactly one [Peer]' }
    if ($Text -notmatch '(?m)^\s*Endpoint\s*=\s*\d+\.\d+\.\d+\.\d+:\d+\s*$') { $bad += 'no Endpoint = IPv4:port' }
    $bad
}

# Every secret the farm needs (and the optional ones), checked. -Online also asks Telegram whether its
# token works. Returns objects: target, for, state (ok / missing / bad / optional), note, howto.
function Get-FarmSecrets($Farm, [switch]$Online) {
    $out = New-Object System.Collections.Generic.List[object]
    function Add($Target, $For, $State, $Note, $HowTo) {
        $out.Add([pscustomobject]@{ target = $Target; for = $For; state = $State; note = $Note; howto = $HowTo })
    }
    # the VMs' Windows password (made by new-bot-vm when missing)
    if (Get-StoredSecret $VmCredential) { Add $VmCredential 'VM Windows logins' 'ok' '' '' }
    else { Add $VmCredential 'VM Windows logins' 'missing' 'made by new-bot-vm.ps1 at the first build' '' }

    # Telegram
    if ($Farm.Def.alerts -and $Farm.Def.alerts.telegram) {
        $t = if ($Farm.Def.alerts.credential) { $Farm.Def.alerts.credential } else { 'wizard-eyes/telegram' }
        $how = "@BotFather in Telegram, /newbot; then: cmdkey /generic:$t /user:bot /pass (paste the token)"
        $tok = Get-StoredSecret $t
        if (-not $tok) { Add $t 'alerts (Telegram)' 'missing' '' $how }
        elseif ($tok.Trim() -notmatch '^\d+:[\w-]{30,}$') { Add $t 'alerts (Telegram)' 'bad' 'not shaped like a bot token' $how }
        elseif ($Online) {
            $ok = try { (Invoke-RestMethod "https://api.telegram.org/bot$($tok.Trim())/getMe" -TimeoutSec 15).ok } catch { $false }
            if ($ok) { Add $t 'alerts (Telegram)' 'ok' 'Telegram knows the bot' '' } else { Add $t 'alerts (Telegram)' 'bad' 'Telegram refused the token' $how }
        } else { Add $t 'alerts (Telegram)' 'ok' '' '' }
        $tok = $null
    }

    foreach ($r in $Farm.Roles.Values) {
        # the role's VPN
        if ($r.Vpn -and $r.Vpn.provider -eq 'proton') {
            $t = Get-VpnConfigTarget $r.Vm
            $how = "account.protonvpn.com > Downloads > WireGuard configuration; then: botfarm\vpn-config.ps1 -Vm $($r.Vm) -Path <file.conf>"
            $c = Get-StoredSecret $t
            if (-not $c) { Add $t "VPN of $($r.Vm) ($($r.Name))" 'missing' '' $how }
            else {
                $bad = Test-WireGuardConfig $c
                $ep = if ($c -match '(?m)^\s*Endpoint\s*=\s*(\S+)') { $Matches[1] } else { '?' }
                if ($bad) { Add $t "VPN of $($r.Vm) ($($r.Name))" 'bad' ($bad -join ', ') $how }
                else { Add $t "VPN of $($r.Vm) ($($r.Name))" 'ok' "server $ep" '' }
            }
            $c = $null
        }
        # the role's game account
        $acct = Get-Account $Farm $r.Name -Quiet
        if (-not $acct) {
            Add "accounts.$($r.Name)" "game account of the $($r.Name)" 'missing' "no account name in $($Farm.LocalPath)" "add `"accounts`": {`"$($r.Name)`": `"<account>`"} to farm.local.json"
            continue
        }
        $t = Get-JagexCredentialTarget $acct
        $how = "cmdkey /generic:$t /user:<the account's email> /pass"
        if (-not (Get-StoredSecret $t)) { Add $t "Jagex login of $acct ($($r.Name))" 'missing' '' $how }
        elseif ((Get-StoredUser $t) -notmatch '@') { Add $t "Jagex login of $acct ($($r.Name))" 'bad' 'the user name is not an email address' $how }
        else { Add $t "Jagex login of $acct ($($r.Name))" 'ok' '' '' }
        # its bank PIN lives in the VM's own Credential Manager (bank_pin.py), not here
        if ($Farm.Local.bank_pin -and $Farm.Local.bank_pin.($r.Name) -eq $false) { }
        else { Add "wizard-eyes/bank-pin/$acct (in $($r.Vm))" "bank PIN of $acct" 'optional' 'kept in the VM, checked at the bot start' "in the VM: cmdkey /generic:wizard-eyes/bank-pin/$acct /user:$acct /pass" }
    }

    # email codes at the Jagex login (optional: without it a code pauses the setup and Telegram says so)
    $how = "Google: 2-Step Verification on, then myaccount.google.com/apppasswords; cmdkey /generic:$GmailCredential /user:<gmail address> /pass (the 16-letter app password)"
    $g = Get-StoredSecret $GmailCredential
    if (-not $g) { Add $GmailCredential 'email codes at the Jagex login' 'optional' 'not stored' $how }
    elseif ($g -replace '\s' -notmatch '^[a-z]{16}$') { Add $GmailCredential 'email codes at the Jagex login' 'bad' 'not an app password (16 letters): Google refuses normal passwords for IMAP' $how }
    else { Add $GmailCredential 'email codes at the Jagex login' 'ok' '' '' }
    $g = $null

    # read-only asset storage config (a file, not a secret in Credential Manager)
    $cfg = Join-Path $Farm.Engine 'config\assets.local.json'
    if (Test-Path $cfg) { Add 'config\assets.local.json' 'assets (R2, read-only into the VMs)' 'ok' '' '' }
    else { Add 'config\assets.local.json' 'assets (R2, read-only into the VMs)' 'missing' '' 'engine setup.ps1' }
    $out
}

Export-ModuleMember -Function * -Variable GuestDrop, VmCredential
