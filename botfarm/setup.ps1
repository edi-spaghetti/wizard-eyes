<#
.SYNOPSIS
    Set a PC up to run a bot farm, from a bare Windows PC (after bootstrap.ps1
    has cloned the repos) to the bots working. Safe to re-run at any time.

.DESCRIPTION
    Every step checks first and only acts when it isn't done. It stops at the
    first step that is still not done after its turn (the later steps build on
    it): fix what it says and run it again. Steps that need a reboot register
    this script to run again after it (RunOnce) and offer to reboot.

      tools         git, VirtualBox (7.1+), Tailscale (only if the farm has a phone page)
      engine        the engine's own setup.ps1 (Python, venv, packages, R2 asset config)
      accounts      farm.local.json: the game account of each role (git-ignored)
      vm:NAME       each role's VM built and provisioned (vm\new-bot-vm.ps1); a new
                    build first needs the Windows ISO and the hypervisor off (reboot)
      checklist:ROLE  what only a person can do in the VM (Jagex login, RuneLite
                    profile, bank layouts, ...), from farm.json; ticked off here,
                    or done when the role's bot is working
      hypervisor    Windows hypervisor / Memory Integrity back on once the VMs are built
      services      install.ps1: default farm, service tasks, desktop icon
      alerts        Telegram (farm.json "alerts"): token, pairing with the phone, test
      phone         Tailscale (farm.json "phone"): logged in, page host + key, serve
      start         start-farm.ps1: services up, bots started

    Ticked checklists: %LOCALAPPDATA%\wizard-eyes\botfarm-setup.json

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File botfarm\setup.ps1 -Farm ..\wizard-eyes-experiments\botfarm\herbitrage

.EXAMPLE
    # only say what's done and what isn't; change nothing
    powershell -ExecutionPolicy Bypass -File botfarm\setup.ps1 -Look
#>
[CmdletBinding()]
param(
    # the farm's folder (default: $env:WIZARD_EYES_FARM, then this PC's default farm)
    [string]$Farm,
    # only check and report
    [switch]$Look,
    # step names (or patterns, e.g. 'vm:*') to leave out this time
    [string[]]$Skip = @(),
    # where a new VM's Windows ISO is (as new-bot-vm.ps1)
    [string]$Iso = (Join-Path $env:USERPROFILE 'VirtualBox VMs\iso\Win11_26H2_EnglishUK_x64.iso')
)

$ErrorActionPreference = 'Stop'
# (powershell -File passes 'a','b' as the one string "a,b")
$Skip = @($Skip | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$Here = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Import-Module (Join-Path $Here 'lib\BotFarm.psm1') -Force
$fm = Get-Farm $Farm
$StateFile = Join-Path $env:LOCALAPPDATA 'wizard-eyes\botfarm-setup.json'
$HypervisorState = Join-Path $env:LOCALAPPDATA 'wizard-eyes\hypervisor-state.json'
$RunOnceKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\RunOnce'
$RunOnceName = 'WizardEyesBotfarmSetup'
$Tailscale = Join-Path $env:ProgramFiles 'Tailscale\tailscale.exe'
$WindowsIsoPage = 'https://www.microsoft.com/software-download/windows11'
$Working = 480   # seconds since a bot's last log line: working (as start-farm.ps1)

function Write-Step([string]$t) { Write-Host "`n== $t" -ForegroundColor Cyan }
function Write-Ok([string]$t) { Write-Host "   OK   $t" -ForegroundColor Green }
function Write-Todo([string]$t) { Write-Host "   TODO $t" -ForegroundColor Yellow }
function Write-Info([string]$t) { Write-Host "        $t" }
function Write-Action([string]$t) { Write-Host "   >>   $t" -ForegroundColor Yellow }

function Confirm-Yes([string]$Question) {
    (Read-Host "   $Question [y/N]") -match '^(y|yes)$'
}
function Wait-Enter([string]$What) { [void](Read-Host "   $What, then press Enter") }

function Done([string]$Detail = '') { [pscustomobject]@{ Ok = $true; Detail = $Detail } }
function NotDone([string]$Detail) { [pscustomobject]@{ Ok = $false; Detail = $Detail } }

# --- state (ticked checklists) ----------------------------------------------
function Get-SetupState {
    $all = if (Test-Path $StateFile) { Get-Content $StateFile -Raw | ConvertFrom-Json } else { [pscustomobject]@{} }
    $mine = $all.PSObject.Properties[$fm.Dir]
    if ($mine) { $mine.Value } else { [pscustomobject]@{ ticked = [pscustomobject]@{} } }
}
function Set-Ticked([string]$Key) {
    $all = if (Test-Path $StateFile) { Get-Content $StateFile -Raw | ConvertFrom-Json } else { [pscustomobject]@{} }
    if (-not $all.PSObject.Properties[$fm.Dir]) {
        $all | Add-Member -NotePropertyName $fm.Dir -NotePropertyValue ([pscustomobject]@{ ticked = [pscustomobject]@{} })
    }
    $all.($fm.Dir).ticked | Add-Member -NotePropertyName $Key -NotePropertyValue (Get-Date -Format s) -Force
    New-Item -ItemType Directory -Force (Split-Path $StateFile) | Out-Null
    Write-Utf8 $StateFile ($all | ConvertTo-Json -Depth 5)
}

# --- helpers -----------------------------------------------------------------
function Request-Reboot([string]$Why) {
    $cmd = "powershell.exe -NoExit -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Farm `"$($fm.Dir)`""
    New-Item -Path $RunOnceKey -Force | Out-Null
    Set-ItemProperty -Path $RunOnceKey -Name $RunOnceName -Value $cmd
    Write-Action "$Why needs a reboot. This setup carries on by itself after it (when you log in)."
    if (Confirm-Yes 'Reboot now?') { Restart-Computer -Force }
    else { Write-Info 'Reboot when you are ready.' }
    exit 0
}

function Install-Winget([string]$Id, [string]$What) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget is missing (App Installer from the Microsoft Store); install $What by hand"
    }
    Write-Info "installing $What (winget $Id; accept the admin prompt)"
    winget install -e --id $Id --accept-source-agreements --accept-package-agreements --silent
    if ($LASTEXITCODE -ne 0) { Write-Info "winget exit code $LASTEXITCODE" }
}

# python -m MODULE ARGS with the engine + the farm's repo importable; output lines
function Invoke-FarmModule([string]$Module, [string[]]$ModuleArgs, [switch]$Live) {
    if (-not (Test-Path $fm.Python)) { return 'no engine venv yet (step "engine")' }
    $env:PYTHONPATH = "$($fm.Engine)\python;$($fm.Repo)"
    $env:WIZARD_EYES_FARM = $fm.Dir
    Push-Location $fm.Repo
    try {
        if ($Live) { & $fm.Python -m $Module @ModuleArgs; return }
        $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        $out = & cmd /c "`"$($fm.Python)`" -m $Module $($ModuleArgs -join ' ') 2>&1"
        $ErrorActionPreference = $old
        $out
    } finally { Pop-Location }
}

function Get-BotStatus([string]$Role) {
    try {
        $j = & (Join-Path $Here 'bot-control.ps1') -Role $Role -Action status -Farm $fm.Dir | Out-String
        $j | ConvertFrom-Json
    } catch { $null }
}
function Test-BotWorking([string]$Role) {
    $s = Get-BotStatus $Role
    $s -and $s.running -and $s.log_age -ne $null -and $s.log_age -lt $Working
}

function Get-TailscaleStatus {
    if (-not (Test-Path $Tailscale)) { return $null }
    try { & $Tailscale status --json | Out-String | ConvertFrom-Json } catch { $null }
}

# --- steps -------------------------------------------------------------------
$Steps = New-Object System.Collections.Generic.List[object]
# Check / Do get the step's -Arg as their first argument
function Add-Step([string]$Name, [string]$Title, [scriptblock]$Check, [scriptblock]$Do, $Arg = $null) {
    $Steps.Add([pscustomobject]@{ Name = $Name; Title = $Title; Check = $Check; Do = $Do; Arg = $Arg })
}

Add-Step 'tools' 'Tools: git, VirtualBox, Tailscale' {
    $miss = @()
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { $miss += 'git' }
    $vbox = Get-VBoxManage
    if (-not $vbox) { $miss += 'VirtualBox' }
    else {
        $ver = (Invoke-VBox @('--version')).Trim()
        if ([version]($ver -replace 'r\d+$', '') -lt [version]'7.1') { $miss += "VirtualBox 7.1+ (have $ver)" }
    }
    if ($fm.Def.phone -and -not (Test-Path $Tailscale)) { $miss += 'Tailscale' }
    if ($miss) { NotDone "missing: $($miss -join ', ')" } else { Done "git, VirtualBox $ver$(if ($fm.Def.phone) { ', Tailscale' })" }
} {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Install-Winget 'Git.Git' 'git' }
    if (-not (Get-VBoxManage)) { Install-Winget 'Oracle.VirtualBox' 'VirtualBox' }
    if ($fm.Def.phone -and -not (Test-Path $Tailscale)) { Install-Winget 'Tailscale.Tailscale' 'Tailscale' }
    # winget's installs change PATH / VBOX_MSI_INSTALL_PATH for new shells only
    $env:VBOX_MSI_INSTALL_PATH = [Environment]::GetEnvironmentVariable('VBOX_MSI_INSTALL_PATH', 'Machine')
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}

Add-Step 'engine' 'Engine: Python, venv, packages, R2 asset config' {
    if (-not (Test-Path $fm.Python)) { return NotDone 'no venv' }
    if (-not (Test-Path (Join-Path $fm.Engine 'config\assets.local.json'))) { return NotDone 'no R2 asset config' }
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $out = & cmd /c "`"$($fm.Python)`" -c `"import wizard_eyes, cv2, numpy`" 2>&1"
    $code = $LASTEXITCODE; $ErrorActionPreference = $old
    if ($code -ne 0) { NotDone "venv can't import the engine: $out" } else { Done $fm.Python }
} {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $fm.Engine 'setup.ps1')
}

Add-Step 'accounts' "Accounts ($(Split-Path $fm.LocalPath -Leaf), not in git)" {
    $miss = @($fm.Roles.Keys | Where-Object { -not (Get-Account $fm $_ -Quiet) })
    if ($miss) { return NotDone "no account for: $($miss -join ', ')" }
    git -C $fm.Repo check-ignore -q $fm.LocalPath
    if ($LASTEXITCODE -ne 0) { return NotDone "$($fm.LocalPath) is NOT git-ignored: add *.local.json to the repo's .gitignore" }
    Done "$($fm.Roles.Count) roles, git-ignored"
} {
    $local = $fm.Local
    if (-not $local.accounts) { $local | Add-Member -NotePropertyName accounts -NotePropertyValue ([pscustomobject]@{}) -Force }
    foreach ($r in $fm.Roles.Keys) {
        if (Get-Account $fm $r -Quiet) { continue }
        $name = (Read-Host "   Game account (display name) for the $r").Trim()
        if ($name) { $local.accounts | Add-Member -NotePropertyName $r -NotePropertyValue $name -Force }
    }
    Write-Utf8 $fm.LocalPath ($local | ConvertTo-Json -Depth 5)
    $script:fm = Get-Farm $fm.Dir
}

# one build step per VM (roles may share one)
foreach ($vmName in @($fm.Roles.Values | ForEach-Object Vm | Select-Object -Unique)) {
    $roles = @($fm.Roles.Values | Where-Object Vm -eq $vmName | ForEach-Object Name)
    Add-Step "vm:$vmName" "VM $vmName ($($roles -join ', '))" {
        param($vmName)
        $info = Get-VmInfo $vmName
        if (-not $info.ContainsKey('VMState')) { return NotDone 'not built yet' }
        if (-not (Test-VmRunning $vmName)) { return Done "exists ($($info['VMState'])); provisioning checked when it runs" }
        if (-not (Get-VmPassword)) { return NotDone "no VM password in Credential Manager ($VmCredential)" }
        $r = Use-VmPassword { Invoke-Guest $vmName "Test-Path '$GuestRoot\wizard-eyes\venv\Scripts\python.exe'; Test-Path '$GuestRoot\$($fm.RepoName)\.git'" 60 -AllowFail }
        $r = @($r | Where-Object { $_ -match 'True|False' })
        if ($r.Count -ge 2 -and $r[0] -eq 'True' -and $r[1] -eq 'True') { Done 'running, engine + farm repo provisioned' }
        else { NotDone 'running, not provisioned (no engine venv or farm repo)' }
    } {
        param($vmName)
        $info = Get-VmInfo $vmName
        if (-not $info.ContainsKey('VMState')) {
            # a new Windows install: the ISO, then the hypervisor off (installs hang under it)
            while (-not (Test-Path $Iso)) {
                Write-Action "Windows 11 ISO needed at $Iso"
                Write-Info "Download page opened: $WindowsIsoPage (Windows 11 Disk Image (ISO) for x64, English)."
                New-Item -ItemType Directory -Force (Split-Path $Iso) | Out-Null
                Start-Process $WindowsIsoPage
                Wait-Enter "Save it as $Iso"
            }
            if ((Get-CimInstance Win32_ComputerSystem).HypervisorPresent) {
                if (-not (Test-Path $HypervisorState)) {
                    Write-Action 'The Windows hypervisor must be off while VMs install (Windows setup hangs under it).'
                    Write-Info 'This turns Memory Integrity and the hypervisor off (admin prompt); a later step turns them back on.'
                    if (-not (Confirm-Yes 'Turn them off now?')) { return }
                    & (Join-Path $Here 'vm\hypervisor.ps1') off
                }
                Request-Reboot 'Switching the hypervisor off'
            }
        }
        if (-not $info.ContainsKey('VMState') -and -not (Confirm-Yes "Build $vmName now (Windows install + provisioning, about an hour)?")) { return }
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Here 'vm\new-bot-vm.ps1') -Name $vmName -Farm $fm.Dir -Iso $Iso
    } $vmName
}

$vmChecklist = @(
    'Jagex Launcher: finish its install (admin prompt in the VM), log in with the account',
    'Jagex Launcher: RuneLite as the Old School client, start the game once',
    'RuneLite: import the engine''s data\runelite\naughty.properties as a profile, activate it; GPU plugin off; sidebar closed',
    'Game: Resizable - Modern layout, default zoom, screen brightness at minimum'
)
foreach ($role in $fm.Roles.Values) {
    $items = @($vmChecklist) + @($fm.Def.checklist.all) + @($fm.Def.checklist.($role.Name)) | Where-Object { $_ }
    Add-Step "checklist:$($role.Name)" "By hand in $($role.Vm) for the $($role.Name) ($($items.Count) items)" {
        param($a); $role = $a.Role; $items = $a.Items
        $t = (Get-SetupState).ticked
        $left = @(for ($i = 0; $i -lt $items.Count; $i++) { if (-not $t.PSObject.Properties["$($role.Name):$i"]) { $i } })
        if (-not $left) { return Done 'all ticked' }
        if ((Test-VmRunning $role.Vm) -and (Test-BotWorking $role.Name)) { return Done 'the bot is working' }
        NotDone "$($left.Count) of $($items.Count) not ticked yet"
    } {
        param($a); $role = $a.Role; $items = $a.Items
        if (-not (Test-VmRunning $role.Vm)) { Start-FarmVm $role.Vm -Window } else { Show-VmWindow $role.Vm }
        Write-Action "In the $($role.Vm) window (mouse: click into it; Right Ctrl gives it back):"
        $t = (Get-SetupState).ticked
        for ($i = 0; $i -lt $items.Count; $i++) {
            if ($t.PSObject.Properties["$($role.Name):$i"]) { Write-Ok $items[$i]; continue }
            Write-Info "[ ] $($items[$i])"
            if (Confirm-Yes 'Done?') { Set-Ticked "$($role.Name):$i" } else { Write-Info '    (left for later)'; return }
        }
    } ([pscustomobject]@{ Role = $role; Items = $items })
}

Add-Step 'hypervisor' 'Windows hypervisor + Memory Integrity back on' {
    if (Test-Path $HypervisorState) { return NotDone 'switched off for the VM builds; not back on yet' }
    $hvci = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\DeviceGuard\Scenarios\HypervisorEnforcedCodeIntegrity' -ErrorAction SilentlyContinue).Enabled
    $now = (Get-CimInstance Win32_ComputerSystem).HypervisorPresent
    if ($hvci -eq 1 -and -not $now) { return Done 'on from the next reboot' }
    Done $(if ($hvci -eq 1) { 'on' } else { 'as it was before (Memory Integrity was off)' })
} {
    & (Join-Path $Here 'vm\hypervisor.ps1') on
    Write-Info 'Takes effect at the next reboot (no hurry: the bots run either way).'
}

Add-Step 'services' 'Services as scheduled tasks, desktop icon (install.ps1)' {
    if ((Get-MachineSettings).default -ne $fm.Dir) { return NotDone "this PC's default farm is $((Get-MachineSettings).default)" }
    $out = & (Join-Path $Here 'install.ps1') -Farm $fm.Dir -Look 6>&1 | Out-String
    $todo = @($out -split "`r?`n" | Where-Object { $_ -match 'would be' } | ForEach-Object { $_.Trim() })
    if ($todo) { NotDone ($todo -join '; ') } else { Done "$(@($fm.Def.services).Count) service tasks + icon up to date" }
} {
    & (Join-Path $Here 'install.ps1') -Farm $fm.Dir
}

if ($fm.Def.alerts.telegram) {
    $alertModule = $fm.Def.alerts.telegram
    $tgCred = if ($fm.Def.alerts.credential) { $fm.Def.alerts.credential } else { 'wizard-eyes/telegram' }
    Add-Step 'alerts' 'Telegram alerts' {
        if (-not (Get-StoredSecret $tgCred)) { return NotDone "no bot token in Credential Manager ($tgCred)" }
        $s = (Invoke-FarmModule $alertModule @('status')) -join ' '
        if ($s -match '\bpaired') { if ($s -match 'not paired') { NotDone 'token stored, phone not paired' } else { Done 'token stored, paired' } }
        else { NotDone "status: $s" }
    } {
        if (-not (Get-StoredSecret $tgCred)) {
            Write-Action 'Make a Telegram bot: in Telegram, open @BotFather, send /newbot, pick a name; it gives a token.'
            Start-Process 'https://t.me/BotFather'
            Write-Info 'Then paste the token at the password prompt below (nothing shows while you type).'
            cmdkey /generic:$tgCred /user:bot /pass
            if (-not (Get-StoredSecret $tgCred)) { Write-Info 'no token stored'; return }
        }
        $s = (Invoke-FarmModule $alertModule @('status')) -join ' '
        if ($s -match 'not paired') {
            Write-Action 'Pairing: send your bot the message shown below from your phone (waits a few minutes).'
            Invoke-FarmModule $alertModule @('pair') -Live
        }
        $s = (Invoke-FarmModule $alertModule @('status')) -join ' '
        if ($s -notmatch 'not paired' -and $s -match 'paired') {
            Invoke-FarmModule $alertModule @('test') -Live
            Write-Info 'A test alert went to your phone.'
        }
    }
}

if ($fm.Def.phone) {
    $phonePort = $fm.Def.phone.port
    $phoneModule = $fm.Def.phone.module
    Add-Step 'phone' "Phone page through Tailscale (port $phonePort)" {
        $ts = Get-TailscaleStatus
        if (-not $ts) { return NotDone 'Tailscale not installed / not answering' }
        if ($ts.BackendState -ne 'Running') { return NotDone "Tailscale: $($ts.BackendState) (log in)" }
        $host_ = $ts.Self.DNSName.TrimEnd('.')
        $s = (Invoke-FarmModule $phoneModule @('status')) -join ' '
        if ($s -notmatch "host: $([regex]::Escape($host_))") { return NotDone "page host not set to $host_ ($s)" }
        if ($s -notmatch 'key: set') { return NotDone 'no key for the page' }
        $serve = (& $Tailscale serve status --json) -join ' '
        if ($serve -notmatch "127\.0\.0\.1:$phonePort") { return NotDone "tailscale serve doesn't point at port $phonePort" }
        Done "https://$host_ (tailnet only)"
    } {
        $ts = Get-TailscaleStatus
        if (-not $ts -or $ts.BackendState -ne 'Running') {
            Write-Action 'Log in to Tailscale (a browser page opens); use the same account on the phone (Tailscale app).'
            & $Tailscale up
            $ts = Get-TailscaleStatus
            if (-not $ts -or $ts.BackendState -ne 'Running') { Write-Info 'Tailscale is not logged in yet'; return }
        }
        $host_ = $ts.Self.DNSName.TrimEnd('.')
        $s = (Invoke-FarmModule $phoneModule @('status')) -join ' '
        if ($s -notmatch "host: $([regex]::Escape($host_))") { Invoke-FarmModule $phoneModule @('host', $host_) -Live }
        if ($s -notmatch 'key: set') {
            Write-Action 'A new key for the phone page is shown ONCE: keep it in your password manager.'
            Invoke-FarmModule $phoneModule @('key') -Live
            Wait-Enter 'Saved it'
        }
        $serve = (& $Tailscale serve status --json) -join ' '
        if ($serve -notmatch "127\.0\.0\.1:$phonePort") {
            Write-Info 'HTTPS certificates must be on for the tailnet (admin console > DNS) the first time.'
            & $Tailscale serve --bg $phonePort
        }
        Write-Info "Phone: Tailscale app logged in, then open https://$host_"
        if ($fm.Def.phone.app) { Write-Info "App: $($fm.Def.phone.app)" }
    }
}

Add-Step 'start' 'Bots working' {
    $off = @($fm.Roles.Keys | Where-Object { -not (Test-BotWorking $_) })
    if ($off) { NotDone "not working: $($off -join ', ')" } else { Done "all $($fm.Roles.Count) working" }
} {
    & (Join-Path $Here 'start-farm.ps1') -Farm $fm.Dir
    Write-Info 'Started; logging in takes a few minutes. Watch them on the page.'
    $script:StartedBots = $true
}

# --- run ---------------------------------------------------------------------
Write-Host "$($fm.Name) farm setup$(if ($Look) { ' (look only)' })" -ForegroundColor Cyan
Write-Info "farm $($fm.Dir)"
Write-Info "engine $($fm.Engine)"
if (-not $Look) { Remove-ItemProperty -Path $RunOnceKey -Name $RunOnceName -ErrorAction SilentlyContinue }

$todo = 0; $skipped = 0
foreach ($s in $Steps) {
    Write-Step "$($s.Name): $($s.Title)"
    if ($Skip | Where-Object { $s.Name -like $_ }) { Write-Info 'skipped (-Skip)'; $skipped++; continue }
    $r = @(& $s.Check $s.Arg)[-1]
    if ($r.Ok) { Write-Ok $r.Detail; continue }
    if ($Look) { Write-Todo $r.Detail; $todo++; continue }
    Write-Todo $r.Detail
    & $s.Do $s.Arg
    $r = @(& $s.Check $s.Arg)[-1]
    if ($r.Ok) { Write-Ok $r.Detail; continue }
    if ($s.Name -eq 'start' -and $script:StartedBots) { Write-Info $r.Detail; continue }
    Write-Todo $r.Detail
    Write-Host "`nStopped at '$($s.Name)': run this again once it's sorted (done steps are skipped)." -ForegroundColor Yellow
    exit 1
}
if ($Look -and $todo) { Write-Host "`n$todo step(s) to do." -ForegroundColor Yellow; exit 1 }
if ($skipped) { Write-Host "`nThe rest is set up ($skipped step(s) skipped)." -ForegroundColor Green }
else { Write-Host "`nEverything is set up." -ForegroundColor Green }
