<#
.SYNOPSIS
    Set a bot VM up for Wizard Eyes. Runs INSIDE the guest as the 'bot' user
    (botfarm\vm\new-bot-vm.ps1 copies it in with the repo bundles and runs it).

.DESCRIPTION
    Safe to re-run (that's also how repos are refreshed):
      1. MinGit (portable git, no installer, no admin)
      2. Repos from the bundles in -Drop, listed in its remotes.json (the
         engine and the farm's repo): clone, or fast-forward main, under
         C:\Users\bot\wizard-eyes\
      3. Read-only R2 config into the engine
      4. RuneLite (silent per-user install)
      5. Engine setup.ps1 -NonInteractive (Python 3.9, venv, packages, OCR
         data, asset check, smoke test)
      5b. RuneLite profile: the reference "naughty" profile imported and active
      6. Jagex Launcher: installer downloaded and opened on the VM's desktop
         (it needs an admin prompt clicked there)
    Prints the manual checklist at the end.
#>
param([Parameter(Mandatory = $true)][string]$Drop)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is very slow with it
$Root = Join-Path $env:USERPROFILE 'wizard-eyes'
$Engine = Join-Path $Root 'wizard-eyes'
# the repos sent in (the engine + the farm's), with their GitHub remotes
$Remotes = Get-Content (Join-Path $Drop 'remotes.json') -Raw | ConvertFrom-Json
$RuneLiteInstallerUrl = 'https://github.com/runelite/launcher/releases/latest/download/RuneLiteSetup.exe'
$JagexLauncherInstallerUrl = 'https://cdn.jagex.com/Jagex%20Launcher%20Installer.exe'

function Write-Step([string]$t) { "`n== $t" }
New-Item -ItemType Directory -Force $Root | Out-Null

Write-Step '1. git'
$git = Join-Path $env:LOCALAPPDATA 'MinGit\cmd\git.exe'
if (-not (Test-Path $git)) {
    $rel = Invoke-RestMethod 'https://api.github.com/repos/git-for-windows/git/releases/latest' -UseBasicParsing
    $asset = $rel.assets | Where-Object { $_.name -match '^MinGit-[\d.]+-64-bit\.zip$' } | Select-Object -First 1
    $zip = Join-Path $env:TEMP $asset.name
    Invoke-WebRequest $asset.browser_download_url -OutFile $zip -UseBasicParsing
    Expand-Archive $zip (Join-Path $env:LOCALAPPDATA 'MinGit') -Force
    Remove-Item $zip
}
"   OK   $(& $git --version)"

Write-Step '2. Repos'
foreach ($name in $Remotes.PSObject.Properties.Name) {
    $bundle = Join-Path $Drop "$name.bundle"
    $dir = Join-Path $Root $name
    if (-not (Test-Path (Join-Path $dir '.git'))) {
        & $git clone -q -b main $bundle $dir
        if ($LASTEXITCODE -ne 0) { throw "clone of $name failed" }
        & $git -C $dir remote set-url origin $Remotes.$name
        "   OK   $name cloned at $(& $git -C $dir rev-parse --short HEAD)"
    } else {
        & $git -C $dir fetch -q $bundle main:refs/remotes/bundle/main
        if ($LASTEXITCODE -ne 0) { throw "fetch of $name failed" }
        $msg = & $git -C $dir merge --ff-only -q bundle/main
        if ($LASTEXITCODE -ne 0) { "   WARN $name not fast-forwarded (local changes?): $msg" }
        else { "   OK   $name at $(& $git -C $dir rev-parse --short HEAD)" }
    }
}

Write-Step '3. R2 config (read-only)'
New-Item -ItemType Directory -Force (Join-Path $Engine 'config') | Out-Null
Copy-Item (Join-Path $Drop 'assets.local.json') (Join-Path $Engine 'config\assets.local.json') -Force
'   OK   config\assets.local.json'

Write-Step '4. RuneLite'
$rl = Join-Path $env:LOCALAPPDATA 'RuneLite\RuneLite.exe'
if (-not (Test-Path $rl)) {
    $inst = Join-Path $env:TEMP 'RuneLiteSetup.exe'
    Invoke-WebRequest $RuneLiteInstallerUrl -OutFile $inst -UseBasicParsing
    $p = Start-Process $inst -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-' -Wait -PassThru
    "   installer exit $($p.ExitCode)"
}
if (Test-Path $rl) { '   OK   RuneLite installed' } else { '   WARN RuneLite not found after install' }

Write-Step '4b. Visual C++ runtime (tesserocr needs msvcp140.dll)'
$dll = "$env:SystemRoot\System32\msvcp140.dll"
if (-not (Test-Path $dll)) {
    $vc = Join-Path $Drop 'vc_redist.x64.exe'
    Invoke-WebRequest 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $vc -UseBasicParsing
    $sig = Get-AuthenticodeSignature $vc
    if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') { throw 'vc_redist: bad signature' }
    # needs an admin prompt, so open it on the logged-in desktop and wait
    $action = New-ScheduledTaskAction -Execute $vc -Argument '/install /passive /norestart'
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
    Register-ScheduledTask -TaskName 'WizardEyesVcRedist' -Action $action -Principal $principal -Force | Out-Null
    Start-ScheduledTask -TaskName 'WizardEyesVcRedist'
    $end = (Get-Date).AddMinutes(10)
    while (-not (Test-Path $dll) -and (Get-Date) -lt $end) { Start-Sleep 5 }
    Unregister-ScheduledTask -TaskName 'WizardEyesVcRedist' -Confirm:$false
}
if (Test-Path $dll) { '   OK   msvcp140.dll' } else { '   WARN not installed (accept the admin prompt on the VM desktop, re-run)' }
Write-Step '5. Engine setup (setup.ps1 -NonInteractive)'
# PS 5.1: redirected native stderr would be a terminating error under 'Stop'
$ErrorActionPreference = 'Continue'
$setupOut = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Engine 'setup.ps1') -NonInteractive 2>&1 |
    ForEach-Object { "$_" }
$setupExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($setupExit -ne 0) { "   WARN setup.ps1 exit code $setupExit"; $setupOut | Select-Object -Last 15 }
$summary = $false
foreach ($l in $setupOut) { if ($l -match '^== Summary') { $summary = $true; continue }; if ($summary -and $l -match '^==') { break }; if ($summary -and $l.Trim()) { $l } }

Write-Step '5b. RuneLite profile (the reference "naughty" profile, made the active one)'
# the same files as RuneLite's own Import button; not while RuneLite runs (it would overwrite them)
$ErrorActionPreference = 'Continue'
$env:PYTHONPATH = Join-Path $Engine 'python'
$profOut = & (Join-Path $Engine 'venv\Scripts\python.exe') (Join-Path $Engine 'utilityunelite_profile.py') --import-profile 2>&1 | ForEach-Object { "$_" }
$profExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($profExit -eq 0) { $profOut | Where-Object { $_ -match 'already|Imported' } | ForEach-Object { "   OK   $_" } }
else { "   WARN profile not imported (close RuneLite and re-run): $(($profOut | Select-Object -Last 2) -join ' ')" }

Write-Step '6. Jagex Launcher'
$jagex = Get-ItemProperty 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*', 'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*', 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*' -ErrorAction SilentlyContinue |
    Where-Object { $_.DisplayName -match '^Jagex Launcher' } | Select-Object -First 1
if ($jagex) { "   OK   Jagex Launcher $($jagex.DisplayVersion)" }
else {
    $inst = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Jagex Launcher Installer.exe'
    Invoke-WebRequest $JagexLauncherInstallerUrl -OutFile $inst -UseBasicParsing
    # open it on the logged-in desktop (this script runs in a background session)
    $action = New-ScheduledTaskAction -Execute $inst
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
    Register-ScheduledTask -TaskName 'WizardEyesJagexInstall' -Action $action -Principal $principal -Force | Out-Null
    Start-ScheduledTask -TaskName 'WizardEyesJagexInstall'
    Start-Sleep 5
    Unregister-ScheduledTask -TaskName 'WizardEyesJagexInstall' -Confirm:$false
    '   TODO Jagex Launcher installer opened on the VM desktop (also saved there)'
}

Write-Step 'Manual checklist (VirtualBox Manager > this VM > Show)'
'[ ] Jagex Launcher: finish the install (accept the admin prompt), log in'
'[ ] Jagex Launcher: choose RuneLite as the Old School client, start the game'
"[ ] RuneLite: import $Engine\data\runelite\naughty.properties as a profile and activate it"
'[ ] Game: Resizable - Modern layout, default zoom; keep the RuneLite sidebar closed'
"[ ] Check: $Engine\venv\Scripts\python.exe $Engine\scripts\diagnostics.py --run-time 30"
