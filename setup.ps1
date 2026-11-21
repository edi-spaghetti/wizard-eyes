<#
.SYNOPSIS
    Set up everything needed to run Wizard Eyes scripts on this machine.

.DESCRIPTION
    Safe to re-run: steps that are already done are skipped.

      1. Windows check
      2. Jagex Launcher and RuneLite installed
      3. Python 3.9 (offers a per-user install if missing; no admin needed)
      4. Virtual environment
      5. Pinned requirements + the engine (editable install)
      6. Tesseract OCR data (English)
      7. Remote asset config (Cloudflare R2)
      8. Asset connection check (+ optional full download)
      9. Display scaling at 100% (offers to fix)
     10. RuneLite profile (download + check the active profile)
     11. Smoke test + manual checklist

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1 -SyncAssets -Reconfigure
#>
[CmdletBinding()]
param(
    # Where to create the virtual environment (default: <repo>\venv).
    [string]$VenvPath = '',
    # Skip downloading Tesseract OCR data.
    [switch]$SkipOcr,
    # Ask for R2 settings again even if config/assets.local.json exists.
    [switch]$Reconfigure,
    # Download every asset now instead of on demand.
    [switch]$SyncAssets,
    # Never prompt; take the default answer for every question.
    [switch]$NonInteractive
)

$ErrorActionPreference = 'Stop'
$Repo = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
# (PowerShell 5.1 doesn't populate $PSScriptRoot for param defaults)
if (-not $VenvPath) { $VenvPath = Join-Path $Repo 'venv' }
$PythonVersion = '3.9'
$PythonInstallerUrl = 'https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe'
$TessdataUrl = 'https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata'
$JagexLauncherUrl = 'https://www.jagex.com/en-GB/launcher'
$RuneLiteUrl = 'https://runelite.net/'
$JagexLauncherInstallerUrl = 'https://cdn.jagex.com/Jagex%20Launcher%20Installer.exe'
$RuneLiteInstallerUrl = 'https://github.com/runelite/launcher/releases/latest/download/RuneLiteSetup.exe'
$ProfileAsset = 'runelite/naughty.properties'

$script:Results = [ordered]@{}

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

function Write-Step([string]$Text) { Write-Host "`n== $Text" -ForegroundColor Cyan }
function Write-Ok([string]$Text) { Write-Host "   OK   $Text" -ForegroundColor Green }
function Write-Warn([string]$Text) { Write-Host "   WARN $Text" -ForegroundColor Yellow }
function Write-Info([string]$Text) { Write-Host "        $Text" }

function Set-Result([string]$Step, [string]$Status, [string]$Detail = '') {
    $script:Results[$Step] = [pscustomobject]@{ Status = $Status; Detail = $Detail }
}

function Confirm-Choice([string]$Question, [bool]$Default = $false) {
    if ($NonInteractive) { return $Default }
    $hint = if ($Default) { '[Y/n]' } else { '[y/N]' }
    $answer = Read-Host "   $Question $hint"
    if ([string]::IsNullOrWhiteSpace($answer)) { return $Default }
    return $answer.Trim().ToLower().StartsWith('y')
}

# Simple (non-advanced) functions so arguments like -m pass straight through.
function Invoke-Native {
    $exe = $args[0]; $rest = @($args | Select-Object -Skip 1)
    & $exe @rest
    if ($LASTEXITCODE -ne 0) { throw "$exe $($rest -join ' ') failed (exit $LASTEXITCODE)" }
}

# Run a native command and capture stdout+stderr as text without PowerShell
# 5.1 turning stderr lines into terminating errors. Sets $script:LastExit.
function Invoke-Capture {
    # (@() so a single remaining argument isn't splatted character by character)
    $exe = $args[0]; $rest = @($args | Select-Object -Skip 1)
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $exe @rest 2>&1 | ForEach-Object { "$_" } | Out-String
        $script:LastExit = $LASTEXITCODE
    } finally { $ErrorActionPreference = $old }
    return $out
}

function Get-InstalledApp([string]$NamePattern) {
    $keys = @(
        'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*'
    )
    Get-ItemProperty $keys -ErrorAction SilentlyContinue |
        Where-Object { $_.DisplayName -match $NamePattern } |
        Select-Object -First 1
}

function Write-Utf8NoBom([string]$Path, [string]$Text) {
    [IO.File]::WriteAllText($Path, $Text, (New-Object Text.UTF8Encoding $false))
}

function Read-Secret([string]$Prompt) {
    $secure = Read-Host "   $Prompt" -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
}

# --------------------------------------------------------------------------
# display scaling (uses the undocumented DisplayConfig DPI calls that the
# Settings app uses; the same approach as tools like SetDPI)
# --------------------------------------------------------------------------

$DisplayScaleSource = @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public static class WizardDisplayScale {
    [StructLayout(LayoutKind.Sequential)]
    struct LUID { public uint Low; public int High; }

    [StructLayout(LayoutKind.Sequential)]
    struct Header { public int Type; public int Size; public LUID AdapterId; public uint Id; }

    [StructLayout(LayoutKind.Sequential)]
    struct GetScale { public Header H; public int Min; public int Cur; public int Max; }

    [StructLayout(LayoutKind.Sequential)]
    struct SetScale { public Header H; public int Scale; }

    [DllImport("user32.dll")] static extern int GetDisplayConfigBufferSizes(uint flags, out uint paths, out uint modes);
    [DllImport("user32.dll")] static extern int QueryDisplayConfig(uint flags, ref uint numPaths, byte[] paths, ref uint numModes, byte[] modes, IntPtr topology);
    [DllImport("user32.dll")] static extern int DisplayConfigGetDeviceInfo(ref GetScale info);
    [DllImport("user32.dll")] static extern int DisplayConfigSetDeviceInfo(ref SetScale info);

    const uint QDC_ONLY_ACTIVE_PATHS = 2;
    const int PATH_SIZE = 72, MODE_SIZE = 64;
    static readonly int[] Steps = { 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500 };

    public class Display {
        public uint Low; public int High; public uint Id;
        public int Current; public int Recommended; public int MinRel; public int CurRel;
    }

    public static List<Display> List() {
        uint np, nm;
        int err = GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, out np, out nm);
        if (err != 0) throw new Exception("GetDisplayConfigBufferSizes failed: " + err);
        byte[] paths = new byte[np * PATH_SIZE], modes = new byte[nm * MODE_SIZE];
        err = QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, ref np, paths, ref nm, modes, IntPtr.Zero);
        if (err != 0) throw new Exception("QueryDisplayConfig failed: " + err);

        var result = new List<Display>();
        var seen = new HashSet<string>();
        for (int i = 0; i < np; i++) {
            int o = i * PATH_SIZE;   // sourceInfo: LUID adapterId, uint id
            var d = new Display {
                Low = BitConverter.ToUInt32(paths, o),
                High = BitConverter.ToInt32(paths, o + 4),
                Id = BitConverter.ToUInt32(paths, o + 8)
            };
            if (!seen.Add(d.Low + ":" + d.High + ":" + d.Id)) continue;

            var g = new GetScale();
            g.H.Type = -3; g.H.Size = Marshal.SizeOf(typeof(GetScale));
            g.H.AdapterId.Low = d.Low; g.H.AdapterId.High = d.High; g.H.Id = d.Id;
            if (DisplayConfigGetDeviceInfo(ref g) != 0) continue;

            int rec = Math.Abs(g.Min);              // 100% is always the lowest step
            int cur = rec + g.Cur;
            d.MinRel = g.Min; d.CurRel = g.Cur;
            d.Recommended = Steps[Math.Min(rec, Steps.Length - 1)];
            d.Current = Steps[Math.Max(0, Math.Min(cur, Steps.Length - 1))];
            result.Add(d);
        }
        return result;
    }

    public static bool SetTo100(Display d) {
        var s = new SetScale();
        s.H.Type = -4; s.H.Size = Marshal.SizeOf(typeof(SetScale));
        s.H.AdapterId.Low = d.Low; s.H.AdapterId.High = d.High; s.H.Id = d.Id;
        s.Scale = d.MinRel;
        return DisplayConfigSetDeviceInfo(ref s) == 0;
    }
}
'@

# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------

function Step-Windows {
    Write-Step '1. Windows'
    if ([Environment]::OSVersion.Platform -ne 'Win32NT') {
        throw 'Wizard Eyes only runs on Windows.'
    }
    Write-Ok ([Environment]::OSVersion.VersionString)
    Set-Result 'Windows' 'ok'
}

function Find-JagexLauncher { Get-InstalledApp '^Jagex Launcher' }

function Find-RuneLite {
    $app = Get-InstalledApp '^RuneLite'
    if ($app) { return $app }
    if (Test-Path (Join-Path $env:LOCALAPPDATA 'RuneLite\RuneLite.exe')) {
        return [pscustomobject]@{ DisplayVersion = '' }
    }
}

# Download an official installer and run it, waiting for it to finish.
# The Jagex Launcher installer asks for admin (a UAC prompt to accept).
function Install-FromUrl([string]$Name, [string]$Url, [string]$FileName) {
    if ($NonInteractive) {
        Write-Warn "$Name not installed; skipping its installer (-NonInteractive)."
        return
    }
    $installer = Join-Path $env:TEMP $FileName
    Write-Info "Downloading $Name installer from $Url"
    Invoke-WebRequest $Url -OutFile $installer -UseBasicParsing
    Write-Info "Running the $Name installer; accept any admin prompt and follow it through."
    $p = Start-Process $installer -Wait -PassThru
    Write-Info "$Name installer finished (exit $($p.ExitCode))."
}

function Step-GameClients {
    Write-Step '2. Jagex Launcher and RuneLite'
    $missing = @()

    if (-not (Find-JagexLauncher)) {
        Write-Warn 'Jagex Launcher not found. It is required to log in to OSRS.'
        Install-FromUrl 'Jagex Launcher' $JagexLauncherInstallerUrl 'JagexLauncherInstaller.exe'
    }
    $jagex = Find-JagexLauncher
    if ($jagex) { Write-Ok "Jagex Launcher $($jagex.DisplayVersion)" }
    else { $missing += 'Jagex Launcher'; Write-Warn "Install it from $JagexLauncherUrl" }

    if (-not (Find-RuneLite)) {
        Write-Warn 'RuneLite not found.'
        Install-FromUrl 'RuneLite' $RuneLiteInstallerUrl 'RuneLiteSetup.exe'
    }
    $runelite = Find-RuneLite
    if ($runelite) {
        Write-Ok "RuneLite $($runelite.DisplayVersion)"
        Write-Info 'In the Jagex Launcher, select RuneLite as the Old School client.'
    } else { $missing += 'RuneLite'; Write-Warn "Install it from $RuneLiteUrl" }

    if ($missing) { Set-Result 'Game clients' 'warn' "missing: $($missing -join ', ')" }
    else { Set-Result 'Game clients' 'ok' }
}

function Find-Python39 {
    try {
        $exe = Invoke-Capture py "-$PythonVersion" -c 'import sys; print(sys.executable)'
        if ($script:LastExit -eq 0 -and $exe -and (Test-Path $exe.Trim())) { return $exe.Trim() }
    } catch { }
    $candidate = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python39\python.exe'
    if (Test-Path $candidate) { return $candidate }
    return $null
}

function Step-Python {
    Write-Step '3. Python 3.9'
    $python = Find-Python39
    if (-not $python) {
        Write-Warn 'Python 3.9 not found (the pinned tesserocr wheel needs it).'
        if (-not (Confirm-Choice 'Download and install Python 3.9.13 for this user only?' $true)) {
            throw 'Python 3.9 is required.'
        }
        $installer = Join-Path $env:TEMP 'python-3.9.13-amd64.exe'
        Write-Info "Downloading $PythonInstallerUrl"
        Invoke-WebRequest $PythonInstallerUrl -OutFile $installer -UseBasicParsing
        Write-Info 'Installing (per-user, no admin)...'
        $p = Start-Process $installer -Wait -PassThru -ArgumentList @(
            '/quiet', 'InstallAllUsers=0', 'PrependPath=0', 'Include_launcher=1',
            'InstallLauncherAllUsers=0', 'Include_test=0', 'Shortcuts=0')
        if ($p.ExitCode -ne 0) { throw "Python installer failed (exit $($p.ExitCode))" }
        $python = Find-Python39
        if (-not $python) { throw 'Python 3.9 installed but not found; open a new terminal and re-run.' }
    }
    $version = & $python -c 'import platform; print(platform.python_version())'
    Write-Ok "Python $version at $python"
    Set-Result 'Python 3.9' 'ok' $version
    return $python
}

function Step-Venv([string]$Python) {
    Write-Step '4. Virtual environment'
    $venvPython = Join-Path $VenvPath 'Scripts\python.exe'
    if (Test-Path $venvPython) {
        $v = & $venvPython -c 'import sys; print(str(sys.version_info[0]) + chr(46) + str(sys.version_info[1]))'
        if ($v -ne $PythonVersion) {
            throw "Existing venv at $VenvPath is Python $v, expected $PythonVersion. Delete it or pass -VenvPath."
        }
        Write-Ok "Using existing venv: $VenvPath"
    } else {
        Write-Info "Creating $VenvPath"
        Invoke-Native $Python -m venv $VenvPath
        Write-Ok "Created venv: $VenvPath"
    }
    Set-Result 'Virtual environment' 'ok' $VenvPath
    return $venvPython
}

function Step-Packages([string]$VenvPython) {
    Write-Step '5. Packages'
    Invoke-Native $VenvPython -m pip install --quiet --upgrade pip
    Write-Info 'Installing requirements.txt (first run takes a few minutes)...'
    Invoke-Native $VenvPython -m pip install --quiet -r (Join-Path $Repo 'requirements.txt')
    Invoke-Native $VenvPython -m pip install --quiet -e $Repo
    Write-Ok 'Requirements and wizard_eyes (editable) installed'
    Set-Result 'Packages' 'ok'
}

function Step-Ocr {
    Write-Step '6. Tesseract OCR data'
    $tessdata = Join-Path (Split-Path $Repo -Parent) 'tessdata'
    $eng = Join-Path $tessdata 'eng.traineddata'
    if ($SkipOcr) {
        Write-Warn 'Skipped (-SkipOcr). OCR features will be disabled.'
        Set-Result 'OCR data' 'skipped'
        return
    }
    if ((Test-Path $eng) -and (Get-Item $eng).Length -gt 1MB) {
        Write-Ok "Found $eng"
    } else {
        New-Item -ItemType Directory -Force $tessdata | Out-Null
        Write-Info "Downloading eng.traineddata to $tessdata"
        Invoke-WebRequest $TessdataUrl -OutFile $eng -UseBasicParsing
        Write-Ok "Downloaded $eng"
    }
    Set-Result 'OCR data' 'ok' $tessdata
}

function Step-AssetConfig {
    Write-Step '7. Remote asset config (Cloudflare R2)'
    $config = Join-Path $Repo 'config\assets.local.json'
    $fromEnv = $env:WIZARD_ASSETS_R2_ACCOUNT_ID -and $env:WIZARD_ASSETS_R2_ACCESS_KEY_ID -and $env:WIZARD_ASSETS_R2_SECRET_ACCESS_KEY

    if ((Test-Path $config) -and -not $Reconfigure) {
        Write-Ok "Using existing $config (pass -Reconfigure to change it)"
        Set-Result 'Asset config' 'ok' 'existing file'
        return
    }
    if ($fromEnv -and -not $Reconfigure) {
        Write-Ok 'Using WIZARD_ASSETS_* environment variables'
        Set-Result 'Asset config' 'ok' 'environment'
        return
    }
    if ($NonInteractive) {
        Write-Warn 'No asset config; engine will only use local files. Re-run interactively or set WIZARD_ASSETS_* variables.'
        Set-Result 'Asset config' 'warn' 'not configured'
        return
    }

    Write-Info 'Enter your R2 details. Use a READ-ONLY token for this machine.'
    $account = (Read-Host '   Account ID').Trim()
    $bucket = (Read-Host '   Bucket [wizard-eyes-assets]').Trim()
    if (-not $bucket) { $bucket = 'wizard-eyes-assets' }
    $keyId = (Read-Host '   Read-only Access Key ID').Trim()
    $secret = (Read-Secret 'Read-only Secret Access Key (hidden)').Trim()

    $data = [ordered]@{
        backend = 'r2'; local_dir = ''
        r2 = [ordered]@{ account_id = $account; bucket = $bucket; access_key_id = $keyId; secret_access_key = $secret }
    }
    if (Confirm-Choice 'Will this machine publish assets (needs a read & write token)?' $false) {
        $data.r2_publish = [ordered]@{
            access_key_id = (Read-Host '   Publish Access Key ID').Trim()
            secret_access_key = (Read-Secret 'Publish Secret Access Key (hidden)').Trim()
        }
    }
    Write-Utf8NoBom $config ($data | ConvertTo-Json -Depth 4)
    Write-Ok "Wrote $config (gitignored)"
    Set-Result 'Asset config' 'ok' 'written'
}

function Step-Assets([string]$VenvPython) {
    Write-Step '8. Asset connection'
    $assets = Join-Path $Repo 'utility\assets.py'
    $out = Invoke-Capture $VenvPython $assets --limit 0 status
    if ($script:LastExit -ne 0) {
        Write-Warn 'Could not reach remote assets:'
        Write-Info $out.Trim()
        Set-Result 'Assets' 'warn' 'status failed'
        return $false
    }
    $out.Trim().Split("`n") | ForEach-Object { Write-Info $_.TrimEnd() }
    if ($out -match 'No remote asset manifest') {
        Write-Warn 'Connected, but nothing has been published to the bucket yet.'
        Set-Result 'Assets' 'warn' 'empty bucket'
        return $false
    }
    if ($SyncAssets) {
        Write-Info 'Downloading all assets (-SyncAssets)...'
        Invoke-Native $VenvPython $assets sync
    }
    Write-Ok 'Remote assets reachable'
    Set-Result 'Assets' 'ok' $(if ($SyncAssets) { 'synced' } else { 'on demand' })
    return $true
}

function Step-Scaling {
    Write-Step '9. Display scaling'
    try {
        if (-not ('WizardDisplayScale' -as [type])) { Add-Type -TypeDefinition $DisplayScaleSource }
        $displays = [WizardDisplayScale]::List()
    } catch {
        Write-Warn "Could not read display scaling: $($_.Exception.Message)"
        Write-Info 'Check Settings > System > Display > Scale is 100% on the game monitor.'
        Set-Result 'Display scaling' 'warn' 'unknown'
        return
    }

    $wrong = @()
    $n = 0
    foreach ($d in $displays) {
        $n++
        $text = "Display $n`: $($d.Current)% (recommended $($d.Recommended)%)"
        if ($d.Current -eq 100) { Write-Ok $text } else { Write-Warn $text; $wrong += $d }
    }
    if (-not $wrong) { Set-Result 'Display scaling' 'ok' '100%'; return }

    Write-Info 'Templates are captured at 100% scaling; other values make the game image'
    Write-Info 'larger and templates stop matching. This changes the whole monitor.'
    if (Confirm-Choice 'Set these displays to 100% now?' $false) {
        $failed = 0
        foreach ($d in $wrong) { if (-not [WizardDisplayScale]::SetTo100($d)) { $failed++ } }
        $after = [WizardDisplayScale]::List() | Where-Object { $_.Current -ne 100 }
        if (-not $failed -and -not $after) {
            Write-Ok 'All displays now at 100%'
            Set-Result 'Display scaling' 'ok' 'fixed'
            return
        }
        Write-Warn 'Could not change scaling automatically.'
    }
    Write-Info 'Opening Display settings; set Scale to 100%.'
    if (-not $NonInteractive) { Start-Process 'ms-settings:display' }
    Set-Result 'Display scaling' 'warn' 'not 100%'
}

function Step-Profile([string]$VenvPython, [bool]$AssetsOk) {
    Write-Step '10. RuneLite profile'
    $checker = Join-Path $Repo 'utility\runelite_profile.py'
    $profileFile = Join-Path $Repo ("data\" + $ProfileAsset.Replace('/', '\'))

    if ($AssetsOk) {
        Invoke-Capture $VenvPython (Join-Path $Repo 'utility\assets.py') sync --prefix $ProfileAsset | Out-Null
    }

    while ($true) {
        $out = Invoke-Capture $VenvPython $checker
        $ok = $script:LastExit -eq 0
        $out.Trim().Split("`n") | ForEach-Object { Write-Info $_.TrimEnd() }
        if ($ok) {
            Write-Ok 'Active RuneLite profile has the settings the engine needs'
            Set-Result 'RuneLite profile' 'ok'
            return
        }
        Write-Warn 'The active RuneLite profile is missing settings the engine needs.'
        if (Test-Path $profileFile) {
            Write-Info 'Import the reference profile in RuneLite:'
            Write-Info '  Configuration (wrench) > Profiles > Import, choose:'
            Write-Info "  $profileFile"
            Write-Info '  then activate the imported profile.'
        } else {
            Write-Info 'Reference profile not available (remote assets not configured).'
        }
        if (-not (Confirm-Choice 'Re-check after importing?' $false)) {
            Set-Result 'RuneLite profile' 'warn' 'settings mismatch'
            return
        }
    }
}

function Step-SmokeTest([string]$VenvPython) {
    Write-Step '11. Smoke test'
    $tessdata = Join-Path (Split-Path $Repo -Parent) 'tessdata'
    $code = @"
import os, sys
import cv2, numpy, tesserocr, boto3
import wizard_eyes
from wizard_eyes.file_path_utils import get_root
assert os.path.normcase(get_root()) == os.path.normcase(r'$Repo'), get_root()
path = r'$tessdata'
if os.path.exists(os.path.join(path, 'eng.traineddata')):
    api = tesserocr.PyTessBaseAPI(path=path); api.End(); ocr = 'ok'
else:
    ocr = 'skipped'
print('python %s | opencv %s | numpy %s | %s | ocr %s' % (
    sys.version.split()[0], cv2.__version__, numpy.__version__,
    tesserocr.tesseract_version().splitlines()[0], ocr))
"@
    $out = Invoke-Capture $VenvPython -c $code
    if ($script:LastExit -ne 0) {
        Write-Warn 'Smoke test failed:'
        Write-Info $out.Trim()
        Set-Result 'Smoke test' 'FAIL'
        return
    }
    Write-Ok $out.Trim()
    Set-Result 'Smoke test' 'ok'
}

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

Write-Host 'Wizard Eyes setup' -ForegroundColor Cyan
Write-Host "Repo: $Repo"

Step-Windows
Step-GameClients
$python = Step-Python
$venvPython = Step-Venv $python
Step-Packages $venvPython
Step-Ocr
Step-AssetConfig
$assetsOk = Step-Assets $venvPython
Step-Scaling
Step-Profile $venvPython $assetsOk
Step-SmokeTest $venvPython

Write-Step 'Summary'
foreach ($k in $script:Results.Keys) {
    $r = $script:Results[$k]
    $colour = switch ($r.Status) { 'ok' { 'Green' } 'skipped' { 'Gray' } 'FAIL' { 'Red' } default { 'Yellow' } }
    Write-Host ('   {0,-20} {1,-8} {2}' -f $k, $r.Status, $r.Detail) -ForegroundColor $colour
}

Write-Step 'Manual checklist (in game)'
Write-Info '[ ] Log in through the Jagex Launcher with RuneLite as the client'
Write-Info '[ ] Game layout: Resizable - Modern'
Write-Info '[ ] Camera: fully top-down, facing north, default zoom'
Write-Info ''
Write-Info "Run scripts with: $venvPython <script.py>"

if ($script:Results.Values | Where-Object Status -eq 'FAIL') { exit 1 }
