<#
Put the committed code (HEAD) of the engine and the farm's repo into bot VMs,
over what's there (data / venv stay). Bots keep running their old code until
they're restarted (bot-control.ps1 -Action restart, or the farm's page).

    deploy-vm.ps1 [-Name wizard-eyes-vm1] [-Farm <folder>]

Without -Name: every running VM of the farm. Commit first: only HEAD goes.
farm.json "deploy_paths" limits what's copied of the farm's repo (default: all).
#>
param([string]$Name, [string]$Farm)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot '..\lib\BotFarm.psm1') -Force
$fm = Get-Farm $Farm

$vms = if ($Name) { @($Name) } else { @($fm.Roles.Values | ForEach-Object Vm | Select-Object -Unique | Where-Object { Test-VmRunning $_ }) }
if (-not $vms) { throw 'No running VMs to deploy to' }
$repos = @(
    @{ Dir = $fm.Engine; Paths = @('python', 'config', 'scripts', 'utility', 'botfarm') },
    @{ Dir = $fm.Repo; Paths = @($fm.Def.deploy_paths | Where-Object { $_ }) }
)
$work = Join-Path $env:TEMP "botfarm-deploy-$PID"
New-Item -ItemType Directory -Force $work | Out-Null
try {
    foreach ($r in $repos) {
        $leaf = Split-Path $r.Dir -Leaf
        $r.Zip = Join-Path $work "$leaf.zip"
        $r.Leaf = $leaf
        [string[]]$paths = $r.Paths
        git -C $r.Dir archive --format=zip -o $r.Zip HEAD @paths
        if ($LASTEXITCODE -ne 0) { throw "git archive of $leaf failed" }
        $r.Head = (git -C $r.Dir rev-parse --short HEAD)
    }
    Use-VmPassword {
        foreach ($vm in $vms) {
            foreach ($r in $repos) {
                Copy-ToGuest $vm $r.Zip "$GuestDrop\$($r.Leaf).zip"
                $dest = "$(Get-GuestRoot $vm)\$($r.Leaf)"
                $count = Invoke-Guest $vm ("Expand-Archive -Force $GuestDrop\$($r.Leaf).zip $dest; " +
                    "(Get-ChildItem $dest -Recurse -Filter *.py | ? FullName -notmatch 'venv').Count") 180
                "$vm  $($r.Leaf) at $($r.Head): $(@($count)[-1]) py files"
            }
        }
    }
} finally { Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue }
