<#
Runs in a template VM whose sysprep did not shut it down (new-bot-vm.ps1 -Template, stage 9T): what
sysprep said. An app installed for one user but not provisioned for all is the usual reason; with
-Fix those apps are removed (elevated), so a new sysprep run can work.
#>
param([switch]$Fix)
$sp = "$env:SystemRoot\System32\Sysprep\Panther"
'== sysprep errors'
Get-Content "$sp\setuperr.log" -EA 0 | Select-Object -Last 15
$pkgs = @(Get-Content "$sp\setupact.log" -EA 0 | Select-String 'Package (\S+) was installed for a user, but not provisioned for all users' |
    ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -Unique)
foreach ($pk in $pkgs) {
    "app installed for a user only: $pk"
    if ($Fix) { Get-AppxPackage -AllUsers | Where-Object { $_.PackageFullName -eq $pk } | Remove-AppxPackage -AllUsers -EA 0; "  removed" }
}
if ($pkgs) { 'APPX-PROBLEM' } else { 'NO-APPX-PROBLEM' }
