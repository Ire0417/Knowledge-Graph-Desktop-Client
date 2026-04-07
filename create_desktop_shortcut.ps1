$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$sourceExe = Join-Path $root "dist\ZhishiExeDesktop.exe"
if (!(Test-Path $sourceExe)) {
    throw "Build output not found: dist\ZhishiExeDesktop.exe. Please run build_desktop.ps1 first."
}

$desktopPath = [Environment]::GetFolderPath("Desktop")
$appDir = Join-Path $desktopPath "ZhishiExeDesktop"
$targetExe = Join-Path $appDir "ZhishiExeDesktop.exe"
$shortcutPath = Join-Path $desktopPath "ZhishiExeDesktop.lnk"

New-Item -ItemType Directory -Path $appDir -Force | Out-Null
Copy-Item -Path $sourceExe -Destination $targetExe -Force

$wshShell = New-Object -ComObject WScript.Shell
$shortcut = $wshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetExe
$shortcut.WorkingDirectory = $appDir
$shortcut.IconLocation = $targetExe
$shortcut.Save()

Write-Host "Desktop app copied to: $targetExe"
Write-Host "Desktop shortcut created: $shortcutPath"
