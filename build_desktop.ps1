$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (!(Test-Path ".venv\Scripts\python.exe")) {
    throw "Python executable not found in .venv. Please create .venv and install dependencies first."
}

$pythonExe = ".venv\Scripts\python.exe"

function Install-RequirementsWithMirrorFallback {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe,
        [Parameter(Mandatory = $true)]
        [string]$RequirementsFile
    )

    # Prefer local mirrors first; fallback to official PyPI last.
    $indexUrls = @(
        "https://pypi.tuna.tsinghua.edu.cn/simple",
        "https://mirrors.aliyun.com/pypi/simple",
        "https://pypi.mirrors.ustc.edu.cn/simple",
        "https://repo.huaweicloud.com/repository/pypi/simple",
        "https://pypi.org/simple"
    )

    foreach ($indexUrl in $indexUrls) {
        $trustedHost = ([Uri]$indexUrl).Host
        Write-Host "Trying dependency source: $indexUrl"

        # Some Python builds print non-fatal messages to stderr; capture both streams and rely on exit code.
        $installOutput = & $PythonExe -m pip install `
            --disable-pip-version-check `
            --retries 1 `
            --timeout 10 `
            -r $RequirementsFile `
            -i $indexUrl `
            --trusted-host $trustedHost 2>&1

        $installOutput | ForEach-Object { Write-Host $_ }

        if ($LASTEXITCODE -eq 0) {
            Write-Host "Dependency install succeeded from: $indexUrl"
            return
        }

        Write-Warning "Source failed, trying next source..."
    }

    throw "All configured package indexes failed. Please check network or configure a reachable mirror."
}

Install-RequirementsWithMirrorFallback -PythonExe $pythonExe -RequirementsFile "requirements-desktop.txt"

# Clean stale one-folder output from previous builds to keep a single publish artifact.
$legacyOneFolderDist = Join-Path $root "dist\ZhishiExeDesktop"
if (Test-Path $legacyOneFolderDist) {
    Remove-Item -Path $legacyOneFolderDist -Recurse -Force
    Write-Host "Removed legacy one-folder output: $legacyOneFolderDist"
}

# Stop running app process and clear old one-file artifact to avoid WinError 5 during rebuild.
$runningApp = Get-Process -Name "ZhishiExeDesktop" -ErrorAction SilentlyContinue
if ($runningApp) {
    $runningApp | Stop-Process -Force
    Write-Host "Stopped running process: ZhishiExeDesktop"
}

$distExe = Join-Path $root "dist\ZhishiExeDesktop.exe"
if (Test-Path $distExe) {
    Remove-Item -Path $distExe -Force
    Write-Host "Removed previous artifact: $distExe"
}

$buildOutput = & ".venv\Scripts\pyinstaller.exe" `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --name ZhishiExeDesktop `
    --paths "backend" `
    --collect-submodules "app" `
    --collect-all "langchain" `
    --collect-all "langchain_core" `
    --collect-all "langchain_community" `
    --collect-all "langchain_openai" `
    --collect-all "langchain_text_splitters" `
    --collect-all "chromadb" `
    --collect-all "jieba" `
    --collect-all "openpyxl" `
    --collect-all "xlrd" `
    --collect-all "PIL" `
    --collect-all "sklearn" `
    --collect-all "docx" `
    --collect-all "pytesseract" `
    --collect-all "PyPDF2" `
    --collect-all "numpy" `
    --collect-all "pandas" `
    --hidden-import "flask" `
    --hidden-import "flask_cors" `
    --hidden-import "langchain" `
    --hidden-import "langchain.text_splitter" `
    --hidden-import "langchain_core" `
    --hidden-import "langchain_community" `
    --hidden-import "langchain_openai" `
    --hidden-import "langchain_text_splitters" `
    --hidden-import "chromadb" `
    --hidden-import "jieba" `
    --hidden-import "openpyxl" `
    --hidden-import "xlrd" `
    --hidden-import "PIL" `
    --hidden-import "sklearn" `
    --hidden-import "docx" `
    --hidden-import "pytesseract" `
    --hidden-import "PyPDF2" `
    --hidden-import "numpy" `
    --hidden-import "numpy._core._exceptions" `
    --hidden-import "pandas" `
    --add-data "backend;backend" `
    desktop_app.py 2>&1

$buildOutput | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

Write-Host "Build finished: dist\ZhishiExeDesktop.exe"
