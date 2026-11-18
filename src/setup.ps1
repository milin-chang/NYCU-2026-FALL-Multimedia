param([switch]$SkipPackages)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$uv = Join-Path $root 'metrics/.tools/uv/uv.exe'
$basePython = Join-Path $root 'metrics/.venv/Scripts/python.exe'
$venvPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$kair = Join-Path $PSScriptRoot 'upstream/KAIR'
$expectedCommit = 'fc1732f4a4514e42ce15e5b3a1e18c828af47a1e'
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.uv-cache'

if (-not (Test-Path -LiteralPath $uv)) {
    throw "uv is missing: $uv"
}
if (-not (Test-Path -LiteralPath $kair)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $kair) -Force | Out-Null
    & gh repo clone cszn/KAIR $kair -- --depth 1
    if ($LASTEXITCODE -ne 0) { throw 'Failed to clone KAIR.' }
    $clonedCommit = & git -C $kair rev-parse HEAD
    if ($clonedCommit.Trim() -ne $expectedCommit) {
        & git -C $kair fetch --depth 1 origin $expectedCommit
        if ($LASTEXITCODE -ne 0) { throw 'Failed to fetch the pinned KAIR commit.' }
        & git -C $kair checkout --detach $expectedCommit
        if ($LASTEXITCODE -ne 0) { throw 'Failed to check out the pinned KAIR commit.' }
    }
}
$commit = & git -C $kair rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $commit.Trim() -ne $expectedCommit) {
    throw "KAIR must be at $expectedCommit (found $commit)."
}
if (-not (Test-Path -LiteralPath $venvPython)) {
    $pythonOption = if (Test-Path -LiteralPath $basePython) { $basePython } else { '3.11' }
    & $uv venv (Join-Path $PSScriptRoot '.venv') --python $pythonOption
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the training environment.' }
}
if (-not $SkipPackages) {
    & $uv pip install --python $venvPython --torch-backend=cu128 `
        -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install training dependencies.' }
}
& $venvPython -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
if ($LASTEXITCODE -ne 0) { throw 'Failed to validate Python environment.' }
