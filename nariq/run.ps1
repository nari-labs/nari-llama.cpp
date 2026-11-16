# Run Qwen3.8-27B trellis on an NVIDIA GPU with one command (Windows x86-64, PowerShell 5.1 or 7): the Windows twin
# of run.sh. Downloads the llama.cpp trellis fork's binaries and CUDA runtime, the GGUF and the vision projector once,
# checks each file's sha256, and starts llama-server with an OpenAI-compatible API and a chat page at
# http://127.0.0.1:8080.
#
#   powershell -ExecutionPolicy Bypass -File run.ps1                 # the 3.0 file, with vision, the longest context
#   powershell -ExecutionPolicy Bypass -File run.ps1 -Budget 2.5            # or 2.0; 3bit / 2.5bit / 2bit work too
#   powershell -ExecutionPolicy Bypass -File run.ps1 -NoVision -Extra "-np","4","-c","131072"
#
# Environment: TRELLIS_HOME (download folder, %LOCALAPPDATA%\nari-llama.cpp); TRELLIS_RELEASE_URL, TRELLIS_MODEL_URL
# and TRELLIS_TAG (mirrors and tests). Needs Windows 10 1803 or newer (curl.exe) and an NVIDIA driver of 572 or newer.
# Without a GPU it runs on the CPU, slowly.
param(
    [ValidateSet("3.0", "2.5", "2.0", "3bit", "2.5bit", "2bit")][string]$Budget = "3.0",
    [int]$Port = 8080,
    [string]$ListenHost = "127.0.0.1",
    [switch]$NoVision,
    [switch]$DownloadOnly,
    [string[]]$Extra = @()
)
$ErrorActionPreference = "Stop"
$Aliases = @{ "3bit" = "3.0"; "2.5bit" = "2.5"; "2bit" = "2.0" }
if ($Aliases.ContainsKey($Budget)) { $Budget = $Aliases[$Budget] }

$Tag = if ($env:TRELLIS_TAG) { $env:TRELLIS_TAG } else { "b12431-nari2" }  # the release tag
$ReleaseUrl = if ($env:TRELLIS_RELEASE_URL) { $env:TRELLIS_RELEASE_URL } else { "https://github.com/nari-labs/nari-llama.cpp/releases/download/$Tag" }
$ModelUrl = if ($env:TRELLIS_MODEL_URL) { $env:TRELLIS_MODEL_URL } else { "https://huggingface.co/nari-labs/Qwen3.8-27B-NariQ-GGUF/resolve/main" }
$HomeDir = if ($env:TRELLIS_HOME) { $env:TRELLIS_HOME } else { Join-Path $env:LOCALAPPDATA "nari-llama.cpp" }
$Cuda = "12.8"

# file and sha256 per budget; the layout of these files is frozen
$Files = @{
    "3.0" = @("Qwen3.8-27B-NariQ-3bit.gguf", "1eaad934aa05579b8eba47ed9b26450392f9875fc2354efe8cfa8b29be8f4419")
    "2.5" = @("Qwen3.8-27B-NariQ-2.5bit.gguf", "2445372e6de20620b940fbfab53aa718af9ba741f202da30cd2f02b5df78d05c")
    "2.0" = @("Qwen3.8-27B-NariQ-2bit.gguf", "3a9e4955329dba8ba0be5509a9552220e5ccdae2f66bcc72c21c4580cd49bbb2")
}
$Mmproj = @("Qwen3.8-27B-NariQ-mmproj-BF16.gguf", "8832425f593068d3278e08eda4c1591072c7a2d8dc90f21e4f86394e4b2eb002")

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$gpu = $null
if ($smi) {
    try {
        $gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -ne 0) { $gpu = $null }
    } catch {
        # Windows PowerShell 5.1 can turn native stderr into a terminating error.
        $gpu = $null
    }
}
if ($gpu -and $gpu -match '^.+,\s*(\d+)\.\d+.*,') {
    Write-Host "GPU: $gpu"
    $driver = [int]$Matches[1]
    if ($driver -lt 572) { throw "NVIDIA driver $driver is too old for CUDA $Cuda; install 572 or newer" }
} else {
    Write-Warning "no NVIDIA GPU found; llama-server will run on the CPU, which is slow for a 27B model"
}

function Fetch([string]$Url, [string]$Path, [string]$Sha) {
    if ((Test-Path $Path) -and (Test-Path "$Path.ok")) { return }
    New-Item -ItemType Directory -Force -Path (Split-Path $Path) | Out-Null
    Write-Host "downloading $(Split-Path $Path -Leaf)"
    & curl.exe -fL -# --retry 5 --retry-delay 3 -C - -o "$Path.part" $Url
    if ($LASTEXITCODE -ne 0) { throw "download of $Url failed" }
    Write-Host "checking $(Split-Path $Path -Leaf)"
    if ((Get-FileHash "$Path.part" -Algorithm SHA256).Hash.ToLower() -ne $Sha) {
        Remove-Item "$Path.part"
        throw "$(Split-Path $Path -Leaf) does not match its sha256; deleted, run again to download it again"
    }
    Move-Item -Force "$Path.part" $Path
    New-Item -ItemType File -Force -Path "$Path.ok" | Out-Null
}

$file, $sha = $Files[$Budget]
Fetch "$ModelUrl/$file" (Join-Path $HomeDir "models\$file") $sha
if (-not $NoVision) { Fetch "$ModelUrl/$($Mmproj[0])" (Join-Path $HomeDir "models\$($Mmproj[0])") $Mmproj[1] }

$bin = Join-Path $HomeDir "bin\$Tag"
if (-not (Test-Path (Join-Path $bin "llama-server.exe"))) {
    $downloads = Join-Path $HomeDir "downloads"
    New-Item -ItemType Directory -Force -Path $downloads | Out-Null
    $sums = Join-Path $downloads "SHA256SUMS-$Tag"
    & curl.exe -fsSL --retry 5 -o $sums "$ReleaseUrl/SHA256SUMS"
    if ($LASTEXITCODE -ne 0) { throw "download of $ReleaseUrl/SHA256SUMS failed" }
    $part = "$bin.part"
    if (Test-Path $part) { Remove-Item -Recurse -Force $part }
    foreach ($archive in @("nari-llama.cpp-$Tag-bin-win-cuda-$Cuda-x64", "cudart-nari-llama.cpp-$Tag-win-cuda-$Cuda-x64")) {
        $line = Get-Content $sums | Where-Object { $_ -match "^([0-9a-f]{64})  $([regex]::Escape($archive)).zip$" }
        if (-not $line) { throw "$archive.zip is not in the release's SHA256SUMS" }
        Fetch "$ReleaseUrl/$archive.zip" (Join-Path $downloads "$archive.zip") $Matches[1]
        Expand-Archive -Force (Join-Path $downloads "$archive.zip") $part
    }
    if (Test-Path $bin) { Remove-Item -Recurse -Force $bin }
    Move-Item $part $bin
}
if ($DownloadOnly) { exit 0 }

$serverArgs = @("-m", (Join-Path $HomeDir "models\$file"))
# the projector is loaded after the context is fitted to free memory: keep room for it and its image batch
if (-not $NoVision) { $serverArgs += @("--mmproj", (Join-Path $HomeDir "models\$($Mmproj[0])"), "--fit-target", "2560") }
Write-Host "starting llama-server ($Budget) on http://${ListenHost}:$Port"
& (Join-Path $bin "llama-server.exe") @serverArgs --host $ListenHost --port $Port @Extra
exit $LASTEXITCODE
