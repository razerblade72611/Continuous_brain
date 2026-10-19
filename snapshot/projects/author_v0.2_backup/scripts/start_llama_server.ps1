param(
  [Parameter(Mandatory=$true)]
  [string]$LlamaServer,
  [Parameter(Mandatory=$true)]
  [string]$Model,
  [string]$Mmproj = "",
  [int]$Port = 8080,
  [int]$Context = 4096,
  [int]$GpuLayers = 999,
  [int]$Threads = 8
)

if (!(Test-Path $LlamaServer)) { throw "llama-server.exe not found: $LlamaServer" }
if (!(Test-Path $Model)) { throw "Model not found: $Model" }
if ($Mmproj -and !(Test-Path $Mmproj)) { throw "mmproj file not found: $Mmproj" }

$argsList = @(
  "--model", $Model,
  "--host", "0.0.0.0",
  "--port", "$Port",
  "-c", "$Context",
  "-ngl", "$GpuLayers",
  "-t", "$Threads"
)

if ($Mmproj) {
  $argsList += @("--mmproj", $Mmproj)
}

Write-Host "Starting llama.cpp server"
Write-Host "  Server : $LlamaServer"
Write-Host "  Model  : $Model"
if ($Mmproj) { Write-Host "  mmproj : $Mmproj" }
Write-Host "  URL    : http://0.0.0.0:$Port"
Write-Host "  GPU layers: $GpuLayers"

& $LlamaServer @argsList
