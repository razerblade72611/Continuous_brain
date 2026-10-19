param(
  [Parameter(Mandatory=$true)]
  [string]$LlamaServer,
  [Parameter(Mandatory=$true)]
  [string]$Model,
  [string]$Mmproj = "",
  [string]$Alias = "qwen2-vl-2b",
  [int]$Port = 8080,
  [int]$Context = 16384,
  [int]$GpuLayers = 999,
  [int]$Threads = 8,
  [int]$ImageMaxTokens = 1024
)

if (!(Test-Path $LlamaServer)) { throw "llama-server.exe not found: $LlamaServer" }
if (!(Test-Path $Model)) { throw "Model not found: $Model" }
if ($Mmproj -and !(Test-Path $Mmproj)) { throw "mmproj file not found: $Mmproj" }

$argsList = @(
  "--model", $Model,
  "--alias", $Alias,
  "--host", "0.0.0.0",
  "--port", "$Port",
  "-c", "$Context",
  "-ngl", "$GpuLayers",
  "-t", "$Threads"
)

if ($Mmproj) {
  $argsList += @("--mmproj", $Mmproj, "--image-max-tokens", "$ImageMaxTokens")
}

Write-Host "Starting llama.cpp server"
Write-Host "  Server : $LlamaServer"
Write-Host "  Model  : $Model"
Write-Host "  Alias  : $Alias"
if ($Mmproj) { Write-Host "  mmproj : $Mmproj" }
Write-Host "  URL    : http://0.0.0.0:$Port"
Write-Host "  GPU layers: $GpuLayers"

& $LlamaServer @argsList
