# AUTHOR v0.9.3 - generate FloorPlan1Discovery_physics and build the Linux player.
# Run in Windows PowerShell AFTER copying the v0.9.3 source files into the Windows Unity project.
$Unity48   = "C:\Program Files\Unity\Hub\Editor\2020.3.48f1\Editor\Unity.exe"
$Project48 = "C:\Users\davei\a2h\unity-2020.3.48"
$OutDir    = "C:\Users\davei\a2h\builds\thor-discovery-Linux64"
$GenLog    = "C:\Users\davei\a2h\discovery-world-generate.log"
$BuildLog  = "C:\Users\davei\a2h\thor-discovery-Linux64.log"

Get-CimInstance Win32_Process -Filter "Name='Unity.exe'" |
  Where-Object { $_.CommandLine -like "*unity-2020.3.48*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep -Seconds 2
Remove-Item "$Project48\Temp\UnityLockfile" -Force -ErrorAction SilentlyContinue

$GenArgs=@(
  "-quit","-batchmode","-projectPath","`"$Project48`"",
  "-executeMethod","DiscoveryWorldBuilder.BuildFloorPlan1Discovery",
  "-logFile","`"$GenLog`""
)
$Gen=Start-Process -FilePath $Unity48 -ArgumentList $GenArgs -Wait -PassThru
Write-Host "Discovery scene generation exit code: $($Gen.ExitCode)"
if ($Gen.ExitCode -ne 0) { Get-Content $GenLog -Tail 200; exit $Gen.ExitCode }

Remove-Item $OutDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $OutDir | Out-Null
$env:UNITY_BUILD_NAME="$OutDir\thor-discovery-Linux64"
$env:BUILD_SCENES="FloorPlan1Discovery_physics"
$BuildArgs=@(
  "-quit","-batchmode","-projectPath","`"$Project48`"",
  "-buildTarget","Linux64","-executeMethod","Build.Linux64",
  "-logFile","`"$BuildLog`""
)
$Build=Start-Process -FilePath $Unity48 -ArgumentList $BuildArgs -Wait -PassThru
Write-Host "Discovery player build exit code: $($Build.ExitCode)"
Write-Host "Executable:" (Test-Path "$OutDir\thor-discovery-Linux64")
Write-Host "Scene generated:" (Test-Path "$Project48\Assets\Scenes\FloorPlan1Discovery_physics.unity")
if ($Build.ExitCode -ne 0) { Get-Content $BuildLog -Tail 250 }
