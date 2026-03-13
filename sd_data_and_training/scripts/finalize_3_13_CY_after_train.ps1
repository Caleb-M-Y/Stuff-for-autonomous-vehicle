param(
    [string]$RunName = "3-13-CY",
    [int]$PollSeconds = 120,
    [int]$MaxHours = 36,
    [string]$PythonExe = "C:/Users/hogwi/OneDrive/VSCode GitHub Repos/Stuff-for-autonomous-vehicle/.venv/Scripts/python.exe"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$runDir = Join-Path $repoRoot "sd_data_and_training/runs/$RunName"
$bestPath = Join-Path $runDir "weights/best.pt"
$exportDir = Join-Path $repoRoot "sd_data_and_training/exports"
$namedPt = Join-Path $exportDir "3-13-CY.pt"
$namedOnnx = Join-Path $exportDir "3-13-CY.onnx"
$configPath = Join-Path $repoRoot "sd_data_and_training/config.yaml"
$exportScript = Join-Path $repoRoot "sd_data_and_training/scripts/export_for_hailo.py"
$logPath = Join-Path $exportDir "3-13-CY-finalize.log"
$statusPath = Join-Path $exportDir "3-13-CY-finalize.status"

New-Item -ItemType Directory -Path $exportDir -Force | Out-Null

function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $Message"
    Add-Content -Path $logPath -Value $line
}

function Get-TrainProcesses {
    @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object {
        $_.CommandLine -like '*sd_data_and_training/scripts/train_yolo.py*' -and
        $_.CommandLine -like "*--run-name $RunName*"
    })
}

try {
    "RUNNING" | Set-Content -Path $statusPath
    Write-Log "Finalizer started for run $RunName"

    $deadline = (Get-Date).AddHours($MaxHours)
    while ((Get-Date) -lt $deadline) {
        $procs = Get-TrainProcesses
        if ($procs.Count -eq 0) {
            Write-Log "No matching train_yolo.py process found. Proceeding to finalize."
            break
        }

        $pidList = ($procs | ForEach-Object { $_.ProcessId }) -join ","
        Write-Log "Training still running. PIDs=$pidList"
        Start-Sleep -Seconds $PollSeconds
    }

    $stillRunning = Get-TrainProcesses
    if ($stillRunning.Count -gt 0) {
        Write-Log "Timeout waiting for training completion."
        "TIMEOUT" | Set-Content -Path $statusPath
        exit 5
    }

    if (-not (Test-Path $bestPath)) {
        Write-Log "Missing final best checkpoint at $bestPath"
        "MISSING_BEST" | Set-Content -Path $statusPath
        exit 2
    }

    Copy-Item -Path $bestPath -Destination $namedPt -Force
    Write-Log "Copied final best checkpoint to $namedPt"

    & $PythonExe $exportScript --config $configPath --weights $namedPt --out-dir $exportDir
    if (-not (Test-Path $namedOnnx)) {
        Write-Log "ONNX not found after export at $namedOnnx"
        "ONNX_MISSING" | Set-Content -Path $statusPath
        exit 4
    }

    Write-Log "Final export complete: $namedOnnx"
    "SUCCESS" | Set-Content -Path $statusPath
    exit 0
}
catch {
    Write-Log "Unhandled error: $($_.Exception.Message)"
    "ERROR" | Set-Content -Path $statusPath
    exit 1
}
