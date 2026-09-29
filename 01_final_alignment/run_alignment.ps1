# Generate alignment metric A for each model/dataset pair.
# Open PowerShell in this folder, then run: .\run_alignment.ps1
# Supports resume: existing A_*.json files are reused when config matches.

$ErrorActionPreference = "Stop"
$seeds5 = "42","2024","3407","8888","12345"

function Run-Step {
    param(
        [string]$Name,
        [string[]]$ArgsList
    )

    Write-Host $Name -ForegroundColor Cyan
    & python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed with exit code $LASTEXITCODE while running: $Name"
    }
}

$datasets = @(
    @{name="ml1m";   file="ml-1m.txt";      dev="cpu";  pca="100"},
    @{name="ml10m";  file="ml-10m.txt";     dev="cuda"; pca="64"},
    @{name="amazon"; file="AmazonBook.txt"; dev="cuda"; pca="64"}
)

foreach ($ds in $datasets) {
    foreach ($model in @("mf","fpmc","transformer")) {
        $name = "=== A: $model on $($ds.name) ==="
        $maxlen = if ($model -eq "transformer") { "50" } else { "200" }
        $batch = if ($model -eq "transformer") { "64" } else { "4096" }
        $argsList = @(
            "sirm_experiments.py", "--model", $model, "--data", $ds.file,
            "--variants", "base", "--alignment", "--device", $ds.dev,
            "--eval-every", "10", "--pca-dim", $ds.pca, "--maxlen", $maxlen,
            "--batch", $batch, "--seeds"
        ) + $seeds5 + @("--out", "A_$($model)_$($ds.name).json")
        Run-Step $name $argsList
    }
}

Write-Host ""
Write-Host "DONE. Send these 9 files: A_*.json" -ForegroundColor Green
