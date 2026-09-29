# Run ML-10M experiments for the paper.
# Open PowerShell in this folder, then run: .\run_paper_ml10m.ps1
# Uses GPU (CUDA). If interrupted, run again to continue/recreate outputs.
# ML-10M is much heavier than ML-1M, so run overnight, keep power plugged in,
# and disable Sleep.

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

$mfArgs = @("sirm_experiments.py", "--model", "mf", "--data", "ml-10m.txt", "--device", "cuda", "--eval-every", "10", "--pca-dim", "64", "--seeds") + $seeds5 + @("--out", "res_mf_ml10m.json")
Run-Step "=== [1/3] MF on ML-10M ===" $mfArgs

$fpmcArgs = @("sirm_experiments.py", "--model", "fpmc", "--data", "ml-10m.txt", "--device", "cuda", "--eval-every", "10", "--pca-dim", "64", "--seeds") + $seeds5 + @("--out", "res_fpmc_ml10m.json")
Run-Step "=== [2/3] FPMC on ML-10M ===" $fpmcArgs

# Transformer self-attention has O(batch * maxlen^2) memory use.  The default
# batch (4096) is suitable for MF/FPMC but exceeds a 4 GB GPU for ML-10M.
$transformerArgs = @("sirm_experiments.py", "--model", "transformer", "--data", "ml-10m.txt", "--device", "cuda", "--eval-every", "10", "--pca-dim", "64", "--batch", "32", "--seeds") + $seeds5 + @("--out", "res_transformer_ml10m.json")
Run-Step "=== [3/3] Transformer on ML-10M, heaviest, 5 seeds ===" $transformerArgs

Write-Host ""
Write-Host "DONE. Send these 3 files: res_mf_ml10m.json, res_fpmc_ml10m.json, res_transformer_ml10m.json" -ForegroundColor Green
