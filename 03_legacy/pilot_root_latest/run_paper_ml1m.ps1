# Run all core ML-1M experiments (Group 1 + Group 2) for the paper.
# Open PowerShell in the pilot folder, then run: .\run_paper_ml1m.ps1
# Resume-friendly: if interrupted, run this script again.

$ErrorActionPreference = "Stop"
$seeds10 = "42","2024","3407","8888","12345","111","222","333","444","555"
$seeds5  = "42","2024","3407","8888","12345"
$outDir = "paper_ml1m_results"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "=== [1/5] MF (base+SIRM+) ===" -ForegroundColor Cyan
python sirm_experiments.py --model mf   --device cuda --seeds $seeds10 --out "$outDir/res_mf_ml1m.json"

Write-Host "=== [2/5] FPMC (base+SIRM+) ===" -ForegroundColor Cyan
python sirm_experiments.py --model fpmc --device cuda --seeds $seeds10 --out "$outDir/res_fpmc_ml1m.json"

Write-Host "=== [3/5] Transformer (base+SIRM+) - slowest ===" -ForegroundColor Cyan
python sirm_experiments.py --model transformer --device cuda --batch 32 --seeds $seeds5 --out "$outDir/res_transformer_ml1m.json"

Write-Host "=== [4/5] MF ablation: hard assignment ===" -ForegroundColor Cyan
python sirm_experiments.py --model mf --ablation hard  --device cuda --seeds $seeds10 --out "$outDir/res_mf_hard_ml1m.json"

Write-Host "=== [5/5] MF ablation: no PCA ===" -ForegroundColor Cyan
python sirm_experiments.py --model mf --ablation nopca --device cuda --seeds $seeds10 --out "$outDir/res_mf_nopca_ml1m.json"

Write-Host ""
Write-Host "DONE. Send the 5 files in folder: paper_ml1m_results" -ForegroundColor Green
