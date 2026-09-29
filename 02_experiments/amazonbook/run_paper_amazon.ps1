# Chạy thực nghiệm AmazonBook — miền thứ ba (sách), thưa hơn MovieLens nhiều.
# Mở PowerShell trong thư mục pilot rồi chạy:  .\run_paper_amazon.ps1
# Có chạy-tiếp: bị ngắt thì chạy lại là tiếp tục.
#
# Vì sao chạy bộ này: nó KIỂM CHỨNG dự đoán rút ra từ MovieLens —
#   ML-1M (165 item/user) -> MF +3.8%
#   ML-10M (109 item/user) -> MF +28.8%
#   AmazonBook (9.2 item/user, thưa hơn nhiều) -> MF phải hưởng lợi CÒN NHIỀU HƠN.
# Nếu đúng, đây là bằng chứng cơ chế "càng thưa prior càng giúp", không chỉ là tương quan.

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

Write-Host "=== [1/3] MF trên AmazonBook (quan trọng nhất) ===" -ForegroundColor Cyan
$mfArgs = @("sirm_experiments.py", "--model", "mf", "--data", "AmazonBook.txt", "--device", "cuda", "--eval-every", "10", "--pca-dim", "64", "--seeds") + $seeds5 + @("--out", "res_mf_amazon.json")
Run-Step "=== [1/3] MF AmazonBook ===" $mfArgs

Write-Host "=== [2/3] FPMC trên AmazonBook ===" -ForegroundColor Cyan
$fpmcArgs = @("sirm_experiments.py", "--model", "fpmc", "--data", "AmazonBook.txt", "--device", "cuda", "--eval-every", "10", "--pca-dim", "64", "--seeds") + $seeds5 + @("--out", "res_fpmc_amazon.json")
Run-Step "=== [2/3] FPMC AmazonBook ===" $fpmcArgs

Write-Host "=== [3/3] Transformer trên AmazonBook ===" -ForegroundColor Cyan
$transformerArgs = @("sirm_experiments.py", "--model", "transformer", "--data", "AmazonBook.txt", "--device", "cuda", "--eval-every", "10", "--pca-dim", "64", "--maxlen", "50", "--batch", "16", "--seeds") + $seeds5 + @("--out", "res_transformer_amazon.json")
Run-Step "=== [3/3] Transformer AmazonBook (batch 16) ===" $transformerArgs

Write-Host ""
Write-Host "XONG. Gửi lại 3 file: res_mf_amazon.json, res_fpmc_amazon.json, res_transformer_amazon.json" -ForegroundColor Green
