# Bản kê thực nghiệm cho bài báo (chạy trên `sirm_experiments.py`)

Một script duy nhất, một giao thức duy nhất cho mọi mô hình → số liệu so sánh được
với nhau. Đã khử ngẫu nhiên (đơn luồng + cờ tất định), có phân tích đuôi dài, có
chạy‑tiếp sau gián đoạn.

## Chuẩn bị (1 lần)

```powershell
cd đường\dẫn\tới\pilot
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install scikit-learn numpy
python -c "import torch; print(torch.cuda.is_available())"   # phải True
```

10 seed dùng chung: `42 2024 3407 8888 12345 111 222 333 444 555`

**Về thiết bị:** `--device cpu` cho kết quả **tái lập bit‑to‑bit** (KMeans/MF/FPMC nhẹ, chạy CPU đủ nhanh). Transformer nặng → để `--device cuda`. Muốn kiểm tra GPU có tất định không: chạy 2 lần cùng seed, HR@10 phải trùng.

---

## NHÓM 1 — Cốt lõi: phổ mô hình (ML-1M) ⭐ bắt buộc

Trả lời luận điểm trung tâm: *prior thừa với mô hình tự học co‑occurrence, có ích với mô hình không học được*.

```powershell
python sirm_experiments.py --model mf          --device cpu  --seeds 42 2024 3407 8888 12345 111 222 333 444 555 --out res_mf_ml1m.json
python sirm_experiments.py --model fpmc        --device cpu  --seeds 42 2024 3407 8888 12345 111 222 333 444 555 --out res_fpmc_ml1m.json
python sirm_experiments.py --model transformer --device cuda --seeds 42 2024 3407 8888 12345               --out res_transformer_ml1m.json
```

Kỳ vọng: mức cải thiện của SIRM+ giảm dần **MF → FPMC → Transformer**. Nếu đúng, đây là bảng chính của bài.
Thời gian ước tính (RTX 3050 Ti): MF ~30 phút, FPMC ~30 phút, Transformer ~2,5 giờ.

## NHÓM 2 — Ablation trên mô hình thắng (MF, ML-1M) ⭐ bắt buộc

```powershell
python sirm_experiments.py --model mf --ablation hard  --device cpu --seeds 42 2024 3407 8888 12345 111 222 333 444 555 --out res_mf_hard_ml1m.json
python sirm_experiments.py --model mf --ablation nopca --device cpu --seeds 42 2024 3407 8888 12345 111 222 333 444 555 --out res_mf_nopca_ml1m.json
```

- `hard`: gán cứng thay vì mềm → kiểm chứng phát hiện "soft sụp về phân bố đều". So với NHÓM 1 (soft).
- `nopca`: bỏ PCA, dùng thẳng ma trận đồng xuất hiện → kiểm chứng vai trò khử nhiễu của PCA.

## NHÓM 3 — Dataset thứ hai (ML-10M) — nên có để tổng quát hoá

> Cần có `ml-10m.txt` trong thư mục (chép từ `KG-LTSR-SIRM/data/`). Lần đầu dựng ma trận đồng xuất hiện ML-10M mất vài phút (sau đó được cache).

```powershell
python sirm_experiments.py --model mf          --data ml-10m.txt --device cpu  --seeds 42 2024 3407 8888 12345 --pca-dim 64 --out res_mf_ml10m.json
python sirm_experiments.py --model fpmc        --data ml-10m.txt --device cpu  --seeds 42 2024 3407 8888 12345 --pca-dim 64 --out res_fpmc_ml10m.json
python sirm_experiments.py --model transformer --data ml-10m.txt --device cuda --seeds 42 2024 3407           --pca-dim 64 --out res_transformer_ml10m.json
```

## NHÓM 4 — Độ nhạy siêu tham số (MF, ML-1M) — tùy chọn, làm bảng phụ lục

```powershell
# gamma khởi tạo: chứng minh gamma nhỏ làm hỏng (soft sụp về đều)
foreach ($g in 1,10,100,1000) { python sirm_experiments.py --model mf --variants sirm --gamma-init $g --device cpu --seeds 42 2024 3407 --out res_mf_gamma$g.json }
# số prototype K
foreach ($k in 10,15,30,50) { python sirm_experiments.py --model mf --variants sirm --clusters $k --device cpu --seeds 42 2024 3407 --out res_mf_K$k.json }
# số chiều PCA d_p
foreach ($d in 50,100,200,500) { python sirm_experiments.py --model mf --variants sirm --pca-dim $d --device cpu --seeds 42 2024 3407 --out res_mf_dp$d.json }
```

---

## Sau khi chạy

Gửi lại tất cả file `res_*.json`. Mỗi file đã tự in bảng tổng thể + bảng đuôi dài;
tôi sẽ tổng hợp thành các bảng hoàn chỉnh (mean ± std, t‑test, cải thiện %) cho bài.

## Chưa làm được (giới hạn kỹ thuật, cần nói rõ trong bài)

- **LastFM**: ma trận đồng xuất hiện 48.123² ≈ 18,5 GB — phải viết lại bằng ma trận thưa +
  TruncatedSVD trước khi chạy. Đây là giới hạn RAM, GPU không giúp.
- **Baseline knowledge graph thật**: chưa tích hợp; mọi tuyên bố về KG cần KG thật + baseline
  KG thật (KGAT), là một hạng mục lớn riêng.
