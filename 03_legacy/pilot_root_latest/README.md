# Chạy thí điểm SIRM+ trên ML-1M

Thư mục này tự chứa đầy đủ (đã có sẵn dữ liệu `ml-1m.txt`). Chỉ cần copy cả thư mục `pilot` sang máy có GPU rồi làm theo 5 bước.

**Yêu cầu:** Windows + NVIDIA GPU (RTX 3050 Ti, VRAM 4GB) + Python 3.9 trở lên.

---

## Bước 1 — Mở PowerShell tại thư mục pilot

```powershell
cd đường\dẫn\tới\pilot
```

## Bước 2 — Cài thư viện

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install scikit-learn numpy
```

## Bước 3 — Kiểm tra GPU

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Phải in ra `True`.

Nếu ra `False` thì cài lại:
```powershell
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Bước 4 — Chạy thử (khoảng 2 phút)

```powershell
python sirm_pilot_ml1m.py --seeds 42 --epochs 3 --eval-every 1 --patience 99 --batch 64
```

Thấy dòng `device: cuda` và chạy xong không lỗi là được.
(Số ở bước này chưa có ý nghĩa, chỉ để kiểm tra máy chạy được.)

## Bước 5 — Chạy thật (khoảng 3–5 tiếng)

```powershell
python sirm_pilot_ml1m.py --batch 64 > ket_qua.log 2>&1
```

Nên chạy buổi tối. Nhớ cắm sạc, để Windows ở chế độ High performance, tắt Sleep.

---

## Bước 6 — Thí điểm BPR-MF (khoảng 45 phút)

BPR-MF không có self-attention nên không tự học được quan hệ đồng xuất hiện — đây là
nơi prior cấu trúc có cơ hội đóng góp thật.

```powershell
python bprmf_pilot_ml1m.py --batch 4096 > bprmf.log 2>&1
```

Nếu `CUDA out of memory` thì giảm `--batch 2048`.

---

## Lấy kết quả

Chạy xong, gửi lại các file:

- `results_ml1m.json` — thí điểm Transformer
- `results_bprmf_ml1m.json` — thí điểm BPR-MF
- các file `.log` tương ứng

---

## Lỗi thường gặp

| Lỗi | Cách sửa |
|---|---|
| `CUDA out of memory` | Thêm `--batch 32`, vẫn lỗi thì `--batch 16` |
| Chạy quá lâu | Giảm seed: `--seeds 42 2024` |
| Không tìm thấy dữ liệu | Chạy lệnh ngay trong thư mục `pilot` |
