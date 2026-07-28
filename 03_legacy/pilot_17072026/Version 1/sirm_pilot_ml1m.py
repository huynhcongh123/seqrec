#!/usr/bin/env python3
"""
Thí điểm SIRM+ trên ML-1M: so sánh mô hình gốc và mô hình có thêm prior cấu trúc
lấy từ ma trận đồng xuất hiện. In ra HR/NDCG/AD trung bình ± độ lệch chuẩn qua các seed.

Chạy: python sirm_pilot_ml1m.py --batch 64
"""

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, (os.cpu_count() or 1) - 1))
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# numpy 2 dùng BLAS Accelerate trên Apple Silicon báo tràn số giả trong matmul của
# randomized SVD và KMeans. Đã kiểm tra đầu ra vẫn hữu hạn và đúng nên tắt cảnh báo.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
sys.stderr = sys.stdout


# ---------------------------------------------------------------- dữ liệu ----

def load_sequences(path):
    """Mỗi dòng: 'userId item1 item2 ...', các item đã xếp theo thời gian."""
    seqs = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            seqs[int(parts[0])] = [int(x) for x in parts[1:]]
    return seqs


def leave_one_out(seqs, min_len=3):
    """Item cuối để test, áp chót để validation, phần còn lại để train."""
    train, val, test = {}, {}, {}
    for u, items in seqs.items():
        if len(items) < min_len:
            continue
        train[u] = items[:-2]
        val[u] = items[-2]
        test[u] = items[-1]
    return train, val, test


class NextItemDataset(Dataset):
    """Mỗi user một mẫu: đầu vào là chuỗi, nhãn là chính chuỗi đó dịch sang phải 1 bước."""

    def __init__(self, train, maxlen):
        self.users = sorted(train.keys())
        self.train = train
        self.maxlen = maxlen

    def __len__(self):
        return len(self.users)

    def __getitem__(self, i):
        s = self.train[self.users[i]][-(self.maxlen + 1):]
        inp, tgt = s[:-1], s[1:]
        # Đệm bên trái để vị trí cuối luôn là item mới nhất. Id 0 dành cho phần đệm.
        pad = self.maxlen - len(inp)
        return (torch.tensor([0] * pad + inp, dtype=torch.long),
                torch.tensor([0] * pad + tgt, dtype=torch.long))


# ------------------------------------------------------- prior cấu trúc ------

def build_cooccurrence(train, num_items, window):
    """Đếm đồng xuất hiện trong cửa sổ trượt rồi chuẩn hoá theo dòng.

    Chỉ dùng chuỗi train: item validation/test không được lọt vào đây.
    """
    C = np.zeros((num_items + 1, num_items + 1), dtype=np.float32)
    for items in train.values():
        L = len(items)
        for t in range(L):
            it = items[t]
            for k in range(max(0, t - window), min(L, t + window + 1)):
                if k != t:
                    C[it, items[k]] += 1.0
    row = C.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return C / row


def structural_signatures(train, num_items, window, n_components, cache_dir):
    """Đồng xuất hiện -> PCA. Kết quả không phụ thuộc seed nên cache lại dùng chung."""
    cache = Path(cache_dir) / f"pca_w{window}_d{n_components}.npy"
    if cache.exists():
        return np.load(cache)

    print(f"  Dựng ma trận đồng xuất hiện (cửa sổ={window}) ...", flush=True)
    t0 = time.time()
    C = build_cooccurrence(train, num_items, window)
    print(f"    xong sau {time.time()-t0:.1f}s, kích thước {C.shape}")

    print(f"  PCA xuống {n_components} chiều ...", flush=True)
    t0 = time.time()
    pca = PCA(n_components=n_components, random_state=0)
    X = pca.fit_transform(C[1:, 1:])
    ev = pca.explained_variance_ratio_.sum()
    print(f"    xong sau {time.time()-t0:.1f}s, giữ lại {ev*100:.1f}% phương sai")

    X_full = np.zeros((num_items + 1, n_components), dtype=np.float32)
    X_full[1:] = X                      # dòng 0 là phần đệm, để nguyên số 0
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, X_full)
    return X_full


class SIRM(nn.Module):
    """Gán mềm mỗi item vào K prototype, rồi lấy tổ hợp có trọng số làm prior."""

    def __init__(self, X, centroids, emb_dim, gamma_init=1.0):
        super().__init__()
        self.register_buffer("X", torch.tensor(X, dtype=torch.float32))  # chữ ký cấu trúc, cố định
        self.mu = nn.Parameter(torch.tensor(centroids, dtype=torch.float32))
        self.proto = nn.Parameter(torch.randn(centroids.shape[0], emb_dim) * 0.02)
        # Khoảng cách ||x_i - mu_k||^2 trên ML-1M chỉ cỡ 0.05, nên gamma nhỏ khiến
        # softmax gần như đều và mọi item nhận cùng một prior. Cần gamma lớn mới phân biệt được.
        self.log_gamma = nn.Parameter(torch.tensor(float(np.log(gamma_init))))
        self.lam = nn.Parameter(torch.tensor(0.5))       # trọng số hoà trộn

    def forward(self, item_ids):
        shape = item_ids.shape
        x = self.X[item_ids.reshape(-1)]
        d2 = torch.cdist(x, self.mu).pow(2)
        alpha = torch.softmax(-self.log_gamma.exp() * d2, dim=-1)
        return (alpha @ self.proto).reshape(*shape, -1)


# ------------------------------------------------------------------ mô hình --

class SeqRec(nn.Module):
    def __init__(self, num_items, emb_dim, maxlen, n_layers, n_heads, dropout, sirm=None):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, emb_dim)
        self.sirm = sirm
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=4 * emb_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(emb_dim)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def item_matrix(self):
        """Biểu diễn item sau hoà trộn: v_i = e_i + lambda * s_i."""
        W = self.item_emb.weight
        if self.sirm is not None:
            ids = torch.arange(W.size(0), device=W.device)
            W = W + self.sirm.lam * self.sirm(ids)
        return W

    def forward(self, seq):
        B, L = seq.shape
        V = self.item_matrix()
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        h = self.drop(V[seq] + self.pos_emb(pos))
        # Mask nhân quả: mỗi vị trí chỉ được nhìn về quá khứ, nếu không sẽ lộ nhãn.
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=causal, src_key_padding_mask=(seq == 0))
        return self.ln(h), V


# ---------------------------------------------------------- train / đánh giá --

def train_epoch(model, loader, opt, device):
    model.train()
    total, n = 0.0, 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        h, V = model(inp)
        logits = h @ V.t()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=0
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, train, targets, extra_hist, maxlen, device, ks=(5, 10), batch=256):
    """Xếp hạng trên toàn bộ catalogue, bỏ ra những item user đã xem."""
    model.eval()
    users = sorted(targets.keys())
    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    recommended = {k: set() for k in ks}     # để tính AD
    total = 0

    for i in range(0, len(users), batch):
        chunk = users[i:i + batch]
        seqs, seen, golds = [], [], []
        for u in chunk:
            hist = list(train[u])
            if extra_hist is not None:
                hist.append(extra_hist[u])   # lúc test thì item validation đã thấy rồi
            inp = hist[-maxlen:]
            seqs.append([0] * (maxlen - len(inp)) + inp)
            seen.append(hist)
            golds.append(targets[u])

        h, V = model(torch.tensor(seqs, dtype=torch.long, device=device))
        scores = h[:, -1, :] @ V.t()
        scores[:, 0] = -1e9                  # id 0 là phần đệm, không phải item
        for b, hist in enumerate(seen):
            idx = torch.tensor(sorted(set(hist)), dtype=torch.long, device=device)
            scores[b, idx] = -1e9            # không gợi ý lại phim đã xem

        topk = torch.topk(scores, max(ks), dim=-1).indices.cpu().numpy()
        for b, gold in enumerate(golds):
            for k in ks:
                row = topk[b, :k]
                recommended[k].update(row.tolist())
                pos = np.where(row == gold)[0]
                if len(pos) > 0:
                    hits[k] += 1
                    ndcg[k] += 1.0 / math.log2(pos[0] + 2)
            total += 1

    out = {}
    for k in ks:
        out[f"HR@{k}"] = hits[k] / total
        out[f"NDCG@{k}"] = ndcg[k] / total
        out[f"AD@{k}"] = len(recommended[k])
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_one(use_sirm, seed, data, args, device, cache_dir):
    set_seed(seed)
    seqs = load_sequences(data)
    train, val, test = leave_one_out(seqs)
    num_items = max(max(v) for v in seqs.values())

    sirm = None
    if use_sirm:
        X = structural_signatures(train, num_items, args.window, args.pca_dim, cache_dir)
        km = KMeans(n_clusters=args.clusters, random_state=seed, n_init=10).fit(X[1:])
        sirm = SIRM(X, km.cluster_centers_, args.emb_dim, args.gamma_init)

    model = SeqRec(num_items, args.emb_dim, args.maxlen, args.layers,
                   args.heads, args.dropout, sirm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loader = DataLoader(NextItemDataset(train, args.maxlen),
                        batch_size=args.batch, shuffle=True)

    best, best_state, patience = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, opt, device)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            v = evaluate(model, train, val, None, args.maxlen, device)
            print(f"    epoch {epoch:3d} | loss {loss:.4f} | "
                  f"val HR@10 {v['HR@10']:.4f} NDCG@10 {v['NDCG@10']:.4f}", flush=True)
            if v["HR@10"] > best:
                best, patience = v["HR@10"], 0
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"    dừng sớm ở epoch {epoch}")
                    break

    # Đánh giá test bằng checkpoint tốt nhất trên validation, không phải epoch cuối.
    if best_state is not None:
        model.load_state_dict(best_state)
    res = evaluate(model, train, test, val, args.maxlen, device)
    if sirm is not None:
        res["lambda"] = float(model.sirm.lam.detach().cpu())
        res["gamma"] = float(model.sirm.log_gamma.exp().detach().cpu())
    return res


# ------------------------------------------------------------- thống kê ------

def mean_std(values):
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)


def paired_t(a, b):
    """Thống kê t ghép cặp giữa hai nhóm theo từng seed."""
    d = [x - y for x, y in zip(a, b)]
    if len(d) < 2:
        return None
    m, s = mean_std(d)
    if s == 0:
        return None
    return m / (s / math.sqrt(len(d))), len(d) - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(Path(__file__).resolve().parent / "ml-1m.txt"))
    p.add_argument("--variants", nargs="+", choices=["base", "sirm"], default=["base", "sirm"],
                   help="chạy biến thể nào; base không phụ thuộc tham số SIRM nên có thể chạy riêng")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 3407, 8888, 12345])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=5, help="số lần đánh giá không cải thiện thì dừng")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--maxlen", type=int, default=200)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--pca-dim", type=int, default=100)
    p.add_argument("--clusters", type=int, default=30)
    p.add_argument("--gamma-init", type=float, default=1.0,
                   help="gamma khởi tạo cho gán mềm; 1.0 khiến alpha gần như đều trên ML-1M")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    here = Path(__file__).resolve().parent
    out_path = Path(args.out) if args.out else here / "results_ml1m.json"

    print(f"thiết bị : {device}")
    print(f"dữ liệu  : {args.data}")
    print(f"seed     : {args.seeds}\n")

    results = {"config": vars(args), "device": device, "runs": {}}
    for variant in args.variants:
        use_sirm = variant == "sirm"
        results["runs"][variant] = []
        for seed in args.seeds:
            print(f"[{variant}] seed {seed}")
            t0 = time.time()
            r = run_one(use_sirm, seed, args.data, args, device, here / ".cache")
            r["seed"] = seed
            r["minutes"] = round((time.time() - t0) / 60, 2)
            results["runs"][variant].append(r)
            print(f"  -> test HR@10 {r['HR@10']:.4f} NDCG@10 {r['NDCG@10']:.4f} "
                  f"AD@10 {r['AD@10']} ({r['minutes']} phút)\n", flush=True)
            out_path.write_text(json.dumps(results, indent=2))   # lưu dần, lỡ dừng giữa chừng

    metrics = ["HR@5", "HR@10", "NDCG@5", "NDCG@10", "AD@5", "AD@10"]
    fmt_of = lambda m: "{:.1f}" if m.startswith("AD") else "{:.4f}"
    summary = {}

    print("=" * 78)
    print(f"KẾT QUẢ — ML-1M (trung bình ± độ lệch chuẩn qua {len(args.seeds)} seed)")
    print("=" * 78)

    if len(args.variants) == 1:
        v = args.variants[0]
        print(f"{'chỉ số':10} {'gốc' if v == 'base' else '+SIRM+':>20}")
        print("-" * 78)
        for metric in metrics:
            m, s = mean_std([r[metric] for r in results["runs"][v]])
            f = fmt_of(metric)
            print(f"{metric:10} {f.format(m)+' ± '+f.format(s):>20}")
            summary[metric] = {f"{v}_mean": m, f"{v}_std": s}
    else:
        print(f"{'chỉ số':10} {'gốc':>20} {'+SIRM+':>20} {'chênh':>9} {'t ghép cặp':>11}")
        print("-" * 78)
        for metric in metrics:
            a = [r[metric] for r in results["runs"]["base"]]
            b = [r[metric] for r in results["runs"]["sirm"]]
            ma, sa = mean_std(a)
            mb, sb = mean_std(b)
            rel = (mb - ma) / ma * 100 if ma else float("nan")
            t = paired_t(b, a)
            f = fmt_of(metric)
            print(f"{metric:10} {f.format(ma)+' ± '+f.format(sa):>20} "
                  f"{f.format(mb)+' ± '+f.format(sb):>20} {rel:>+8.1f}% "
                  f"{('t=%.2f' % t[0]) if t else 'n/a':>11}")
            summary[metric] = {"base_mean": ma, "base_std": sa,
                               "sirm_mean": mb, "sirm_std": sb, "rel_pct": rel,
                               "paired_t": t[0] if t else None}
        print("-" * 78)
        print(f"Với {len(args.seeds)} seed (df={len(args.seeds)-1}): "
              f"|t| > 2.78 mới tương ứng p < 0.05 (hai phía).")

    results["summary"] = summary
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nĐã lưu -> {out_path}")


if __name__ == "__main__":
    main()
