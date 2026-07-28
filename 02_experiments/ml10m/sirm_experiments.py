#!/usr/bin/env python3
"""
Khung thực nghiệm hợp nhất cho SIRM+ — một giao thức duy nhất cho mọi mô hình.

Ba xương sống trên phổ "khả năng tự nắm bắt đồng xuất hiện":
    mf          : phân rã ma trận thuần, KHÔNG có thông tin thứ tự  (không nắm co-occ)
    fpmc        : Markov bậc 1 (điểm số phụ thuộc item liền trước)   (co-occ cục bộ)
    transformer : SASRec nhân quả, self-attention                    (co-occ đầy đủ)

Mỗi mô hình chạy hai biến thể: base và +SIRM+. Đo HR/NDCG/AD và tách HR@10 theo
nhóm phổ biến (G1 đầu .. G4 đuôi). Kết quả đã KHỬ NGẪU NHIÊN để tái lập được.

Ví dụ:
    python sirm_experiments.py --model mf          --batch 4096
    python sirm_experiments.py --model fpmc        --batch 4096
    python sirm_experiments.py --model transformer --batch 128
    python sirm_experiments.py --model mf --ablation hard    # gán cứng thay vì mềm
    python sirm_experiments.py --model mf --data ml-10m.txt --out mf_ml10m.json
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # cho matmul CUDA tất định
# Ép đơn luồng cho BLAS/OpenMP: cộng dồn song song của KMeans/PCA/numpy không tất định,
# đủ để lệch centroid rồi khuếch đại qua huấn luyện. Phải đặt TRƯỚC khi import numpy.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

np.seterr(divide="ignore", over="ignore", invalid="ignore")


# ============================================================ tất định ======

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)                  # torch CPU đơn luồng -> reduction tất định
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================== dữ liệu =====

def load_sequences(path):
    seqs = {}
    for line in open(path, "r", encoding="utf-8", errors="ignore"):
        p = line.split()
        if len(p) >= 2:
            seqs[int(p[0])] = [int(x) for x in p[1:]]
    return seqs


def leave_one_out(seqs, min_len=3):
    train, val, test = {}, {}, {}
    for u, items in seqs.items():
        if len(items) < min_len:
            continue
        train[u] = items[:-2]
        val[u] = items[-2]
        test[u] = items[-1]
    return train, val, test


def popularity_groups(train, num_items, n_groups=4):
    """G1 phổ biến nhất .. Gn đuôi dài nhất, chia đều theo số item."""
    freq = np.zeros(num_items + 1)
    for items in train.values():
        for it in items:
            freq[it] += 1
    order = np.argsort(freq[1:])[::-1] + 1
    group = np.zeros(num_items + 1, dtype=int)
    for gi, idxs in enumerate(np.array_split(order, n_groups), 1):
        group[idxs] = gi
    return group


def build_cooccurrence(train, num_items, window):
    """Đếm đồng xuất hiện trong cửa sổ trượt (vector hoá) rồi chuẩn hoá theo dòng.

    Mỗi cặp (item tâm, item lân cận) trong bán kính `window` được +1 cho cả hai chiều,
    tương đương vòng lặp t/k nhưng nhanh gấp bội — cần thiết cho ML-10M.
    """
    rows, cols = [], []
    for items in train.values():
        a = np.asarray(items, dtype=np.int64)
        L = len(a)
        for delta in range(1, window + 1):
            if delta >= L:
                break
            i, j = a[:-delta], a[delta:]
            rows.append(i); cols.append(j)      # tâm=i, lân cận=j
            rows.append(j); cols.append(i)      # đối xứng
    C = np.zeros((num_items + 1, num_items + 1), dtype=np.float32)
    if rows:
        np.add.at(C, (np.concatenate(rows), np.concatenate(cols)), 1.0)
    row = C.sum(1, keepdims=True)
    row[row == 0] = 1.0
    return C / row


def structural_signatures(train, num_items, window, n_components, use_pca, cache_dir):
    """Đồng xuất hiện -> (PCA hoặc thô). Không phụ thuộc seed nên cache dùng chung."""
    tag = f"w{window}_d{n_components}" + ("" if use_pca else "_raw")
    cache = Path(cache_dir) / f"sig_{tag}.npy"
    if cache.exists():
        return np.load(cache)

    print(f"  Dựng ma trận đồng xuất hiện (cửa sổ={window}) ...", flush=True)
    C = build_cooccurrence(train, num_items, window)

    if use_pca:
        print(f"  PCA xuống {n_components} chiều ...", flush=True)
        pca = PCA(n_components=n_components, random_state=0)
        X = pca.fit_transform(C[1:, 1:])
        print(f"    giữ lại {pca.explained_variance_ratio_.sum()*100:.1f}% phương sai")
        d = n_components
    else:
        print("  KHÔNG dùng PCA — lấy thẳng hàng đồng xuất hiện làm chữ ký (ablation)", flush=True)
        X = C[1:, 1:]
        d = X.shape[1]

    X_full = np.zeros((num_items + 1, d), dtype=np.float32)
    X_full[1:] = X
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, X_full)
    return X_full


# ================================================================= SIRM+ =====

class SIRM(nn.Module):
    """Gán mỗi item vào K prototype rồi lấy tổ hợp làm prior cấu trúc s_i."""

    def __init__(self, X, centroids, emb_dim, gamma_init=1000.0, hard=False):
        super().__init__()
        self.register_buffer("X", torch.tensor(X, dtype=torch.float32))
        self.mu = nn.Parameter(torch.tensor(centroids, dtype=torch.float32))
        self.proto = nn.Parameter(torch.randn(centroids.shape[0], emb_dim) * 0.02)
        self.log_gamma = nn.Parameter(torch.tensor(float(np.log(gamma_init))))
        self.lam = nn.Parameter(torch.tensor(0.5))
        self.hard = hard

    def forward(self, item_ids):
        x = self.X[item_ids.reshape(-1)]                       # (N, d_p)
        # ||x-mu||^2 tính bằng matmul (tất định với CUBLAS_WORKSPACE_CONFIG), tránh cdist.
        x2 = (x * x).sum(-1, keepdim=True)
        mu2 = (self.mu * self.mu).sum(-1)
        d2 = x2 + mu2.unsqueeze(0) - 2.0 * (x @ self.mu.t())
        if self.hard:                                          # ablation: gán cứng vào prototype gần nhất
            s = self.proto[d2.argmin(-1)]
        else:
            alpha = torch.softmax(-self.log_gamma.exp() * d2, dim=-1)
            s = alpha @ self.proto
        return s.reshape(*item_ids.shape, -1)


def make_sirm(train, num_items, dim, seed, args):
    X = structural_signatures(train, num_items, args.window, args.pca_dim,
                              use_pca=(args.ablation != "nopca"), cache_dir=args.cache)
    km = KMeans(n_clusters=args.clusters, random_state=seed, n_init=10).fit(X[1:])
    return SIRM(X, km.cluster_centers_, dim, args.gamma_init, hard=(args.ablation == "hard"))


# ============================================================= xương sống ====

class FactorModel(nn.Module):
    """MF (use_prev=False) hoặc FPMC bậc-1 với item factor gộp (use_prev=True).

    Điểm số item i cho user u, item liền trước l:
        MF   : <P_u,        V_i>
        FPMC : <P_u + V_l,  V_i>
    """
    mode = "bpr"

    def __init__(self, num_users, num_items, dim, use_prev, sirm=None):
        super().__init__()
        self.use_prev = use_prev
        self.user_emb = nn.Embedding(num_users + 1, dim)
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.item_bias = nn.Embedding(num_items + 1, 1, padding_idx=0)
        self.sirm = sirm
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def item_matrix(self):
        W = self.item_emb.weight
        if self.sirm is not None:
            W = W + self.sirm.lam * self.sirm(torch.arange(W.size(0), device=W.device))
        return W

    def context(self, users, prev, V):
        c = self.user_emb(users)
        if self.use_prev:
            c = c + V[prev]
        return c

    def score_all(self, users, prev, seq, V):
        return self.context(users, prev, V) @ V.t() + self.item_bias.weight.squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d, maxlen):
        super().__init__()
        self.pe = nn.Embedding(maxlen, d)

    def forward(self, L, device):
        return self.pe(torch.arange(L, device=device))


class Transformer(nn.Module):
    """SASRec nhân quả: mỗi vị trí chỉ nhìn về quá khứ, dự đoán item kế tiếp."""
    mode = "ce"

    def __init__(self, num_items, dim, maxlen, layers, heads, dropout, sirm=None):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.pos = PositionalEncoding(dim, maxlen)
        self.sirm = sirm
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def item_matrix(self):
        W = self.item_emb.weight
        if self.sirm is not None:
            W = W + self.sirm.lam * self.sirm(torch.arange(W.size(0), device=W.device))
        return W

    def encode(self, seq, V):
        B, L = seq.shape
        h = self.drop(V[seq] + self.pos(L, seq.device).unsqueeze(0))
        mask = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1)
        h = self.encoder(h, mask=mask, src_key_padding_mask=(seq == 0))
        return self.ln(h)

    def score_all(self, users, prev, seq, V):
        return self.encode(seq, V)[:, -1, :] @ V.t()


# ========================================================== lấy mẫu / data ==

class BPRSampler:
    """Sinh (user, prev, pos, neg). MF: pos là mọi item đã xem, prev=0.
       FPMC: pos là item kế tiếp, prev là item liền trước."""

    def __init__(self, train, num_users, num_items, use_prev, seed):
        self.rng = np.random.RandomState(seed)
        self.num_items = num_items
        u_list, prev_list, pos_list = [], [], []
        for u, items in train.items():
            if use_prev:
                for t in range(1, len(items)):
                    u_list.append(u); prev_list.append(items[t - 1]); pos_list.append(items[t])
            else:
                for it in items:
                    u_list.append(u); prev_list.append(0); pos_list.append(it)
        self.u = np.array(u_list, np.int64)
        self.prev = np.array(prev_list, np.int64)
        self.pos = np.array(pos_list, np.int64)
        self.seen = np.zeros((num_users + 1, num_items + 1), bool)
        for u, items in train.items():
            self.seen[u, items] = True

    def epoch(self, batch):
        order = self.rng.permutation(len(self.u))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            u, pv, po = self.u[idx], self.prev[idx], self.pos[idx]
            ng = self.rng.randint(1, self.num_items + 1, len(idx))
            bad = self.seen[u, ng]
            while bad.any():
                ng[bad] = self.rng.randint(1, self.num_items + 1, int(bad.sum()))
                bad = self.seen[u, ng]
            yield u, pv, po, ng


class NextItemDataset(Dataset):
    def __init__(self, train, maxlen):
        self.users = sorted(train.keys())
        self.train = train
        self.maxlen = maxlen

    def __len__(self):
        return len(self.users)

    def __getitem__(self, i):
        s = self.train[self.users[i]][-(self.maxlen + 1):]
        inp, tgt = s[:-1], s[1:]
        pad = self.maxlen - len(inp)
        return (torch.tensor([0] * pad + inp), torch.tensor([0] * pad + tgt))


# ============================================================ train / eval ===

def train_bpr(model, sampler, opt, batch, device):
    model.train()
    tot, nb = 0.0, 0
    for u, pv, po, ng in sampler.epoch(batch):
        u = torch.from_numpy(u).to(device); pv = torch.from_numpy(pv).to(device)
        po = torch.from_numpy(po).to(device); ng = torch.from_numpy(ng).to(device)
        V = model.item_matrix()
        c = model.context(u, pv, V)
        s_pos = (c * V[po]).sum(-1) + model.item_bias(po).squeeze(-1)
        s_neg = (c * V[ng]).sum(-1) + model.item_bias(ng).squeeze(-1)
        loss = -F.logsigmoid(s_pos - s_neg).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    return tot / max(nb, 1)


def train_ce(model, loader, opt, device):
    model.train()
    tot, nb = 0.0, 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        V = model.item_matrix()
        logits = model.encode(inp, V) @ V.t()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot += loss.item(); nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def evaluate(model, train, targets, extra_hist, num_items, maxlen, device,
             groups=None, ks=(5, 10), batch=512):
    model.eval()
    users = sorted(targets.keys())
    hits = {k: 0 for k in ks}; ndcg = {k: 0.0 for k in ks}
    rec = {k: set() for k in ks}; g_hit, g_cnt = {}, {}
    total = 0
    V = model.item_matrix()

    for i in range(0, len(users), batch):
        chunk = users[i:i + batch]
        prev, seqs, seen, golds = [], [], [], []
        for u in chunk:
            hist = list(train[u])
            if extra_hist is not None:
                hist = hist + [extra_hist[u]]
            prev.append(hist[-1] if hist else 0)
            s = hist[-maxlen:]
            seqs.append([0] * (maxlen - len(s)) + s)
            seen.append(hist); golds.append(targets[u])

        uid = torch.tensor(chunk, device=device)
        pv = torch.tensor(prev, device=device)
        sq = torch.tensor(seqs, device=device)
        scores = model.score_all(uid, pv, sq, V)
        scores[:, 0] = -1e9
        for b, hist in enumerate(seen):
            scores[b, torch.tensor(sorted(set(hist)), device=device)] = -1e9

        topk = torch.topk(scores, max(ks), dim=-1).indices.cpu().numpy()
        for b, gold in enumerate(golds):
            for k in ks:
                row = topk[b, :k]; rec[k].update(row.tolist())
                pos = np.where(row == gold)[0]
                if len(pos):
                    hits[k] += 1; ndcg[k] += 1.0 / math.log2(pos[0] + 2)
            if groups is not None:
                g = int(groups[gold]); g_cnt[g] = g_cnt.get(g, 0) + 1
                if gold in topk[b, :10]:
                    g_hit[g] = g_hit.get(g, 0) + 1
            total += 1

    out = {}
    for k in ks:
        out[f"HR@{k}"] = hits[k] / total
        out[f"NDCG@{k}"] = ndcg[k] / total
        out[f"AD@{k}"] = len(rec[k])
    if groups is not None:
        for g in sorted(g_cnt):
            out[f"HR@10_G{g}"] = g_hit.get(g, 0) / g_cnt[g]
            out[f"count_G{g}"] = g_cnt[g]
    return out


# ============================================================ một lượt chạy ==

def build_model(name, num_users, num_items, sirm, args):
    if name == "mf":
        return FactorModel(num_users, num_items, args.dim, use_prev=False, sirm=sirm)
    if name == "fpmc":
        return FactorModel(num_users, num_items, args.dim, use_prev=True, sirm=sirm)
    if name == "transformer":
        return Transformer(num_items, args.dim, args.maxlen, args.layers,
                           args.heads, args.dropout, sirm=sirm)
    raise ValueError(name)


def run_one(use_sirm, seed, seqs, args, device):
    set_seed(seed)
    train, val, test = leave_one_out(seqs)
    num_items = max(max(v) for v in seqs.values())
    num_users = max(seqs.keys())
    groups = popularity_groups(train, num_items)

    sirm = make_sirm(train, num_items, args.dim, seed, args) if use_sirm else None
    model = build_model(args.model, num_users, num_items, sirm, args).to(device)

    if model.mode == "bpr":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
        sampler = BPRSampler(train, num_users, num_items, model.use_prev, seed)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        loader = DataLoader(NextItemDataset(train, args.maxlen), batch_size=args.batch, shuffle=True)

    best, best_state, wait = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        if model.mode == "bpr":
            loss = train_bpr(model, sampler, opt, args.batch, device)
        else:
            loss = train_ce(model, loader, opt, device)
        if ep % args.eval_every == 0 or ep == args.epochs:
            v = evaluate(model, train, val, None, num_items, args.maxlen, device)
            print(f"    epoch {ep:3d} | loss {loss:.4f} | val HR@10 {v['HR@10']:.4f}", flush=True)
            if v["HR@10"] > best:
                best, wait = v["HR@10"], 0
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                wait += 1
                if wait >= args.patience:
                    print(f"    dừng sớm ở epoch {ep}"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    res = evaluate(model, train, test, val, num_items, args.maxlen, device, groups=groups)
    if sirm is not None:
        res["lambda"] = float(model.sirm.lam.detach().cpu())
        res["gamma"] = float(model.sirm.log_gamma.exp().detach().cpu())
    return res


# =================================================================== main ====

def mean_std(v):
    m = sum(v) / len(v)
    return m, (math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0)


def paired_t(a, b):
    d = [x - y for x, y in zip(a, b)]
    if len(d) < 2:
        return None
    m, s = mean_std(d)
    return None if s == 0 else m / (s / math.sqrt(len(d)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mf", "fpmc", "transformer"])
    p.add_argument("--data", default=str(Path(__file__).resolve().parent / "ml-1m.txt"))
    p.add_argument("--ablation", default="none", choices=["none", "hard", "nopca"])
    p.add_argument("--variants", nargs="+", choices=["base", "sirm"], default=["base", "sirm"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 3407, 8888, 12345])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=1e-5)
    p.add_argument("--maxlen", type=int, default=200)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--pca-dim", type=int, default=100)
    p.add_argument("--clusters", type=int, default=30)
    p.add_argument("--gamma-init", type=float, default=1000.0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="cpu cho kết quả tất định tuyệt đối (chậm hơn)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    args.cache = here / ".cache"

    if args.device != "auto":
        device = args.device
    else:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    tag = args.model + ("" if args.ablation == "none" else f"_{args.ablation}")
    out_path = Path(args.out) if args.out else here / f"results_{tag}.json"

    print(f"mô hình  : {args.model}   ablation: {args.ablation}")
    print(f"thiết bị : {device}   |  dữ liệu: {Path(args.data).name}")
    print(f"seed     : {args.seeds}\n")

    seqs = load_sequences(args.data)

    done = {}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            ignore = {"out", "variants", "seeds", "cache"}
            if all(prev.get("config", {}).get(k) == (str(v) if k == "data" else v)
                   for k, v in vars(args).items() if k not in ignore and k != "data") \
               and prev.get("config", {}).get("model") == args.model \
               and prev.get("config", {}).get("ablation") == args.ablation:
                done = {v: {r["seed"]: r for r in runs} for v, runs in prev.get("runs", {}).items()}
                if any(done.values()):
                    print("(chạy tiếp: bỏ qua lượt đã có)\n")
        except Exception:
            done = {}

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "cache"}
    results = {"config": cfg, "device": device, "runs": {}}
    for variant in args.variants:
        results["runs"][variant] = []
        for seed in args.seeds:
            if seed in done.get(variant, {}):
                r = done[variant][seed]; print(f"[{variant}] seed {seed} — đã có, bỏ qua")
            else:
                print(f"[{variant}] seed {seed}")
                t0 = time.time()
                r = run_one(variant == "sirm", seed, seqs, args, device)
                r["seed"] = seed; r["minutes"] = round((time.time() - t0) / 60, 2)
                print(f"  -> HR@10 {r['HR@10']:.4f} NDCG@10 {r['NDCG@10']:.4f} "
                      f"AD@10 {r['AD@10']} ({r['minutes']} phút)\n", flush=True)
            results["runs"][variant].append(r)
            out_path.write_text(json.dumps(results, indent=2))

    report(results, args)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nĐã lưu -> {out_path}")


def report(results, args):
    runs = results["runs"]
    print("\n" + "=" * 78)
    print(f"KẾT QUẢ — {args.model.upper()}"
          f"{'' if args.ablation=='none' else ' ['+args.ablation+']'} trên {Path(args.data).name}")
    print("=" * 78)
    if len(args.variants) < 2:
        v = args.variants[0]
        for m in ["HR@5", "HR@10", "NDCG@5", "NDCG@10", "AD@5", "AD@10"]:
            mm, ss = mean_std([r[m] for r in runs[v]])
            f = "{:.1f}" if m.startswith("AD") else "{:.4f}"
            print(f"{m:10} {f.format(mm)} ± {f.format(ss)}")
        return

    print(f"{'chỉ số':10} {'base':>17} {'+SIRM+':>17} {'chênh':>9} {'t':>7} {'p<.05':>7}")
    print("-" * 78)
    for m in ["HR@5", "HR@10", "NDCG@5", "NDCG@10", "AD@5", "AD@10"]:
        a = [r[m] for r in runs["base"]]; b = [r[m] for r in runs["sirm"]]
        ma, sa = mean_std(a); mb, sb = mean_std(b); t = paired_t(b, a)
        f = "{:.1f}" if m.startswith("AD") else "{:.4f}"
        sig = "CÓ" if t and abs(t) > 2.78 else "không"
        print(f"{m:10} {f.format(ma)+'±'+f.format(sa):>17} {f.format(mb)+'±'+f.format(sb):>17} "
              f"{(mb-ma)/ma*100:>+7.1f}% {(f'{t:.2f}' if t else 'n/a'):>7} {sig:>7}")

    if all("HR@10_G1" in r for r in runs["sirm"]):
        print("\n" + "=" * 78)
        print("ĐUÔI DÀI — HR@10 theo nhóm (G1 đầu .. G4 đuôi sâu)")
        print("=" * 78)
        print(f"{'nhóm':5} {'#test':>6} {'base':>16} {'+SIRM+':>16} {'chênh':>10} {'t':>7}")
        print("-" * 78)
        for g in [1, 2, 3, 4]:
            k = f"HR@10_G{g}"
            a = [r[k] for r in runs["base"] if k in r]; b = [r[k] for r in runs["sirm"] if k in r]
            if not a or not b:
                continue
            cnt = runs["base"][0].get(f"count_G{g}", 0)
            ma, sa = mean_std(a); mb, sb = mean_std(b); t = paired_t(b, a)
            ch = f"{(mb-ma)/ma*100:+.1f}%" if ma else f"{mb-ma:+.4f}"
            print(f"G{g:<4} {cnt:>6} {f'{ma:.4f}±{sa:.4f}':>16} {f'{mb:.4f}±{sb:.4f}':>16} "
                  f"{ch:>10} {(f'{t:.2f}' if t else 'n/a'):>7}")


if __name__ == "__main__":
    main()
