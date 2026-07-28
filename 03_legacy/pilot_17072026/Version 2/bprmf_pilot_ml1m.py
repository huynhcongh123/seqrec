#!/usr/bin/env python3
"""
Thí điểm SIRM+ trên BPR-MF (ML-1M).

BPR-MF không có self-attention nên không tự học được quan hệ đồng xuất hiện —
đây là nơi prior cấu trúc có cơ hội đóng góp thật, khác với Transformer.

Dùng chung split, cách mask và chỉ số với sirm_pilot_ml1m.py để so sánh công bằng.

Chạy: python bprmf_pilot_ml1m.py --batch 4096
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sirm_pilot_ml1m import (
    load_sequences, leave_one_out, structural_signatures,
    SIRM, set_seed, mean_std, paired_t, resolve_data_path, setup_utf8_log,
)
from sklearn.cluster import KMeans


# ------------------------------------------------------------------ mô hình --

class BPRMF(nn.Module):
    """Phân rã ma trận: điểm số = P_u . Q_i + b_i, không có thông tin thứ tự."""

    def __init__(self, num_users, num_items, dim, sirm=None):
        super().__init__()
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
        """Hoà trộn giống bài báo: v_i = e_i + lambda * s_i."""
        W = self.item_emb.weight
        if self.sirm is not None:
            ids = torch.arange(W.size(0), device=W.device)
            W = W + self.sirm.lam * self.sirm(ids)
        return W

    def score(self, users, items, V=None):
        if V is None:
            V = self.item_matrix()
        return (self.user_emb(users) * V[items]).sum(-1) + self.item_bias(items).squeeze(-1)


# ------------------------------------------------------------ lấy mẫu âm ----

class TripleSampler:
    """Sinh bộ ba (user, item dương, item âm) cho mất mát BPR."""

    def __init__(self, train, num_users, num_items, seed):
        self.rng = np.random.RandomState(seed)
        self.num_items = num_items
        self.users = np.array([u for u, items in train.items() for _ in items], dtype=np.int64)
        self.pos = np.array([i for items in train.values() for i in items], dtype=np.int64)
        # Bảng tra cứu để loại item người dùng đã xem khỏi mẫu âm.
        self.seen = np.zeros((num_users + 1, num_items + 1), dtype=bool)
        for u, items in train.items():
            self.seen[u, items] = True

    def __len__(self):
        return len(self.users)

    def epoch(self, batch):
        order = self.rng.permutation(len(self.users))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            u, p = self.users[idx], self.pos[idx]
            n = self.rng.randint(1, self.num_items + 1, size=len(idx))
            bad = self.seen[u, n]
            while bad.any():                       # lấy lại cho tới khi item âm thật sự chưa xem
                n[bad] = self.rng.randint(1, self.num_items + 1, size=int(bad.sum()))
                bad = self.seen[u, n]
            yield u, p, n


# ---------------------------------------------------------- train / đánh giá --

def train_epoch(model, sampler, opt, batch, device):
    model.train()
    total, nb = 0.0, 0
    for u, p, n in sampler.epoch(batch):
        u = torch.from_numpy(u).to(device)
        p = torch.from_numpy(p).to(device)
        n = torch.from_numpy(n).to(device)
        V = model.item_matrix()
        loss = -F.logsigmoid(model.score(u, p, V) - model.score(u, n, V)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
        nb += 1
    return total / max(nb, 1)


@torch.no_grad()
def evaluate(model, train, targets, extra_hist, device, ks=(5, 10), batch=512):
    """Xếp hạng toàn bộ catalogue, bỏ ra item đã xem — giống hệt thí điểm Transformer."""
    model.eval()
    users = sorted(targets.keys())
    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    recommended = {k: set() for k in ks}
    total = 0
    V = model.item_matrix()
    bias = model.item_bias.weight.squeeze(-1)

    for i in range(0, len(users), batch):
        chunk = users[i:i + batch]
        uid = torch.tensor(chunk, dtype=torch.long, device=device)
        scores = model.user_emb(uid) @ V.t() + bias        # (B, num_items+1)
        scores[:, 0] = -1e9
        for b, u in enumerate(chunk):
            hist = list(train[u])
            if extra_hist is not None:
                hist.append(extra_hist[u])
            idx = torch.tensor(sorted(set(hist)), dtype=torch.long, device=device)
            scores[b, idx] = -1e9

        topk = torch.topk(scores, max(ks), dim=-1).indices.cpu().numpy()
        for b, u in enumerate(chunk):
            gold = targets[u]
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


def run_one(use_sirm, seed, data, args, device, cache_dir):
    set_seed(seed)
    seqs = load_sequences(data)
    train, val, test = leave_one_out(seqs)
    num_items = max(max(v) for v in seqs.values())
    num_users = max(seqs.keys())

    sirm = None
    if use_sirm:
        X = structural_signatures(train, num_items, args.window, args.pca_dim, cache_dir)
        km = KMeans(n_clusters=args.clusters, random_state=seed, n_init=10).fit(X[1:])
        sirm = SIRM(X, km.cluster_centers_, args.dim, args.gamma_init)

    model = BPRMF(num_users, num_items, args.dim, sirm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    sampler = TripleSampler(train, num_users, num_items, seed)

    best, best_state, patience = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, sampler, opt, args.batch, device)
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            v = evaluate(model, train, val, None, device)
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

    if best_state is not None:
        model.load_state_dict(best_state)
    res = evaluate(model, train, test, val, device)
    if sirm is not None:
        res["lambda"] = float(model.sirm.lam.detach().cpu())
        res["gamma"] = float(model.sirm.log_gamma.exp().detach().cpu())
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(Path(__file__).resolve().parent / "ml-1m.txt"))
    p.add_argument("--variants", nargs="+", choices=["base", "sirm"], default=["base", "sirm"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 3407, 8888, 12345])
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=1e-5)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--pca-dim", type=int, default=100)
    p.add_argument("--clusters", type=int, default=30)
    # 1.0 khiến softmax gán mềm sụp về phân bố đều trên ML-1M, nên mặc định dùng thang đã hiệu chỉnh.
    p.add_argument("--gamma-init", type=float, default=1000.0)
    p.add_argument("--out", default=None)
    p.add_argument("--log", default=None, help="write a UTF-8 log file directly from Python")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    here = Path(__file__).resolve().parent
    args.data = str(resolve_data_path(args.data, here))
    out_path = Path(args.out) if args.out else here / "results_bprmf_ml1m.json"
    setup_utf8_log(args.log)

    print(f"thiết bị : {device}")
    print(f"dữ liệu  : {args.data}")
    print(f"seed     : {args.seeds}\n")

    results = {"config": vars(args), "device": device, "runs": {}}
    for variant in args.variants:
        results["runs"][variant] = []
        for seed in args.seeds:
            print(f"[{variant}] seed {seed}")
            t0 = time.time()
            r = run_one(variant == "sirm", seed, args.data, args, device, here / ".cache")
            r["seed"] = seed
            r["minutes"] = round((time.time() - t0) / 60, 2)
            results["runs"][variant].append(r)
            print(f"  -> test HR@10 {r['HR@10']:.4f} NDCG@10 {r['NDCG@10']:.4f} "
                  f"AD@10 {r['AD@10']} ({r['minutes']} phút)\n", flush=True)
            out_path.write_text(json.dumps(results, indent=2))

    metrics = ["HR@5", "HR@10", "NDCG@5", "NDCG@10", "AD@5", "AD@10"]
    fmt_of = lambda m: "{:.1f}" if m.startswith("AD") else "{:.4f}"
    summary = {}

    print("=" * 78)
    print(f"KẾT QUẢ — BPR-MF trên ML-1M (trung bình ± độ lệch chuẩn qua {len(args.seeds)} seed)")
    print("=" * 78)

    if len(args.variants) == 1:
        v = args.variants[0]
        print(f"{'chỉ số':10} {('gốc' if v == 'base' else '+SIRM+'):>20}")
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
