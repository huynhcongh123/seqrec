#!/usr/bin/env python3
"""
Thí điểm SIRM+ trên BPR-MF (ML-1M).

BPR-MF không có self-attention nên không tự học được quan hệ đồng xuất hiện —
đây là nơi prior cấu trúc có cơ hội đóng góp thật, khác với Transformer.

Dùng chung split, cách mask và chỉ số với sirm_pilot_ml1m.py để so sánh công bằng.

Chạy: python bprmf_pilot_ml1m.py --batch 4096
"""

import argparse
import inspect
import json
import math
import os
import sys
import time
import unicodedata
from pathlib import Path

os.environ["LOKY_MAX_CPU_COUNT"] = os.environ.get("LOKY_MAX_CPU_COUNT") or "1"

class AsciiLogStream:
    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        text = str(text).replace("đ", "d").replace("Đ", "D")
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        return self.stream.write(text)

    def flush(self):
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


if not sys.stdout.isatty():
    sys.stdout = AsciiLogStream(sys.stdout)
if not sys.stderr.isatty():
    sys.stderr = AsciiLogStream(sys.stderr)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sirm_pilot_ml1m import (
    load_sequences, leave_one_out, structural_signatures,
    SIRM, set_seed, mean_std, paired_t,
)
from sklearn.cluster import KMeans


def make_sirm(X, centroids, dim, gamma_init):
    if "gamma_init" in inspect.signature(SIRM).parameters:
        return SIRM(X, centroids, dim, gamma_init)
    sirm = SIRM(X, centroids, dim)
    if hasattr(sirm, "log_gamma"):
        with torch.no_grad():
            sirm.log_gamma.fill_(float(np.log(gamma_init)))
    return sirm


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


def popularity_groups(train, num_items, n_groups=4):
    """Chia item thành n nhóm bằng nhau theo tần suất train: G1 phổ biến nhất .. Gn đuôi dài nhất."""
    freq = np.zeros(num_items + 1)
    for items in train.values():
        for it in items:
            freq[it] += 1
    order = np.argsort(freq[1:])[::-1] + 1          # id item, phổ biến -> hiếm
    group = np.zeros(num_items + 1, dtype=int)
    for gi, idxs in enumerate(np.array_split(order, n_groups), 1):
        group[idxs] = gi
    return group


@torch.no_grad()
def evaluate(model, train, targets, extra_hist, device, ks=(5, 10), batch=512, groups=None):
    """Xếp hạng toàn bộ catalogue, bỏ ra item đã xem — giống hệt thí điểm Transformer.

    Nếu truyền groups thì đo thêm HR@10 tách theo nhóm phổ biến của item cần đoán.
    """
    model.eval()
    users = sorted(targets.keys())
    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    recommended = {k: set() for k in ks}
    g_hit, g_cnt = {}, {}
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
            if groups is not None:
                g = int(groups[gold])
                g_cnt[g] = g_cnt.get(g, 0) + 1
                if gold in topk[b, :10]:
                    g_hit[g] = g_hit.get(g, 0) + 1
            total += 1

    out = {}
    for k in ks:
        out[f"HR@{k}"] = hits[k] / total
        out[f"NDCG@{k}"] = ndcg[k] / total
        out[f"AD@{k}"] = len(recommended[k])
    if groups is not None:
        for g in sorted(g_cnt):
            out[f"HR@10_G{g}"] = g_hit.get(g, 0) / g_cnt[g]
            out[f"count_G{g}"] = g_cnt[g]
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
        sirm = make_sirm(X, km.cluster_centers_, args.dim, args.gamma_init)

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
    groups = popularity_groups(train, num_items)
    res = evaluate(model, train, test, val, device, groups=groups)
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
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    here = Path(__file__).resolve().parent
    out_path = Path(args.out) if args.out else here / "results_bprmf_ml1m.json"

    print(f"thiết bị : {device}")
    print(f"dữ liệu  : {args.data}")
    print(f"seed     : {args.seeds}\n")

    # Chạy tiếp sau khi bị gián đoạn: nạp lại các lượt đã xong nếu CÙNG cấu hình.
    done = {}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            ignore = {"out", "variants", "seeds"}
            same = all(prev.get("config", {}).get(k) == v
                       for k, v in vars(args).items() if k not in ignore)
            if same:
                done = {v: {r["seed"]: r for r in runs}
                        for v, runs in prev.get("runs", {}).items()}
                if any(done.values()):
                    print("(thấy kết quả cũ cùng cấu hình -> chạy tiếp, bỏ qua lượt đã xong)\n")
        except Exception:
            done = {}

    results = {"config": vars(args), "device": device, "runs": {}}
    for variant in args.variants:
        results["runs"][variant] = []
        for seed in args.seeds:
            if seed in done.get(variant, {}):
                r = done[variant][seed]
                print(f"[{variant}] seed {seed} — đã có, bỏ qua")
            else:
                print(f"[{variant}] seed {seed}")
                t0 = time.time()
                r = run_one(variant == "sirm", seed, args.data, args, device, here / ".cache")
                r["seed"] = seed
                r["minutes"] = round((time.time() - t0) / 60, 2)
                print(f"  -> test HR@10 {r['HR@10']:.4f} NDCG@10 {r['NDCG@10']:.4f} "
                      f"AD@10 {r['AD@10']} ({r['minutes']} phút)\n", flush=True)
            results["runs"][variant].append(r)
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

    if len(args.variants) == 2 and any("HR@10_G1" in r for r in results["runs"]["sirm"]):
        print("\n" + "=" * 78)
        print("PHÂN TÍCH ĐUÔI DÀI — HR@10 theo nhóm (G1 phổ biến nhất .. G4 đuôi dài nhất)")
        print("=" * 78)
        print(f"{'nhóm':6} {'#test':>7} {'gốc':>14} {'+SIRM+':>14} {'chênh':>9} {'t ghép cặp':>11}")
        print("-" * 78)
        for g in [1, 2, 3, 4]:
            key = f"HR@10_G{g}"
            a = [r[key] for r in results["runs"]["base"] if key in r]
            b = [r[key] for r in results["runs"]["sirm"] if key in r]
            if not a or not b:
                continue
            cnt = results["runs"]["base"][0].get(f"count_G{g}", 0)
            ma, sa = mean_std(a)
            mb, sb = mean_std(b)
            chenh = f"{(mb-ma)/ma*100:+.1f}%" if ma else f"{mb-ma:+.4f}"  # gốc=0 thì báo tuyệt đối
            t = paired_t(b, a)
            print(f"G{g:<5} {cnt:>7} {ma:>14.4f} {mb:>14.4f} {chenh:>9} "
                  f"{('t=%.2f' % t[0]) if t else 'n/a':>11}")
        print("-" * 78)
        print("Nếu SIRM+ có ích thật cho đuôi dài thì HR@10 phải tăng rõ ở G3/G4.")

    results["summary"] = summary
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nĐã lưu -> {out_path}")


if __name__ == "__main__":
    main()
