"""Summarize camera-ready experiment JSONs using matched-seed paired tests."""

import json
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import ttest_rel


ROOT = Path(__file__).resolve().parent
SOURCES = [
    ("MF", "ML-1M", "01_final_alignment/reviewer_mf_ml1m.json"),
    ("MF", "ML-10M", "02_experiments/ml10m/res_mf_ml10m.json"),
    ("MF", "AmazonBook", "02_experiments/amazonbook/res_mf_amazon.json"),
    ("FPMC", "ML-1M", "01_final_alignment/reviewer_fpmc_ml1m.json"),
    ("FPMC", "ML-10M", "02_experiments/ml10m/res_fpmc_ml10m.json"),
    ("FPMC", "AmazonBook", "02_experiments/amazonbook/res_fpmc_amazon.json"),
    ("Transformer", "ML-1M", "02_experiments/ml1m_paper_results/paper_ml1m_results/res_transformer_ml1m.json"),
    ("Transformer", "ML-10M", "02_experiments/ml10m/res_transformer_ml10m.json"),
    ("Transformer", "AmazonBook", "02_experiments/amazonbook/res_transformer_amazon.json"),
    ("MF", "ML-1M 50%", "01_final_alignment/reviewer_mf_ml1m_s50.json"),
    ("MF", "ML-1M 25%", "01_final_alignment/reviewer_mf_ml1m_s25.json"),
]
SEEDS = {42, 2024, 3407, 8888, 12345}


def summarize(model, dataset, relative_path):
    path = ROOT / relative_path
    if not path.exists():
        print(f"{model:11} {dataset:12} MISSING {relative_path}")
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    cfg = result["config"]
    base = {int(row["seed"]): row for row in result["runs"].get("base", [])}
    prior = {int(row["seed"]): row for row in result["runs"].get("sirm", [])}
    if set(base) != SEEDS or set(prior) != SEEDS:
        print(f"{model:11} {dataset:12} INCOMPLETE base={sorted(base)} prior={sorted(prior)}")
        return
    if cfg.get("implementation") != "reviewer-v2-canonical-fpmc":
        print(f"{model:11} {dataset:12} WARNING: old implementation tag")
    print(f"{model:11} {dataset:12} epochs={cfg['epochs']} batch={cfg['batch']}")
    for metric in ("HR@10", "NDCG@10", "AD@10"):
        b = [float(base[s][metric]) for s in sorted(SEEDS)]
        p = [float(prior[s][metric]) for s in sorted(SEEDS)]
        test = ttest_rel(p, b)
        fmt = ".1f" if metric == "AD@10" else ".4f"
        print(
            f"  {metric:7} {format(mean(b), fmt)} +/- {format(stdev(b), fmt)}"
            f" -> {format(mean(p), fmt)} +/- {format(stdev(p), fmt)}"
            f"  delta={100 * (mean(p) / mean(b) - 1):+.2f}%"
            f"  t={test.statistic:+.3f} p={test.pvalue:.5g}"
        )
    if model == "MF" and dataset in {"ML-1M", "ML-10M", "AmazonBook"}:
        print("  tail:", end="")
        for group in range(1, 5):
            key = f"HR@10_G{group}"
            print(
                f" G{group} {mean(base[s][key] for s in SEEDS):.4f}"
                f"->{mean(prior[s][key] for s in SEEDS):.4f}",
                end="",
            )
        print()


if __name__ == "__main__":
    for source in SOURCES:
        summarize(*source)
