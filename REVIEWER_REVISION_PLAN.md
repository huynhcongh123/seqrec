# CSoNet 2026 reviewer-revision pipeline

The code in this folder now implements the requested methodological changes.
The old result JSON files are intentionally not reused because the FPMC
parameterization changed; the scripts contain an implementation-version tag
that invalidates stale results when rerun.

## 1. Canonical FPMC

`FactorModel(use_prev=True)` now scores

```text
<user, item_target> + <transition_source(previous), transition_target(item)>.
```

The source and target transition embeddings are separate. The prior is added
to the item-role matrices but does not tie the learnable transition roles.

Run the three canonical FPMC suites from their dataset folders:

```powershell
python .\sirm_experiments.py --model fpmc --data .\ml-1m.txt --device cuda --seeds 42 2024 3407 8888 12345 --out reviewer_fpmc_ml1m.json
python .\sirm_experiments.py --model fpmc --data .\ml-10m.txt --device cuda --seeds 42 2024 3407 8888 12345 --out reviewer_fpmc_ml10m.json
python .\sirm_experiments.py --model fpmc --data .\AmazonBook.txt --device cuda --maxlen 50 --seeds 42 2024 3407 8888 12345 --out reviewer_fpmc_amazon.json
```

The provided `run_paper_ml10m.ps1` and `run_paper_amazon.ps1` scripts also use
five seeds for every model.

## 2. Within-domain artificial sparsification

`01_final_alignment/sparsify_sequences.py` removes interactions while keeping
the original user IDs, complete item universe, and at least three interactions
per user. The generated `.meta.json` records the audit counts.

```powershell
python .\sparsify_sequences.py --input .\ml-1m.txt --output .\ml-1m-s50.txt --keep-rate 0.50 --seed 2026
python .\sparsify_sequences.py --input .\ml-1m.txt --output .\ml-1m-s25.txt --keep-rate 0.25 --seed 2026
```

The current generated controls preserve all 3,654 ML-1M items and contain
499,570 and 249,492 interactions, respectively.

## 3. SASRec validation and statistics

The main causal Transformer-style model remains available as `--model
transformer`. A canonical SASRec-style pointwise BCE baseline with one sampled
negative per position is available as `--model sasrec`. Both support the same
five seeds and deterministic CUDA settings. The code disables flash and
memory-efficient fused attention during seeded runs so that the deterministic
protocol does not silently fall back to a nondeterministic attention kernel.

Use paired tests across the five matched seeds; the two-sided $p<0.05$ cutoff
is $|t|>2.776$ with four degrees of freedom.
