# DRC

**Dose-response curves for grammar. It trains small language models from scratch with one rare
construction dosed at 0, 4, 16, 64, or every attested instance, then fits a Hill curve to how well
each model learned it.**

Work on how language models learn rare constructions mostly treats exposure as a switch: the
construction is either in the pretraining corpus or filtered out. The closest prior study sampled
several exposure counts but described the resulting curves in prose. DRC turns the switch into a
dial and the curve into numbers. It holds a 10-million-word corpus fixed, varies only how many
times a target construction appears, and fits the four-parameter Hill equation from pharmacology.
Its parameters answer separate questions: how much a model gets from indirect evidence alone, how
much direct evidence the rest takes, and whether learning arrives smoothly or all at once.

Manuscript draft: **`paper/main.tex`** (written for EMNLP 2026; every data-dependent claim is a
marked `RESULT PENDING` placeholder).

```
BabyLM 2026 Strict-Small (10M words) + a 100M-word replacement pool
    ->  Stanza dependency parse, 8 construction detectors, hand-labeled filter audit
    ->  dose corpora: keep 0, 4, 16, or 64 instances, swap the rest for matched sentences
    ->  one frozen 16k BPE tokenizer, LTG-BERT from scratch, 63 runs over 3 seeds
    ->  SLOR acceptability on minimal pairs, plus a 4-gram baseline on the same corpora
    ->  Hill fits, model comparison, clustering, pre-registered decision rule
```

Exposure is a dial, not a switch.

## The core idea

1. **Vary one thing.** Each dose corpus keeps exactly d instances of the target and replaces every
   removed sentence with one from the same domain and of similar length, so corpus size and genre
   mix stay constant while only the construction count moves.
2. **Fit a curve with readable parameters.** Y(D) = E0 + (Emax - E0) · D^n / (E50^n + D^n). E0 is
   the floor at zero exposure (pure indirect evidence), Emax the ceiling, E50 the dose that gets
   halfway, and n the shape: near 1 a gentle rise, much larger a step.
3. **Spend compute where the question is.** The headline quantity, E0, needs only a zero-dose
   model and a ceiling. So eight constructions are trained at dose 0, four core constructions get
   the full ladder, and one unfiltered full-corpus model per seed is the ceiling for all of them:
   21 corpora times 3 seeds is 63 runs, in one frozen recipe.
4. **Decide before looking.** Five hypotheses (smooth scaling, phase transition, heterogeneous
   regimes, indirect evidence dominates, stochastic acquisition) each have a numeric firing rule
   in `drc.analysis.decision`, fixed before any curve is fit. If several fire or none does, the
   output is "Mixed".

## What it does (design; no runs yet)

No model has been trained and no results are committed. The table below is the experimental
design, not a finding.

| Construction | Example | Tier | Expected count in BabyLM-10M |
|---|---|---|---|
| AANN | a beautiful five days in Texas | core | 200 to 500 |
| Comparative correlative | the harder you try, the worse it gets | core | 400 to 800 |
| Tough-movement | the painting is easy to admire | core | 800 to 1,500 |
| Resultative | he hammered the metal flat | core | 500 to 900 |
| Existential *there* | there are three cats on the mat | breadth | 8,000 to 20,000 |
| It-cleft | it was the dog that barked | breadth | 1,500 to 4,000 |
| Negative inversion | never have I seen such a mess | breadth | 300 to 800 |
| Subject-to-object raising | they believed him to be guilty | breadth | 600 to 1,500 |

Core constructions run at doses 0, 4, 16, and 64 plus the shared ceiling. Breadth constructions
run at dose 0 plus the ceiling. Each construction has 30 hand-written minimal pairs (240 in
total) in `evaluation/eval_items/`. The expected counts are sanity bounds for the filter audit,
from `configs/constructions.yaml`; the real count for each corpus is read at build time.

## How it works

**Data.** `drc.data.download` pulls BabyLM 2026 Strict-Small (the 10M-word training set) and
Strict (100M words, used only as a replacement pool) from the Hugging Face Hub. `drc.data.parse`
caches a Stanza dependency parse of both as CoNLL-U.

**Filters.** Eight dependency-pattern detectors in `drc.data.filters`. `drc.data.qa_audit`
samples 200 flagged and 200 unflagged sentences per construction for hand labels and checks
them against targets of 0.90 precision and 0.85 recall.

**Dose corpora.** `drc.data.dose_corpora` keeps d instances and replaces the rest with sentences
from the same domain within 20% of the original length (relaxing to 40%, then to nearest length).
It uses one fixed corpus seed (42), writes an audit JSON per corpus, and re-runs the filter to
confirm the counts.

**Training.** LTG-BERT (hidden size 384, 12 layers, 6 heads, 16,384-token vocabulary, 128-token
sequences), batch 256, AdamW at 1e-3 with 10% warmup and cosine decay, weight decay 0.1, masked
LM at 15% with whole-word masking, 14 epochs. The recipe is frozen across all runs. A pilot
(the full-corpus model, seed 42) must reach held-out perplexity under 40 before the sweep
continues. Seeds 42, 43, and 44.

**Evaluation.** SLOR from masked-LM pseudo-log-likelihood, corrected by a unigram model and
length. A pair counts as correct when the grammatical sentence scores higher. The same test runs
with an interpolated 4-gram model trained on each dose corpus, to check whether surface
statistics alone reproduce the curve.

**Analysis.** Hill fits per construction and seed with parametric-bootstrap intervals; AIC and
BIC against power, step, log-linear, and null curves; k-means over [E0, log E50, n, Emax] with
silhouette scores; the E0 index with a contamination bound from filter recall; and three
exploratory analyses (predicting E50 from corpus properties, a cross-construction transfer
matrix, seen versus novel slot fillers). `analysis_julia/` mirrors the core Hill fit.

## Run it

```bash
git clone https://github.com/keshavkrishnan08/grammar-scaling-laws
cd grammar-scaling-laws
pip install -e ".[dev]"
pytest                                                  # dependency-light tests, no GPU needed
python -m drc.pipeline --config configs/smoke.yaml      # whole pipeline on a tiny model, a wiring check
python -m drc.pipeline --config configs/base.yaml       # the real run, resumable stage by stage
```

`make pipeline` runs the same stages through their individual CLIs (`make data`, `make train`,
`make eval`, `make analyze`, `make figures`). For Kaggle, `notebooks/` holds a smoke notebook, two
session notebooks (data and training, then evaluation and analysis), and a one-session runner;
`scripts/` has dual-T4 and single-T4 launchers.

## Layout

```
src/drc/
  __init__.py          construction codes, dose levels, seeds
  design.py            tiered design: which (construction, dose, seed) runs exist
  pipeline.py          fault-tolerant runner: one subprocess per stage, resume, dependency gating
  data/                download, Stanza parse, 8 filters, QA audit, dose corpora
  tokenizer/           shared BPE tokenizer
  train/               LTG-BERT training and the pilot-gated sweep
  eval/                SLOR scoring, n-gram baseline
  analysis/            Hill fits, model comparison, clustering, E0 index, transfer,
                       predictability, generalization, decision rule, figures
configs/               base (frozen recipe and design), constructions, smoke
evaluation/eval_items/ 240 hand-written minimal pairs, 30 per construction
analysis_julia/        Julia mirror of the Hill fit
notebooks/             Kaggle notebooks, built from _build.py
scripts/               pipeline and Kaggle launchers
docs/                  PRD, methods, reproducibility checklist, literature audit
paper/                 manuscript and supplement
tests/                 filters, Hill math, n-gram model, pipeline, schema, notebooks
```

## Honesty

* Nothing has been run. No model is trained, no results are committed, and every number the
  paper will need is a visible placeholder.
* The novelty is narrower than "first exposure sweep". Oba et al. (2024) already injected
  phenomena at several counts. The claim here is the parametric fit and its comparable
  parameters, as recorded in `docs/research_audit.md`.
* The evaluation sets are small and author-built: 30 template-based pairs per construction.
  Published item sets (Mahowald 2023, SyntaxGym, BLiMP-Supplement) are suggested in
  `evaluation/README.md` but not included.
* The curves are thin. Core constructions get four doses plus a ceiling to fit four parameters,
  breadth constructions get two points and are not fit at all, and the predictability analysis
  has only four constructions.
* E0 is an upper bound. Filter recall below 1 leaks some instances into the dose-0 corpora; the
  code reports the bound as a caveat column instead of correcting for it.
* Training runs 14 epochs, cut from 20 to fit a single-GPU budget. The config also defines an
  AANN accuracy band for the pilot (0.55 to 0.75, to replicate Misra and Mahowald 2024), but no
  code checks it yet; the pilot gate is perplexity only.
* Some docs trail the code. The sweep docstring and the pipeline's `--single-gpu` help mention 60
  or 48 runs, and `evaluation/README.md` lists four constructions; the config and design module
  give 63 runs over eight.

MIT licensed.
