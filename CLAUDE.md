# nanoWM

A pose-conditioned, autoregressive Diffusion Transformer with spatial memory,
in the style of World Labs' RTFM, at nano-scale (~40M params, 20 GPU-hours total).

**The deliverable is the measurement, not the model.** The question: which
frontier training tricks (REPA, Muon, Flow Matching, Diffusion Forcing, muP)
actually pay off at ~40M params / 20 GPU-hours, and by what factor. This is a
speedrun/ablation project (modded-nanogpt style), not a scaling project.

Secondary goal: hook the model into the Target-Bench eval harness (TUM-AVS,
ECCV'26, arXiv 2511.17792) and add the persistence metric it's missing.

## Hard constraints

- Hardware: Kaggle, 2x Tesla T4 (sm75 — no bf16, no FlashAttention-2).
- Session limit: 12h, hard cutoff. Checkpoint/resume is mandatory.
- Weekly quota: 30 GPU-hours (Kaggle), tracked separately from training budget.
- **Training budget: 10h wall-clock on 2xT4 = 20 GPU-hours, total, for the whole project.**
- Precision: fp16 AMP + GradScaler, NaN watchdog. Muon Newton-Schulz in fp32.
- Attention: `F.scaled_dot_product_attention`, memory-efficient backend (sm75-compatible).
- Storage: `/kaggle/working` capped at 20GB. Latents cached as a Kaggle dataset, never recomputed.

Inference (latent precompute, eval rollouts, VGGT decoding) does NOT count
against the 10h training budget, but DOES count against the Kaggle weekly
quota. Track both, separately.

## The budget ledger — the central rule of this project

- Every training run needs a Budget-Ticket approved by the user first:
  planned duration, purpose, expected result, abort criterion.
- Every run appends to `budget/ledger.jsonl`:
  `{run_id, start, end, gpu_seconds, milestone, purpose, result, seed, git_sha}`.
- `scripts/budget.py status` shows remaining budget.
  `scripts/budget.py check-ticket --milestone X --hours Y` refuses tickets
  that exceed remaining milestone or total budget.
- A run that exceeds its ticket is aborted. No extensions without a new ticket.

See `budget/allocation.yaml` for the milestone-by-milestone pre-allocation
(M1 0.5h / M2 1.5h / M3 2.5h / M4 4.0h / M5 1.0h / reserve 0.5h = 10.0h total).
If M3 needs more, it borrows from M4 — never from the total.

**Never start a GPU training run without explicit user approval.**

## Subagents (`.claude/agents/`)

Each has a narrow scope and must not write into another's files.

- `data-engineer` — procedural scenes, exact poses, frame/latent/DINOv2
  precompute, Kaggle dataset packaging. Writes `src/data/`, `scripts/preprocess/`.
  Must generate looping trajectories (return to start) — required for the
  persistence metric.
- `model-architect` — causal DiT, Plücker raymap conditioning, adaLN-single,
  RMSNorm, QK-norm, SwiGLU, rectified flow, diffusion forcing, KV-cache,
  frustum-overlap retrieval. Writes `src/model/`. Never launches training runs.
  Records exact param count in docstrings on every size/config change.
- `train-infra` — trainer, DDP, fp16 AMP/GradScaler, NaN watchdog,
  checkpoint/resume (must survive SIGTERM at 11:45h), EMA in fp32 on CPU,
  W&B logging, budget-ledger enforcement. Writes `src/train/`, `scripts/`, `budget/`.
- `t4-profiler` — micro-benchmarks only (step time, tokens/s, peak VRAM, MFU,
  torch.compile viability, fp16 stability), runs capped at 2 minutes, no code
  changes. Every new config gets profiled before it gets a budget ticket.
- `eval-harness` — Target-Bench integration (VGGT as world decoder), noise-floor
  calibration, revisit-PSNR, drift curves, camera-following error with scale
  alignment. Writes `src/eval/`. Establishes noise floor on ground-truth video
  before any model number is reported.
- `skeptic` — reviews before every gate. Looks for train/eval contamination,
  feature leakage, silently masked NaNs, metrics that look good on noise,
  single-seed ablations, effects within seed variance. Veto power at every
  gate. Report-only, writes no code.

Use `claude-sonnet-5` for all subagents.

## Repo layout

```
nanowm/
├── budget/{ledger.jsonl, allocation.yaml}
├── configs/            # one YAML per run, versioned
├── scripts/{budget.py, preprocess/, train.py}
├── src/{data,model,train,eval}/
├── tests/              # shape, determinism, resume tests
└── runs/               # checkpoints, logs (gitignored)
```

## Milestones and gates (numeric, gated by `skeptic`)

- **M0** (0h) — Autoencoder ceiling: compare DC-AE f32/f16, SD-VAE f8 at
  128px/256px. Gate: PSNR/LPIPS tabulated, one config chosen + justified,
  tokens/frame documented. This is the hard ceiling for everything after.
- **M0.5** — Infra: resume + determinism tests green (identical loss curves,
  identical result after SIGTERM+resume). Profiler reports step time/MFU for
  3 model sizes.
- **M1** (0.5h) — Overfit: 5M model, one scene, pose-conditioned novel-view
  synthesis. Gate: reconstruction near the M0 ceiling, else pose conditioning
  is broken — debug, don't proceed.
- **M2** (1.5h) — HP transfer: muP LR sweep at 5M, verify transfer to 15M.
  Gate: optimal LR ratio between 5M/15M matches muP prediction.
- **M3** (2.5h) — Ablations: REPA on/off, Muon vs AdamW, Rectified Flow vs
  DDPM, 2 seeds each, equal compute per arm (not equal steps). Gate: effect
  sizes reported with seed variance; `skeptic` checks effects exceed variance.
  **This is the actual result of the project** — be careful here.
- **M4** (4.0h) — Main run: best M3 config, ~40M params, one run, no mid-run
  retuning. Gate: revisit-PSNR and drift curve vs. M0 ceiling and eval noise floor.
- **M5** (1.0h) — Rollout finetune: self-forcing against exposure bias.
  Gate: drift curve at 500 steps measurably flatter than post-M4.

## Work rules

- One run, one ticket, one ledger entry. No exceptions.
- Profile before training — `t4-profiler` gives step time/VRAM for every new
  config before a ticket is written.
- Ablations need ≥2 seeds; single runs are labeled anecdotal.
- Config-driven: every run has a versioned YAML, no hyperparameters in code.
- Report all results budget-normalized ("revisit-PSNR X after Y GPU-minutes"),
  never as a bare number.
- When a design decision is unclear: ask, don't guess. The budget doesn't
  forgive dead ends.

## Decision log

Decisions made by the user; do not re-litigate without new evidence.

**2026-08-27 — M3 ablation matrix cut to 4 arms.** REPA on/off and Muon vs
AdamW, 2 seeds each (~2250 s/arm). Rectified Flow vs DDPM is dropped: it is
the axis with the strongest literature prior, so it carries the least
information per GPU-hour. Two effects measured cleanly beats three measured
inside the noise. Rectified flow remains the training objective; only the
*comparison against DDPM* is out.

**2026-08-27 — Causal DiT context length: 16 frames.** At 256 tokens/frame
this is a 4096-token sequence. Attention roughly doubles cost-per-token
relative to a naive 6N estimate — the earlier profiler pass treated
tokens/frame as the full sequence length and therefore underestimated. Real
step-count expectations: ~85k steps for M4, ~20k steps per M3 arm. Re-profile
against these numbers once model code exists.

**2026-08-27 — Renderer: verify moderngl EGL context on Kaggle first.** A
throwaway notebook check spends weekly quota, not training budget. A GL
context failure discovered during M0 is exactly the dead end the budget cannot
absorb. numpy/numba software rasterizer stays as the automatic fallback.

**2026-08-27 — DINOv2-small for REPA features.** ~200 KB/frame, ~4 GB total.
REPA's payoff is itself what M3 measures; paying double precompute for an
untested assumption is premature. Revisit only if M3 shows REPA matters and
feature quality is the suspected limiter.

**Proposed, NOT yet decided — M3 statistical power.** With 2 seeds per arm
and ~20k steps at 15M, effects will likely sit inside seed variance, and the
gate would then block for lack of power rather than lack of effect. Two
changes would raise power at zero extra budget, and eval/ should be built so
they stay possible:
  1. *Paired comparisons.* Run both arms of an axis on the same seed, data
     order, and init, and analyze the per-seed difference rather than the
     absolute values. This removes the dominant variance component before it
     enters the measurement.
  2. *Learning curves, not endpoints.* An endpoint is one data point per run;
     the curve is hundreds. Two curves separating over their whole length is
     far more defensible than a single final number, at identical compute.
Needs the user's sign-off before M3 is configured.

**Open, blocked on M0:** tokens per frame. The autoencoder choice sets it
(see M0 table below), and it drives every downstream step-time estimate.
Model code must therefore be config-driven on this parameter, not hardcoded.

### M0 autoencoder candidates → tokens per frame

| AE | 128 px | 256 px |
|---|---|---|
| DC-AE f32 | 16 | 64 |
| DC-AE f16 | 64 | 256 |
| SD-VAE f8 | 256 | 1024 |

Licensing note (hard project constraint): DC-AE is Apache-2.0 and
unproblematic. SD-VAE weights carry the relevant Stable Diffusion license —
verify before M0, not after.

## Explicitly out of scope

- Few-step distillation and the interactive WASD demo (needs its own budget;
  only after M5 with a new ticket).
- Real datasets (DL3DV-10K, RealEstate10K) — synthetic scenes with exact
  poses only; real data costs preprocessing time and introduces pose noise
  that would contaminate the ablations.
- Model scale beyond 40M — this measures training efficiency, not scaling.
- Semantic goal-conditioning (the actual Target-Bench task) — we use the
  harness, not the task it was built for.
- Training our own autoencoder — frozen, pretrained, done.
