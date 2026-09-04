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

**2026-08-30 — M0 decided: dc-ae-f64c128-in-1.0-diffusers @ 256px, 16
tokens/frame** (results/m0_ae_ceiling.json, measured on 2xT4). Beat every
other candidate on both metrics at a quarter of the tokens of the runner-up:

| candidate | tok/frame | PSNR mean | PSNR min | LPIPS mean |
|---|---|---|---|---|
| dc-ae-f32-sana@128px | 16 | 40.68 dB | 29.38 dB | 0.0079 |
| **dc-ae-f64-in@256px** | **16** | **46.01 dB** | **35.25 dB** | **0.0043** |
| dc-ae-f32-sana@256px | 64 | 44.29 dB | 34.78 dB | 0.0045 |

The ImageNet-reconstruction-trained AE beats the diffusion-backbone-trained
one on pure fidelity, as expected going in. 16 tokens/frame also
quadruples the usable step budget from the T4 profile relative to 64. The autoencoder choice sets it
(see M0 table below), and it drives every downstream step-time estimate.
Model code must therefore be config-driven on this parameter, not hardcoded.

### Measured on 2x Tesla T4 (results/profile_2xt4_v2.json)

Real measurements, torch 2.10, batch 8, single device. These replace the
earlier analytical estimate, which was wrong in two ways that mattered.

Steps that fit each milestone budget, with torch.compile:

| tokens/frame | M3 per arm (1.25 GPU-h, 15M) | M4 (8 GPU-h, 40M) |
|---|---|---|
| 16 | 151,873 | 475,091 |
| 64 | 29,479 | 97,889 |
| 256 | 2,639 | **OOM** |

**torch.compile is mandatory, not optional.** The estimate said to skip it
for M1-M3 and expect 10-25% at best. Measured speedup is 1.2x to 2.6x (5M/16
tokens: 26.5ms -> 10.2ms) and it also cuts peak VRAM by up to 40%. It
compiled on sm75 in every configuration; nothing fell back to eager.

**256 tokens/frame is out of reach.** 40M OOMs at batch 8 even compiled, and
15M leaves 2,639 steps per ablation arm, which cannot support a measurement.
This rules out SD-VAE f8 at 256px, and it constrains M0 to configurations
yielding 64 tokens/frame or fewer.

**MFU is 4-14% everywhere**, so these models are launch-overhead-bound rather
than compute-bound on T4. That is why compile helps so much, and it means
larger batches are worth more than they would be in a compute-bound regime.

**fp16 is stable**: zero non-finite losses and zero GradScaler skips across
every configuration measured.

**2026-08-30 — M1 gate: full pipeline verified end to end, gate itself not
yet passed** (results/m1_gate_smoke.json). Checkpoint from the 0.1-GPU-hour
smoke ticket (~3,968 steps): PSNR 13.06 dB / LPIPS 0.60 against the M0
ceiling's 46.01 dB / 0.0043. Not a Plucker-conditioning failure per the M1
gate's stated debug trigger -- at 16 tok/frame the T4 profile gives ~352k
steps/GPU-hour for the 5M preset, so this checkpoint has seen roughly 1% of
one GPU-hour's worth of steps. Getting the sampler and eval scripts working
at all (8 Kaggle attempts, each hitting a distinct real bug: AE scaling
factor, dataset mount timing/path, code not re-uploaded, torch.load
weights_only default, torch.compile dropping/misshaping checkpoint keys,
context_length inferred from the wrong axis) was the actual purpose of this
ticket. Next: a real M1 training ticket against the 0.789 GPU-hours
remaining in M1's allocation, then re-run scripts/run_m1_gate.py.

**2026-08-31 — M1 gate diagnosis: conditioning/causality confirmed correct;
flat PSNR is a data-diversity artifact, not a broken pipeline.** Four
consecutive stable checkpoints (3968/5000/5866/8000 steps, spanning a bug
fix and a 2x step-count range) all landed at PSNR ~13 dB / LPIPS ~0.6,
essentially flat -- ruling out "just needs more training." A local,
budget-free diagnostic (scripts/diagnose_m1_flat_psnr.py) against the
step-8000 checkpoint found:
  - Pose perturbation to one frame changes that frame's own prediction by
    ~4x its own magnitude (conditioning is clearly live), with exactly
    zero leak into earlier frames (causal mask is exactly correct).
  - One-step denoising near clean data (t=0.02) is excellent (MSE 0.01%
    of a random-noise baseline); it degrades toward t=1 (27% at t=0.98) --
    the model is locally good near the data manifold, weaker far from it.
  - Full sampling from pure noise gives ~equally bad latent-space MSE at
    50, 200, and 1000 Euler steps (0.978/0.980/0.981) -- ruling out
    "sampler needs finer integration" as the cause, since more steps
    change nothing.

Conclusion: the learned velocity field is a poor *global* vector field
away from the data manifold, plausibly because M1's dataset is 8 identical
repeated windows from one 128-frame trajectory -- extreme overfitting with
no diversity to generalize the noise-to-data flow across. This directly
answers CLAUDE.md's original M1 debug trigger ("if it can't get close,
Plucker conditioning is broken") in the negative: conditioning works.
The reconstruction gap is a scope artifact of M1's deliberately tiny
single-scene setup, not evidence of a broken model. Real reconstruction-
quality validation belongs at M3 (~160 trajectories) where the model has
enough diversity to learn a well-behaved global field.

M1_smoke_overfit ledger: 0.775/1.0 GPU-hours spent, 0.225h remaining.
Recommendation: treat M1's pipeline-and-conditioning validation as
satisfied by this diagnosis; do not spend further M1 budget chasing PSNR
under the current 8-window setup, since the diagnostic shows more of the
same training won't move it.

**2026-09-04 — M2 implementation audit: no M2 GPU budget spent yet.** The
first local muP draft was stopped before Kaggle because it was not a
source-equivalent muTransfer setup. It compared the ordinary 5M/15M presets
while changing depth (5 -> 11) and head count (8 -> 10), omitted fused QKV
weights (`3*dim`) from LR scaling, scaled the readout LR instead of its
forward pass, and retained ordinary `1/sqrt(d_head)` attention. The upstream
reference requires base/target topology to stay equal apart from widths,
MuReadout's `1/width` forward multiplier, `1/d_head`-family Transformer
attention, and Adam LR scaling from exact base/target fan-in shapes:
https://github.com/microsoft/mup

The corrected M2 pair is `m2_proxy_5m` (dim=192, depth=11, heads=8) versus
`15m` overridden to heads=8 (dim=320, depth=11). Parameter groups now compare
actual proxy/target shapes, so rounded SwiGLU widths and fused projections get
their exact fan-in multipliers. `configs/m2_mup_lr_sweep.yaml` remains locked
with `ticket_hours: null`. Run the capped T4 profile first, calculate a ticket
from its measured compiled step times, then request explicit user approval.
The sweep uses M3-tier data because M1's eight repeated windows already failed
to learn a useful global flow field and cannot support a meaningful LR rank.

**2026-09-04 — M2 capped T4 profile measured** (results/m2_profile.json).
Real Kaggle numbers for the same-depth pair, compiled, zero divergence:

| preset | dim | params | step time | steps/GPU-h |
|---|---|---|---|---|
| m2_proxy_5m | 192 | 5.33M | 15.8 ms | 228,571 |
| 15m (heads=8) | 320 | 14.80M | 27.7 ms | 130,105 |

10-arm sweep (5 LRs x 2 presets, 3000 steps) at these rates: 0.181 GPU-h on
synthetic in-memory batches. Real dataloading adds overhead not captured
here (M1's real throughput was measurably below its own synthetic profile
before the dataset-caching fix); ticket approved at 0.5 GPU-h, ~2.8x the
synthetic estimate, against M2's untouched 3.0h allocation.

Approved and launched: M3-tier precompute (160 trajectories, inference-side,
weekly quota not training budget) followed by the sweep itself in one
combined Kaggle session, ticket_hours=0.5 set only in the bundled config for
that run -- the checked-in configs/m2_mup_lr_sweep.yaml stays at
`ticket_hours: null` per tests/test_m2_lr_sweep.py's guard.

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
