# PROGRESS

**Single next action:** M4 — run the generation baselines (`bench/m4_baselines.py`) on frozen test families.

| Milestone | Status | Evidence |
|---|---|---|
| M0 Bootstrap + prior work | done | docs/PRIOR_WORK.md, ENVIRONMENT.md, DECISIONS.md, CI (`make ci` green) |
| M1 Capture kit + metrology | done (synthetic) / awaiting-human (real) | capture/scale_card_A4.pdf, capture/CAPTURE_GUIDE.md, results/m1/metrology_synth.json (E-M1-1) |
| M2 Synthetic data engine + verifiers | done | results/m2/dataset_stats.json (E-M2-1), results/m2/verifier_agreement.json (E-M2-2) |
| M3 Features, mating, completion, tolerance | done (synthetic) / awaiting-human (real) | results/m3/inference.json (E-M3-1), results/m3/render.json (E-M3-2) |
| M4 Baselines | in-progress | |
| M5 SFT + verifier-RL | todo | |
| M6 Strength & printability optimization | todo | |
| M7 Print-test feedback loop | todo | |
| M8 App | todo | |
| M9 Real-world benchmark | todo (expected awaiting-human) | |
| M10 Integration, release, write-up | todo | |

## M4 plan
- Generators × inputs on frozen test families (180 parts) plus test_id (train families, held-out instances):
  - inputs: `spare` repr (caliper measurements + mating inference + tolerance engine) and `similarity` repr (measured original geometry copied).
  - generators: untuned Qwen2.5-Coder 1.5B and 7B (zero-shot, few-shot from train families), best-of-N (N=8) with primary verifiers on the *inferred* task (D-019), template oracle (upper bound).
  - geometry-similarity references: exact-original "perfect scan" (upper bound for any scan-to-mesh / image-to-3D method), visual-hull scan-to-mesh from synthetic photos.
- Accept: held-out-verifier fit/print/strength/editability rates with CIs per baseline.

## M5 plan
- SFT (mlx_lm LoRA), curriculum stage1 → stage2, on train families + generic compositions.
- GRPO on MLX with composite primary reward; KL to SFT reference; length penalty; exec-error reward 0.
- Ablations: −fit, −strength, −edit reward; SFT-only; RL-only (from base). Reward-hacking audit.

## M6–M10: see the milestone list in the build prompt; plans get written here as each starts.
