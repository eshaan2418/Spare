# DECISIONS

Dated log of design decisions and rationale. Newest last.

- **D-001 (2026-10-24) Compute rung.** Apple M5 Pro, 48 GB unified memory, no CUDA, no LLM API key. Training uses MLX (`mlx-lm`) LoRA on a local ~1.5B code model (Qwen2.5-Coder-1.5B-Instruct, 4-bit), with GRPO implemented on MLX. This is the "Fallback 1" rung: a smaller model, LoRA, RL feasible but small-scale. Claims about 7B-scale models are out of scope.
- **D-002 (2026-10-24) Vision rung.** No local VLM good enough at metrology. Use the **metrology pipeline → structured representation → text-only code LLM** path (Vision Fallback 1). Photos feed `metrology/` and `features/`; the generator consumes the feature graph.
- **D-003 (2026-10-24) Physical validation rung.** No printer or real parts were present at start, so physical validation is `awaiting-human`. Every reported fit/strength number is `simulated` until real parts arrive (see HUMAN_TODO.md).
- **D-004 (2026-10-24) Slicer.** PrusaSlicer 2.9.6 CLI installed via Homebrew cask. CuraEngine is unavailable.
- **D-005 (2026-10-24) Package layout.** The prompt's top-level dirs (`capture/`, `metrology/`, ...) live under a single Python package `spare/` to avoid import clashes (e.g. `features`, `app`). `data/`, `results/`, `paper/` and `bench` outputs stay at the repo root.
- **D-006 (2026-10-24) Family split (frozen).** There are 12 procedural families.
  - Train: spacer, knob_dshaft, cap_round, wheel_grooved, battery_cover, l_bracket, cable_clip, shelf_peg.
  - Val: hook.
  - Test: hinge_leaf, rack_clip, end_cap_rect.
  - Test families reuse the feature vocabulary (holes, pins, C-snaps, press plugs) in configurations never seen in training.
- **D-007 (2026-10-24) Simulated printing.** "Simulated fit" means design geometry plus a first-order FDM deviation model: wall growth `e_xy`, vertical-hole shrink `hole_c + hole_k/d`, horizontal-hole extra shrink, and z growth. Fit bands per fit class are in `spare/features/tolerance.py`.
  - The RL reward uses the default printer profile.
  - Held-out evaluation assigns each test part one of three fixed **virtual printers** (`vp_tight`, `vp_nominal`, `vp_loose`) unknown to the generator. This mimics not knowing the user's printer.
  - The band and deviation values are engineering assumptions, not measurements. M7 learns real compensation from physical prints.
- **D-008 (2026-10-24) Fit measurement.** Sample points on the *mating* functional surface, with normals pointing into the gap, and take the signed directional gap to the part. The primary verifier uses OCCT ray casting/classification plus an OCCT boolean for interference. The held-out verifier uses trimesh ray casting at a finer tessellation, tangentially jittered samples and a manifold3d boolean.
- **D-009 (2026-10-24) Strength verifiers.** The primary (reward) verifier uses analytic formulas: cantilever snap strain `1.5 t y / L^2` and cantilever bending stress `6FL/(bt^2)`, fed by thickness and deflection **measured on the generated solid** by ray probes rather than read from parameters. The held-out verifier is FEA (gmsh tetra mesh + scikit-fem linear elasticity). The material allowables are assumptions in `MATERIALS`.
- **D-010 (2026-10-24) Printability verifiers.** The primary verifier is geometric analysis (overhang area, min wall, bed contact, build volume) because it is fast. The held-out verifier is a real PrusaSlicer CLI slice (time, filament, errors). This inverts the prompt's suggested assignment for speed; the slicer stays independent of the reward.
- **D-011 (2026-10-24) Novelty position (from docs/PRIOR_WORK.md).** SFT followed by GRPO for CadQuery with geometric/IoU/Chamfer rewards **is prior work**: CAD-Coder (Guan et al., arXiv 2505.19713), cadrille, CAD-RL, ConCAD (2609.34494). Spare claims no novelty for "RL for CAD code". Its claimed contributions are narrowed to:
  1. function-first rewards (simulated fit against separate mating geometry under a printer model, printability, strength, editability) replacing similarity to a reference;
  2. inferring the mating counterpart from context photos, with fit-class tolerancing;
  3. mapping print-test fit reports to parameter corrections, with per-printer compensation.
  - Wording follows "we found no prior work that ...".
  - I spot-checked the citations for 2609.34494, 2610.26861 and 2505.19713 against the arXiv API.
- **D-012 (2026-10-24) Program sandbox.** Generated code is AST-screened (only cadquery/math imports, no dunders, no I/O builtins, no while loops) and run in a spawn-pool worker with SIGALRM plus a hard pool timeout. This is research-grade isolation, not a security boundary.
- **D-013 (2026-10-24) Unsigned local commits.** The global git config signs commits through 1Password `op-ssh-sign`, which blocks on an interactive approval. For this local repo, commits use `git -c commit.gpgsign=false` per invocation. The global config is not changed. History can be re-signed later (`git rebase --exec 'git commit --amend --no-edit -S' --root`).
- **D-014 (2026-10-24) Held-out slicer pass rule.** Fixed **before** evaluating any generator, calibrated only on oracle designs. A slice passes if it succeeds and auto-support (45°) is ≤15% of filament. PrusaSlicer "print stability" warnings, such as low bed adhesion, are a −0.1 score penalty, not a fail. Rationale: small horizontal holes and lips get auto-supports that are not needed in practice.
- **D-015 (2026-10-24) DfAM design rules in the tolerance engine.** Cosmetic geometry may change to make a part printable or strong. Grip-groove count is capped so teeth stay ≥1 mm. Snap-beam thickness, overlap and aperture are reduced until the analytic strain is under 0.9 × the material allowable. Press interference is capped by the hoop or wall strain limit. Bracket, hook and peg thickness is increased until bending stress ≤ yield/2.
