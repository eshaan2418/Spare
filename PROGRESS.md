# PROGRESS

**Single next action:** M2 — build the synthetic dataset engine (`spare/synth/build_dataset.py`: ≥50k train programs + repr text, frozen splits with checksums) and the verifier agreement benchmark.

| Milestone | Status | Evidence |
|---|---|---|
| M0 Bootstrap + prior work | in-progress | docs/PRIOR_WORK.md (94 verified entries), ENVIRONMENT.md, DECISIONS.md D-001..D-012, CI config, 50 tests green |
| M1 Capture kit + metrology | todo | |
| M2 Synthetic data engine + verifiers | in-progress | 12 families (spare/synth/families), all verifiers + held-out variants (spare/verifiers), tests/test_verifiers.py |
| M3 Features, mating, completion, tolerance | todo (tolerance engine started: spare/features/tolerance.py) | |
| M4 Baselines | todo | |
| M5 SFT + verifier-RL | todo | |
| M6 Strength & printability optimization | todo | |
| M7 Print-test feedback loop | todo | |
| M8 App | todo | |
| M9 Real-world benchmark | todo (expected awaiting-human) | |
| M10 Integration, release, write-up | todo | |

## M0 plan
- [x] Environment probe → ENVIRONMENT.md; fallback rungs → DECISIONS.md D-001..D-004
- [x] Prior-work review with verified citations → docs/PRIOR_WORK.md (spot-checked by hand: 2609.34494, 2610.26861, 2505.19713)
- [x] Repo skeleton, pyproject with pinned deps, Makefile, CI workflow, Dockerfile
- [x] HUMAN_TODO.md shopping list and collection guide
- [ ] `make ci` green locally (lint + mypy + tests) → then M0 done

## M2 plan (acceptance: ≥50k synthetic programs; verifier unit tests pass; primary vs held-out agreement reported)
- [x] Procedural families with programs, mating, fit specs, probes, strength, feature graph, breakage
- [x] Verifiers: geometry/validity, fit (OCCT), printability (geometric), strength (analytic), editability (static)
- [x] Held-out: fit (mesh + manifold), probes (mesh), printability (PrusaSlicer), strength (FEA), editability (functional)
- [x] Unit tests on known-good and adversarial parts
- [ ] Dataset builder (≥50k train programs), frozen splits with checksums
- [ ] Rendered photos on virtual scale card (needs M1 card)
- [ ] Verifier agreement benchmark → results/m2/
