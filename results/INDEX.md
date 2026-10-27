# results/INDEX

Every experiment: command, output path, one-line finding. Numbers only come from files listed here. Evidence stage is `simulated` unless marked `physical`.

| ID | Milestone | Command | Output | Finding |
|---|---|---|---|---|
| E-M2-1 | M2 | `make synth` (`python -m spare.synth.build_dataset --n 50000 --seed 0`) | `results/m2/dataset_stats.json`, `data/benchmarks/MANIFEST.json` | 49,989/50,000 train programs execute (11 exec failures dropped). Frozen splits: 200 test_id, 100 val (hook), 180 test (hinge_leaf, rack_clip, end_cap_rect); all execute. |
| E-M2-2 | M2 | `python -m bench.verifier_agreement --n 120 --seed 0` | `results/m2/verifier_agreement.json` | Primary vs held-out agreement over 600 programs (oracle + mutations). all_pass 0.90, κ 0.81. fit 0.975, κ 0.95. printability 0.91, κ 0.15 (imbalanced). strength 0.94, κ 0.62. All ≥ the pre-registered 0.80. |
| E-M1-1 | M1 | `python -m bench.metrology_synth --n 90 --seed 0` | `results/m1/metrology_synth.json` | Synthetic renders, unseen-family eval parts. Extent |err| median 0.56 mm (p90 1.63). Height |err| median 1.94 mm (p90 8.2; raised small features under-estimated). Vertical holes (train-family parts) median 1.2 mm. Conformal σ gives 1σ/2σ coverage 0.73/0.95 for extents. |
| E-M3-1 | M3 | `python -m bench.m3_inference` | `results/m3/inference.json` | Oracle generator, held-out fit verifier, unseen test families. Calipers + tolerance engine: fit 0.956 (known printer) / 0.783 (hidden printer). Copy measured geometry: 0.033 / 0.067. Copy + printer compensation, no mating reasoning: 0.078 / 0.156. Photo-only: 0.006. No mating item (clearance prior): 0.556 / 0.433. |
| E-M3-2 | M3 | `python -m bench.m3_render` | `results/m3/render.json` | From synthetic photos (unseen families): feature-type exact-set match 0.33 (hole-type detection weak: blind holes 0/1 recall, many false positives). Mating diameter from context photos: median |err| 0.34 mm raw, 0.02 mm after standard-size snapping (54% within 0.1 mm; tube OD not snappable, 1.9 mm). Round-part completion from a broken view: median |err| 0.90 mm vs 1.80 mm naive extent. |
