# BLOCKERS

| Date | Blocker | Fallback taken | Impact on claims |
|---|---|---|---|
| 2026-10-24 | No CUDA GPU; Apple M5 Pro (48 GB unified) | MLX LoRA + MLX GRPO on a ~1.5B model | Results are for a small model; scale-up claims are not supported |
| 2026-10-24 | No printer or real parts collected yet | All fit/strength results `simulated`; physical milestones `awaiting-human` | Physical fit@1/fit@2 cannot be claimed until the human steps are done |
| 2026-10-24 | No VLM / no hosted LLM API key | Metrology → structured representation → text-only code LLM | Image understanding is bounded by the hand-built metrology/feature pipeline |
| 2026-10-24 | Docker daemon not verified | Dockerfile provided; CI defined for GitHub Actions; local `make ci` used | Container reproducibility is untested locally |
