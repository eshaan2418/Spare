# ENVIRONMENT

Probed 2026-10-24 (update when it changes).

| Item | Value |
|---|---|
| Machine | Apple M5 Pro, 18 CPU cores, 48 GB unified memory, Metal 4 GPU (no CUDA) |
| OS | macOS 27.0.1 (26A434) |
| Disk | 926 GB, ~754 GB free |
| Python | 3.12.14 (uv venv `.venv`) |
| CAD kernel | CadQuery 2.8.0, cadquery-ocp 7.9.3.1.1 (OCCT 7.9.3) |
| Mesh / booleans | trimesh 5.1.1, manifold3d 3.5.4 |
| FEA | gmsh 4.15.2 + scikit-fem 12.0.2 (linear elasticity, P1/P2 tets) |
| Vision | OpenCV (contrib) 5.0.0 incl. ArUco/ChArUco |
| Slicer | PrusaSlicer 2.9.6 CLI (`/Applications/PrusaSlicer.app`); CuraEngine not available |
| ML | MLX 0.32.4, mlx-lm 0.32.0; base model `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` |
| LLM API | none (no ANTHROPIC/OPENAI/HF keys in env) |
| Docker | client 29.7.2 present; daemon not verified (Dockerfile provided) |
| Printer | **none yet** (awaiting human) |
| Real parts | collected 0 / photographed 0 / caliper-measured 0 / printed 0 |
