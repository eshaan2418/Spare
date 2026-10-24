# Prior Work: Literature and Product Review for Spare

Compiled 2026-10-24. Scope: work related to Spare, which turns phone photos of a broken or missing small plastic part (plus photos of where it attaches) into an editable parametric CadQuery program that is meant to fit and function once printed.

## How entries were verified

- **arXiv entries:** title, first author, author count and submission date were pulled from the arXiv export API (`export.arxiv.org/api/query?id_list=...`) on 2026-10-24. Venue is given only where the arXiv comment field, a CVF/ACM/ICLR/NeurIPS/ASME proceedings page, or the official repo states it.
- **Rewards and method details:** where these matter for the novelty analysis (the RL-for-CAD section), they were checked against the arXiv abstract or HTML full text.
- **Non-arXiv items:** standards, products and journal papers were confirmed through publisher, ISO, ACM, vendor or official documentation pages found by search.
- **Marking:** anything not confirmed is marked **UNVERIFIED** and is not used to support any claim. Partly confirmed items are marked *(partial)*, and the unconfirmed field is named.
- **Summaries:** these are short paraphrases of abstracts. Numbers are quoted only where they were read on the primary page.

---

## 1. CAD datasets, sketch-and-extrude sequence generation, and CAD from point clouds or images

| # | Work | Authors | Venue / year | URL |
|---|---|---|---|---|
| 1.1 | ABC: A Big CAD Model Dataset for Geometric Deep Learning | Koch et al. | CVPR 2019 | https://arxiv.org/abs/1812.06216 |
| 1.2 | SketchGraphs: A Large-Scale Dataset for Modeling Relational Geometry in CAD | Seff et al. | ICML 2020 Workshop on Object-Oriented Learning (per official repo) | https://arxiv.org/abs/2007.08506 |
| 1.3 | Fusion 360 Gallery: A Dataset and Environment for Programmatic CAD Construction from Human Design Sequences | Willis et al. | SIGGRAPH 2021 (ACM TOG) | https://arxiv.org/abs/2010.02392 |
| 1.4 | DeepCAD: A Deep Generative Network for Computer-Aided Design Models | Wu et al. | ICCV 2021 | https://arxiv.org/abs/2105.09492 |
| 1.5 | SkexGen: Autoregressive Generation of CAD Construction Sequences with Disentangled Codebooks | Xu et al. | ICML 2022 | https://arxiv.org/abs/2207.04632 |
| 1.6 | Hierarchical Neural Coding for Controllable CAD Model Generation (HNC-CAD) | Xu et al. | ICML 2023 | https://arxiv.org/abs/2307.00149 |
| 1.7 | ComplexGen: CAD Reconstruction by B-Rep Chain Complex Generation | Guo et al. | SIGGRAPH 2022 (ACM TOG) | https://arxiv.org/abs/2205.14573 |
| 1.8 | SECAD-Net: Self-Supervised CAD Reconstruction by Learning Sketch-Extrude Operations | Li et al. | CVPR 2023 | https://arxiv.org/abs/2303.10613 |
| 1.9 | Point2CAD: Reverse Engineering CAD Models from 3D Point Clouds | Liu et al. | CVPR 2024 | https://arxiv.org/abs/2312.04962 |
| 1.10 | CAD-SIGNet: CAD Language Inference from Point Clouds using Layer-wise Sketch Instance Guided Attention | Khan et al. | CVPR 2024 | https://arxiv.org/abs/2402.17678 |
| 1.11 | Text2CAD: Generating Sequential CAD Models from Beginner-to-Expert Level Text Prompts | Khan et al. | NeurIPS 2024 (Spotlight) | https://arxiv.org/abs/2409.17106 |
| 1.12 | Img2CAD: Reverse Engineering 3D CAD Models from Images through VLM-Assisted Conditional Factorization | You et al. | SIGGRAPH Asia 2025 (per arXiv comment) | https://arxiv.org/abs/2408.01437 |
| 1.13 | Img2CAD: Conditioned 3D CAD Model Generation from Single Image with Structured Visual Geometry | Chen et al. | arXiv 2024 | https://arxiv.org/abs/2410.03417 |
| 1.14 | CAD-Recode: Reverse Engineering CAD Code from Point Clouds | Rukhovich et al. | ICCV 2025 (per official repo) | https://arxiv.org/abs/2412.14042 |
| 1.15 | JoinABLe: Learning Bottom-up Assembly of Parametric CAD Joints | Willis et al. | CVPR 2022 | https://arxiv.org/abs/2111.12772 |
| 1.16 | AutoMate: A Dataset and Learning Approach for Automatic Mating of CAD Assemblies | Jones et al. | arXiv 2021 (SIGGRAPH Asia 2021, venue not re-checked) | https://arxiv.org/abs/2105.12238 |

**Summaries and how each relates to Spare**

- **1.1 ABC:** About one million CAD models from Onshape, with B-rep and parametric data, for geometric deep learning. *Relation:* a possible source of real mechanical-part geometry for Spare's part-family priors and synthetic SFT data. It contains no mating or fit labels.
- **1.2 SketchGraphs:** Millions of real Onshape sketches, stored as geometric-constraint graphs. *Relation:* a source of constraint structure such as coincident, concentric and equal, which Spare's editability reward and parameter naming could exploit.
- **1.3 Fusion 360 Gallery:** Human sketch-and-extrude design sequences, plus a gym environment for programmatic CAD reconstruction. *Relation:* a standard benchmark for reconstruction-style generation. Spare's target is fit, not reconstruction of a reference.
- **1.4 DeepCAD:** A transformer autoencoder over sketch-and-extrude command sequences, released with a roughly 178k-model dataset. *Relation:* the canonical command-sequence representation that CadQuery-code approaches replace with executable code.
- **1.5 SkexGen:** Autoregressive CAD sequence generation, with codebooks that separate topology from geometry. *Relation:* sequence-level controllability. It has no conditioning on photos or mating geometry.
- **1.6 HNC-CAD:** Hierarchical codes (loop, profile, solid) that allow controllable generation and autocompletion of CAD models. *Relation:* its autocompletion is the closest sequence-model analogue to Spare's "complete broken regions" step.
- **1.7 ComplexGen:** Reconstructs B-rep chain complexes (corners, curves, patches) directly from point clouds. *Relation:* scan-to-B-rep output that is not a parametric or editable program.
- **1.8 SECAD-Net:** Self-supervised reconstruction of compact, editable sketch-extrude CAD from voxels or points. *Relation:* puts editability first, as Spare does. It has no function or fit objective.
- **1.9 Point2CAD:** Hybrid analytic and neural surface fitting from segmented point clouds into CAD topology. *Relation:* the geometric reverse-engineering baseline Spare would face if it took a scan rather than photos.
- **1.10 CAD-SIGNet:** An autoregressive model that infers sketch-and-extrude design history from point clouds. *Relation:* point cloud to editable history, without fit.
- **1.11 Text2CAD:** Text prompts at several expertise levels, mapped to DeepCAD-style sequences. *Relation:* conditioning on language, not on images or context.
- **1.12 and 1.13 Img2CAD (two different papers with the same name):** Single image to CAD. You et al. use a VLM to predict structure and then regress parameters. Chen et al. condition generation on structured visual geometry. *Relation:* the most direct "photo to parametric CAD" prior art. Neither uses mating context or fit.
- **1.14 CAD-Recode:** A Qwen2-1.5B model with a point-cloud projector that outputs CadQuery sketch-extrude scripts. It is trained on one million procedurally generated programs. *Relation:* a strong prior for "LLM emits CadQuery" and for procedural synthetic SFT data, which Spare's plan also uses.
- **1.15 JoinABLe:** Predicts joints between pairs of B-rep parts. *Relation:* learned mating between parts. It predicts how existing parts join and does not generate a new part with clearances.
- **1.16 AutoMate:** A dataset and model for predicting mates between CAD parts. *Relation:* the same as JoinABLe. Mates are ranked or predicted, but no fitting geometry is synthesized.

## 2. LLM/VLM generation of CAD code (CadQuery, OpenSCAD, FreeCAD, Blender) and code verification or refinement

| # | Work | Authors | Venue / year | URL |
|---|---|---|---|---|
| 2.1 | Generating CAD Code with Vision-Language Models for 3D Designs (CADCodeVerify, CADPrompt) | Alrashedy et al. | ICLR 2025 | https://arxiv.org/abs/2410.05340 |
| 2.2 | CAD-Coder: An Open-Source Vision-Language Model for Computer-Aided Design Code Generation | Doris et al. | ASME IDETC-CIE 2025 | https://arxiv.org/abs/2505.14646 |
| 2.3 | CAD-Assistant: Tool-Augmented VLLMs as Generic CAD Task Solvers | Mallis et al. | ICCV 2025 | https://arxiv.org/abs/2412.13810 |
| 2.4 | CADTalk: An Algorithm and Benchmark for Semantic Commenting of CAD Programs | Yuan et al. | CVPR 2024 | https://arxiv.org/abs/2311.16703 |
| 2.5 | Query2CAD: Generating CAD models using natural language queries | Badagabettu et al. | arXiv 2024 | https://arxiv.org/abs/2406.00144 |
| 2.6 | BlenderLLM: Training LLMs for CAD with Self-improvement | Du et al. | arXiv 2024 | https://arxiv.org/abs/2412.14203 |
| 2.7 | LLM4CAD: Multimodal LLMs for Three-Dimensional CAD Generation | Li, Sun, Sha | ASME J. Comput. Inf. Sci. Eng. 25(2), 2025 (IDETC 2024 earlier) | https://doi.org/10.1115/1.4067085 |
| 2.8 | Text-to-CAD Generation Through Infusing Visual Feedback in LLMs (CADFusion) | Wang et al. | ICML 2025 | https://arxiv.org/abs/2501.19054 |
| 2.9 | CAD-Llama: Leveraging LLMs for CAD Parametric 3D Model Generation | Li et al. | arXiv 2025 | https://arxiv.org/abs/2505.04481 |
| 2.10 | CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM | Xu et al. | arXiv 2024 | https://arxiv.org/abs/2411.04954 |
| 2.11 | Text-to-CadQuery: A New Paradigm for CAD Generation with Scalable Large Model Capabilities | Xie et al. | arXiv 2025 | https://arxiv.org/abs/2505.06507 |
| 2.12 | Seek-CAD: A Self-refined Generative Modeling for 3D Parametric CAD Using Local Inference via DeepSeek | Li et al. | ICLR 2026 (per arXiv comment) | https://arxiv.org/abs/2505.17702 |
| 2.13 | CADEvolve: Creating Realistic CAD via Program Evolution | Elistratov et al. | arXiv 2026 | https://arxiv.org/abs/2602.16317 |
| 2.14 | CADReasoner: Iterative Program Editing for CAD Reverse Engineering | Kabisov et al. | Findings of CVPR 2026 (per arXiv comment) | https://arxiv.org/abs/2603.29847 |
| 2.15 | IterCAD: An Iterative Multimodal Agent for Visually-Grounded CAD Generation and Editing | Hu et al. | arXiv 2026 | https://arxiv.org/abs/2606.13368 |
| 2.16 | Text-to-CAD Evaluation with CADTests | Mallis et al. | arXiv 2026 | https://arxiv.org/abs/2605.07807 |
| 2.17 | Text2CAD-Bench: A Benchmark for LLM-based Text-to-Parametric CAD Generation | Wang et al. | arXiv 2026 | https://arxiv.org/abs/2605.18430 |
| 2.18 | BenchCAD: A Comprehensive, Industry-Standard Benchmark for Programmatic CAD | Zhang et al. | arXiv 2026 | https://arxiv.org/abs/2605.10865 |
| 2.19 | Embodied CAD: Solver-Grounded LLM Agents for Parametric B-Rep Assembly Modeling | Liu et al. | arXiv 2026 | https://arxiv.org/abs/2606.31252 |
| 2.20 | Can It Actually Work? Synthesizing Functional CAD Mechanisms (FunCAD-Harness, FunCADBench) | Zhang et al. | arXiv 2026 (19 Oct 2026) | https://arxiv.org/abs/2610.26861 |
| 2.21 | CadQuery (library) | CadQuery project | Software, OCCT-based | https://github.com/CadQuery/cadquery |

**Summaries and how each relates to Spare**

- **2.1 CADCodeVerify:** A VLM writes and answers yes/no verification questions about renders of the generated CAD, and the answers become feedback for refining the code. The paper also introduces the 200-object CADPrompt benchmark. *Relation:* inference-time verification with a VLM judge. Spare instead uses programmatic, physics- and fit-based verifiers as RL rewards.
- **2.2 CAD-Coder (Doris et al., MIT):** A LLaVA-style VLM fine-tuned on the GenCAD-Code dataset (more than 163k image and CadQuery pairs). It generates CadQuery from images and shows some transfer to real photos. *Relation:* the closest SFT-only "image to CadQuery" prior. It has no RL, no mating context and no fit.
- **2.3 CAD-Assistant:** A VLLM planner that calls FreeCAD Python tools zero-shot and adapts to the current design state. *Relation:* tool-using agents as an alternative to a trained policy.
- **2.4 CADTalk:** Semantic commenting of CAD programs, with a benchmark. *Relation:* relevant to Spare's editability goal of human-readable, named parameters.
- **2.5 Query2CAD:** An LLM generates CAD macros, with self-refinement driven by captioned renders. *Relation:* a refinement loop without training.
- **2.6 BlenderLLM:** An LLM fine-tuned with self-improvement to write Blender Python scripts. *Relation:* code-as-CAD in a mesh tool, which does not give parametric solids.
- **2.7 LLM4CAD:** Studies GPT-4 and GPT-4V generating CAD programs for five mechanical part families (shafts, nuts, flanges, springs, gears), with a debugger for syntax errors. *Relation:* uses part-family evaluation on mechanical parts, as Spare does. It has no fit criterion.
- **2.8 CADFusion:** Alternates sequence SFT with a visual-feedback stage that scores rendered outputs and applies preference optimization. *Relation:* an early "feedback beyond token likelihood" approach. The feedback is visual preference, not geometric verification.
- **2.9 CAD-Llama and 2.10 CAD-MLLM:** LLMs and MLLMs adapted to parametric CAD sequence generation from text and multimodal inputs. *Relation:* general multimodal CAD generation baselines.
- **2.11 Text-to-CadQuery:** Fine-tunes LLMs to emit CadQuery directly from text. *Relation:* supports choosing CadQuery as the output language.
- **2.12 Seek-CAD:** Training-free DeepSeek-R1 generation with VLM-checked step-wise renders and self-refinement. The authors release a dataset that includes fillets, chamfers and shells. *Relation:* inference-time refinement.
- **2.13 CADEvolve:** VLM-guided evolution of CadQuery programs, from primitives up to complex parts (8k generators, 1.3M scripts). It reports state of the art on image-to-CAD. *Relation:* a source and technique for richer synthetic CadQuery SFT data. A secondary search snippet says it uses online RL with an IoU reward. The arXiv abstract does not mention this, so treat it as **UNVERIFIED**.
- **2.14 CADReasoner and 2.15 IterCAD:** Iterative program editing agents for CAD reverse engineering and editing. IterCAD argues that point-cloud metrics capture only global discrepancies and cannot localize faulty parameters. *Relation:* parameter-level correction, which is related to Spare's fit-report-to-parameter mapping. Their corrections are driven by geometry, not by physical print tests.
- **2.16 CADTests:** Executable software tests that check whether a generated CAD model meets a prompt's geometric and topological requirements. The authors also show the tests can guide generation. *Relation:* conceptually the closest work to Spare's "critical-dimension match" verifier. The abstract does not say CADTests is used as an RL reward.
- **2.17 Text2CAD-Bench and 2.18 BenchCAD:** Benchmarks for programmatic CAD. Both explicitly list multi-part assemblies, mate relationships, or manufacturing tolerances as out of scope or future work. *Relation:* direct evidence that fit and tolerance are not covered by current CAD-LLM benchmarks.
- **2.19 Embodied CAD:** LLM agents grounded in a solver for parametric B-rep assembly modeling. Its scope excludes tolerance-stack analysis and manufacturing simulation, per its paper as surfaced by search. *Relation:* assembly-aware generation without clearance or tolerance.
- **2.20 FunCAD-Harness:** Converts functional requirements into candidate-specific physics-simulation tests and uses the results to revise CAD mechanisms. It also introduces FunCADBench (46 mechanism families). *Relation:* the closest prior on "function-first" CAD generation. It works at inference time in simulation. The abstract mentions no RL training, no fabrication or printing, and no photo-to-replacement setting.
- **2.21 CadQuery:** A Python parametric CAD library on OpenCascade (OCCT) with STEP import and export. *Relation:* Spare's output language.

## 3. Reinforcement learning with geometric or execution rewards for CAD generation (closest prior art)

| # | Work | Authors | Venue / year | Output | RL and reward (as verified) | URL |
|---|---|---|---|---|---|---|
| 3.1 | CAD-Coder: Text-to-CAD Generation with Chain-of-Thought and Geometric Reward | Guan et al. | NeurIPS 2025 | CadQuery | SFT then GRPO. Reward is Chamfer-distance geometric plus format. | https://arxiv.org/abs/2505.19713 |
| 3.2 | cadrille: Multi-modal CAD Reconstruction with (Online) Reinforcement Learning | Kolodiazhnyi et al. | ICLR 2026 (Oral) | Python CAD code (CadQuery lineage of CAD-Recode) | SFT on procedural data, then online RL (GRPO), which beats offline alternatives. The reward is computed automatically. | https://arxiv.org/abs/2505.22914 |
| 3.3 | From Intent to Execution: Multimodal CoT RL for Precise CAD Code Generation (CAD-RL, ExeCAD) | Niu et al. | arXiv 2025 | CadQuery | R = R_exec · (λ·IoU + λ·R_eval). R_exec is binary executability, R_eval is a GPT-4o evaluator. Future work lists "real-world manufacturing constraints". | https://arxiv.org/abs/2508.10118 |
| 3.4 | GACO-CAD: Geometry-Augmented and Conciseness-Optimized CAD Model Generation from Single Image | Wang et al. | arXiv 2025 | CAD code | SFT with depth and normal maps, then RL with a group length (conciseness) reward. | https://arxiv.org/abs/2510.17157 |
| 3.5 | ReCAD: RL Enhanced Parametric CAD Model Generation with VLMs | Li et al. | AAAI 2026 (Oral, per arXiv) | Parametric code | SFT then RL with a unified reward for "geometric accuracy and semantic fidelity". | https://arxiv.org/abs/2512.06328 |
| 3.6 | CME-CAD: Heterogeneous Collaborative Multi-Expert RL for CAD Code Generation | Niu et al. | arXiv 2025 | CadQuery | Multi-expert SFT then multi-expert RL. The abstract does not give the reward design. | https://arxiv.org/abs/2512.23333 |
| 3.7 | STEP-LLM: Generating CAD STEP Models from Natural Language with LLMs | Shi et al. | DATE 2026 | STEP files | RL with a Chamfer-distance reward. Lists manufacturability and constraint checks in the reward as future work. | https://arxiv.org/abs/2601.12641 |
| 3.8 | FutureCAD: High-Fidelity CAD Generation via LLM-Driven Program Generation and Text-Based B-Rep Primitive Grounding | Li et al. | arXiv 2026 | CadQuery | SFT then RL "to improve generalization". The abstract does not give the reward. | https://arxiv.org/abs/2603.11831 |
| 3.9 | ToolCAD: Tool-Using LLMs in Text-to-CAD Generation with RL | Gong et al. | ACL 2026 (per arXiv comment) | FreeCAD tool calls | Curriculum RL with CAD-engine feedback. | https://arxiv.org/abs/2604.07960 |
| 3.10 | Ortho2CAD: 3D CAD generation from orthographic drawings using VLMs | Joglekar et al. | arXiv 2026 | CadQuery | SFT where labels exist. Otherwise "geometry-grounded" RL with an IoU reward. Also inference-time self-refinement. | https://arxiv.org/abs/2607.08891 |
| 3.11 | ConCAD: Constraint-Aware Image-to-CAD Generation with Dual-Granularity Rewards | Zhai et al. | arXiv 2026 (28 Sep 2026) | CadQuery | GRPO. R = 0.1·R_exec (executes and has positive volume) + 0.3·R_con (regex match of 8 constraint relations against the ground-truth code) + 1.0·R_iou (alignment-max volumetric IoU). The full text does not address printability, fit, mating or strength. | https://arxiv.org/abs/2609.34494 |
| 3.12 | RLCAD: RL Training Gym for Revolution Involved CAD Command Sequence Generation | Yin et al. | arXiv 2025, Computer-Aided Design (journal) | Command sequences (B-rep gym) | Policy RL in a CAD gym, with a reward based on distance to the target geometry. | https://arxiv.org/abs/2503.18549 |

**How this section relates to Spare.** This literature already covers one combination in depth: an LLM or VLM emitting CadQuery or other CAD code, trained by SFT and then RL (often GRPO), with automatic rewards for executability or validity (3.1, 3.3, 3.11), geometric similarity to a ground-truth shape (Chamfer distance in 3.1 and 3.7; IoU in 3.3, 3.10 and 3.11), code-level constraint relations (3.11), conciseness (3.4), and LLM-judge design intent (3.3). Spare's RL stage as a technique (SFT then GRPO on CadQuery, with an execution-validity reward and a geometric-match reward) is therefore **not new**.

Every reward we verified scores similarity to a reference shape or reference program. None of the verified papers scores the generated part against separate mating geometry, slicer printability, a strength proxy, or a tolerance band per fit class. Two papers (3.3 and 3.7) explicitly list manufacturing constraints as future work.

## 4. Image-to-3D generative models (mesh or radiance outputs, not parametric CAD)

| # | Work | Authors | Venue / year | URL |
|---|---|---|---|---|
| 4.1 | Zero-1-to-3: Zero-shot One Image to 3D Object | Liu et al. | arXiv 2023 (venue not re-checked) | https://arxiv.org/abs/2303.11328 |
| 4.2 | One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization | Liu et al. | arXiv 2023 (venue not re-checked) | https://arxiv.org/abs/2306.16928 |
| 4.3 | LRM: Large Reconstruction Model for Single Image to 3D | Hong et al. | ICLR 2024 | https://arxiv.org/abs/2311.04400 |
| 4.4 | TripoSR: Fast 3D Object Reconstruction from a Single Image | Tochilkin et al. | Tech report 2024 | https://arxiv.org/abs/2403.02151 |
| 4.5 | InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view LRMs | Xu et al. | Tech report 2024 | https://arxiv.org/abs/2404.07191 |
| 4.6 | SF3D: Stable Fast 3D Mesh Reconstruction with UV-unwrapping and Illumination Disentanglement | Boss et al. | arXiv 2024 | https://arxiv.org/abs/2408.00653 |
| 4.7 | Structured 3D Latents for Scalable and Versatile 3D Generation (TRELLIS) | Xiang et al. | CVPR 2025 | https://arxiv.org/abs/2412.01506 |
| 4.8 | Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation | Zhao et al. (Tencent) | Tech report 2025 | https://arxiv.org/abs/2501.12202 |

- **What they do:** Zero-1-to-3 learns viewpoint-conditioned novel-view synthesis. One-2-3-45, InstantMesh and LRM lift those views, or a single image, into meshes through feed-forward reconstruction. TripoSR and SF3D are fast single-image mesh reconstructors, and SF3D adds UV unwrapping and material estimation. TRELLIS and Hunyuan3D 2.0 are large-scale latent generative models that output textured meshes, Gaussians or radiance fields.
- **Relation to Spare:** these are the "looks similar" baseline Spare argues against. Their outputs are meshes with no metric scale, no editable parameters, and no guarantee on hole diameters, shaft flats or snap geometry. They remain useful as a shape prior or a visual-hull initializer, and as a baseline in Spare's evaluation (image to mesh to print, versus image to CadQuery to print). Consumer tools built on this class of model (Meshy, Hyper3D, PrintReady3D, found by search) market photo-to-STL for printing. Hyper3D's own use-case page says it does not hold tight engineering tolerances. These vendor pages were seen only in search results, so treat the vendor details as **UNVERIFIED**.

## 5. Scan-to-CAD, photogrammetry, and reverse-engineering tools

| # | Tool / paper | Authors / vendor | Venue / year | URL |
|---|---|---|---|---|
| 5.1 | COLMAP; "Structure-from-Motion Revisited" | Schönberger & Frahm | CVPR 2016; BSD software | https://colmap.github.io/ |
| 5.2 | AliceVision Meshroom: An open-source 3D reconstruction pipeline | Griwodz et al. | ACM MMSys 2021 (Best Open Dataset & Software Paper) | https://doi.org/10.1145/3458305.3478443 |
| 5.3 | Apple Object Capture (RealityKit `PhotogrammetrySession`) | Apple | Developer documentation | https://developer.apple.com/documentation/realitykit/photogrammetrysession |
| 5.4 | Polycam | Polycam Inc. | Product (photogrammetry and LiDAR capture) | https://poly.cam |
| 5.5 | Geomagic Design X | Hexagon / resellers | Commercial scan-to-CAD (LiveTransfer of feature tree to SOLIDWORKS, NX, Inventor) | https://gomeasure3d.com/geomagic-design-x-software/ |
| 5.6 | QuickSurface | QuickSurface | Commercial scan-to-CAD (incl. SOLIDWORKS add-in) | https://tetravision.be/en/reverse-engineering-software/ (reseller page) |
| 5.7 | Backflip AI scan-to-CAD foundation model | Backflip | Announced Mar 2025 (news) | https://3dprintingindustry.com/news/new-ai-model-from-backflip-accelerates-3d-scan-to-cad-237055/ |
| 5.8 | Zoo (formerly KittyCAD) Text-to-CAD / ML-ephant API | Zoo | Product, launched 2024 | https://zoo.dev/machine-learning-api |

- **What they do:** COLMAP and Meshroom are open SfM/MVS pipelines that turn unordered photos into point clouds and meshes. Object Capture and Polycam are consumer and pro capture products that produce meshes, with LiDAR on supported iPhones. Design X and QuickSurface are interactive, expert-driven tools that fit features and surfaces to scan meshes and produce editable CAD. Backflip has announced an AI model that converts scan data into native CAD parts. Its availability was not confirmed. Zoo provides text-to-CAD generation as an API.
- **Relation to Spare:** this is the current professional workflow, scan then mesh then manual feature fitting in reverse-engineering software. It assumes the part is intact (or a scanner is available) and a skilled operator. Photogrammetry of small, glossy, featureless plastic parts is hard, and none of these tools reasons about the mating counterpart or adds fit clearance. Spare could use COLMAP or Object Capture only for metric scale and rough geometry. Its contribution would be the fit-aware, function-first program on top.

## 6. Design for additive manufacturing, FDM accuracy, tolerances and snap-fits

| # | Work | Authors | Venue / year | URL |
|---|---|---|---|---|
| 6.1 | Design for Additive Manufacturing: Trends, opportunities, considerations, and constraints | Thompson et al. | CIRP Annals 65(2):737–760, 2016 | https://doi.org/10.1016/j.cirp.2016.05.004 |
| 6.2 | ISO/ASTM 52910:2018 Additive manufacturing — Design — Requirements, guidelines and recommendations | ISO/ASTM | Standard, 2018 | https://www.iso.org/standard/67289.html |
| 6.3 | ISO 286-1:2010 GPS — ISO code system for tolerances on linear sizes — Part 1: Basis of tolerances, deviations and fits | ISO | Standard, 2010 (current) | https://www.iso.org/standard/45975.html |
| 6.4 | Snap-Fit Joints for Plastics — A Design Guide | Bayer MaterialScience LLC | Industry guide (copy hosted on MIT course site) | https://fab.cba.mit.edu/classes/S62.12/people/vernelle.noel/Plastic_Snap_fit_design.pdf |
| 6.5 | Design Guidelines for Additive Manufactured Snap-Fit Joints | Klahn, Singer, Meboldt | Procedia CIRP 50:264–269, 2016 | https://www.sciencedirect.com/science/article/pii/S2212827116303213 |
| 6.6 | Investigations on Factors Affecting 3D-Printed Holes Dimensional Accuracy and Repeatability *(partial: authors not confirmed)* | UNVERIFIED authors | Applied Sciences (MDPI) 13(1):41, 2023 | https://www.mdpi.com/2076-3417/13/1/41 |
| 6.7 | Optimal offline compensation of shape shrinkage for three-dimensional printing processes | Huang et al. | IIE Transactions 47(5):431–441, 2015 | https://doi.org/10.1080/0740817X.2014.955599 |
| 6.8 | In-Plane Shape-Deviation Modeling and Compensation for Fused Deposition Modeling Processes *(partial: author list not confirmed)* | UNVERIFIED authors | IEEE T-ASE, vol. 14, 2017 | https://doi.org/10.1109/TASE.2016.2544941 |
| 6.9 | GraphCompNet: A Position-Aware Model for Predicting and Compensating Shape Deviations in 3D Printing | Lee et al. | IEEE T-ASE (per arXiv comment), 2025 | https://arxiv.org/abs/2502.09652 |
| 6.10 | Understanding Uncertainty in Measurement and Accommodating its Impact in 3D Modeling and Printing | Kim et al. | ACM DIS 2017 | https://cmu-devlab.github.io/research/measurement-uncertainty/ |

- **6.1 and 6.2:** DfAM surveys and standards that frame design opportunities and constraints such as overhangs, minimum feature size and orientation. *Relation:* the source of Spare's printability rules. ISO/ASTM 52910 gives general guidance only, with no process- or material-specific numbers, so Spare's per-printer clearance values must come from calibration.
- **6.3 ISO 286-1:** Defines tolerance grades (IT01 to IT18), fundamental deviations, and clearance, transition and interference fits for holes and shafts. *Relation:* the vocabulary for Spare's "fit class" (clearance, transition, press). ISO numeric bands target machined parts and are far tighter than typical FDM capability, so Spare would map fit classes to printer-specific clearances, not to raw IT grades.
- **6.4 and 6.5:** The Bayer guide covers permissible strain, mating-force calculations, and cantilever, torsional and annular snap geometry. Klahn et al. adapt snap-fit rules to AM, in particular decoupling the joining direction from the orientation of the flexible feature. *Relation:* the basis for Spare's simplified snap-hook strength and deflection check.
- **6.6:** For material-extrusion holes, the authors find layer thickness and shell count to be the dominant factors in diameter accuracy, and inclined holes print undersized. *Relation:* empirical support for Spare's hole-shrinkage compensation.
- **6.7 to 6.9:** Model-based and learned compensation of printed shape deviation. Huang et al. (2015) work on SLA. The T-ASE paper (6.8) addresses FDM in-plane deviation. GraphCompNet learns position-aware deviation for powder-bed fusion. *Relation:* prior art for Spare contribution (3)'s "learned per-printer compensation". These methods learn from measured scans of printed geometry, not from qualitative fit reports, and they act on the mesh or contour rather than on named CAD parameters.
- **6.10:** Studies how end users' measurement errors (length, angle, diameter) make 3D-printed parts fail to fit, often over several iterations. The authors propose "accommodating measurement error" through buffer insertion and reprinting minimal parts. *Relation:* the closest HCI prior on the fit-iteration problem Spare's print-test loop targets. It is a design-principle and tool study, not learned correction.

## 7. RL from verifiers and execution feedback for code, GRPO, RLVR, and reward hacking

| # | Work | Authors | Venue / year | URL |
|---|---|---|---|---|
| 7.1 | CodeRL: Mastering Code Generation through Pretrained Models and Deep RL | Le et al. | NeurIPS 2022 | https://arxiv.org/abs/2207.01780 |
| 7.2 | Execution-based Code Generation using Deep RL (PPOCoder) | Shojaee et al. | TMLR 2023 | https://arxiv.org/abs/2301.13816 |
| 7.3 | RLEF: Grounding Code LLMs in Execution Feedback with RL | Gehring et al. | arXiv 2024 | https://arxiv.org/abs/2410.02089 |
| 7.4 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open LMs (introduces GRPO) | Shao et al. | arXiv 2024 | https://arxiv.org/abs/2402.03300 |
| 7.5 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL | DeepSeek-AI | arXiv 2025 | https://arxiv.org/abs/2501.12948 |
| 7.6 | Tulu 3: Pushing Frontiers in Open Language Model Post-Training (RLVR) | Lambert et al. | arXiv 2024 | https://arxiv.org/abs/2411.15124 |
| 7.7 | Concrete Problems in AI Safety | Amodei et al. | arXiv 2016 | https://arxiv.org/abs/1606.06565 |
| 7.8 | The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models | Pan et al. | ICLR 2022 | https://arxiv.org/abs/2201.03544 |
| 7.9 | Defining and Characterizing Reward Hacking | Skalse et al. | arXiv 2022 (NeurIPS 2022, venue not re-checked) | https://arxiv.org/abs/2209.13085 |
| 7.10 | Scaling Laws for Reward Model Overoptimization | Gao et al. | arXiv 2022 | https://arxiv.org/abs/2210.10760 |
| 7.11 | Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation | Baker et al. | arXiv 2025 | https://arxiv.org/abs/2503.11926 |

- **7.1 to 7.3:** RL for code with unit-test or execution rewards. CodeRL uses an actor-critic with a learned critic. PPOCoder uses PPO with compiler feedback and structural alignment. RLEF trains multi-turn repair from execution feedback. *Relation:* Spare's verifiers play the role of unit tests, with a CAD kernel, slicer and fit simulator as the "interpreter".
- **7.4 to 7.6:** GRPO (critic-free, group-relative advantages) was introduced in DeepSeekMath. DeepSeek-R1 scaled RL with rule-based verifiable rewards. Tulu 3 named the RLVR recipe. *Relation:* the algorithmic basis for Spare's RL stage. Using GRPO is not a contribution in itself.
- **7.7 to 7.11:** Reward misspecification, hacking, overoptimization, and obfuscation under optimization pressure. *Relation:* Spare's composite reward invites specific hacks, which the paper should test and report:
  - inflating clearance to pass fit while breaking function (a wobbly fit);
  - degenerate solids that "print" trivially;
  - matching the measured critical dimensions while distorting everything else;
  - gaming an editability metric with unused parameters.

## 8. Repair, replacement parts, fabrication-aware and fit-aware generation (research and products)

| # | Work | Authors | Venue / year | URL |
|---|---|---|---|---|
| 8.1 | AutoConnect: Computational Design of 3D-Printable Connectors | Koyama et al. | ACM TOG 34(6) / SIGGRAPH Asia 2015 | https://dl.acm.org/doi/10.1145/2816795.2818060 |
| 8.2 | Encore: 3D Printed Augmentation of Everyday Objects with Printed-Over, Affixed and Interlocked Attachments | Chen et al. | ACM UIST 2015 | https://cmu-devlab.github.io/research/encore/ |
| 8.3 | Reprise: A Design Tool for Specifying, Generating, and Customizing 3D Printable Adaptations on Everyday Objects *(partial: author list not confirmed)* | UNVERIFIED authors | ACM UIST 2016 | https://dx.doi.org/10.1145/2984511.2984512 |
| 8.4 | Greater than the Sum of its PARTs: Expressing and Reusing Design Intent in 3D Models | Hofmann et al. | ACM CHI 2018 | https://dl.acm.org/doi/10.1145/3173574.3173875 |
| 8.5 | Style2Fab: Functionality-Aware Segmentation for Fabricating Personalized 3D Models with Generative AI *(partial: author list taken from the related arXiv paper, not opened)* | Faruqi et al. | ACM UIST 2023 | https://dl.acm.org/doi/10.1145/3586183.3606723 |
| 8.6 | Shaping Realities: Enhancing 3D Generative AI with Fabrication Constraints | Faruqi et al. | arXiv 2024 (CHI 2024 workshop) | https://arxiv.org/abs/2404.10142 |
| 8.7 | DeepJoin: Learning a Joint Occupancy, Signed Distance, and Normal Field Function for Shape Repair | Lamb et al. | SIGGRAPH Asia 2022 (ACM TOG) | https://arxiv.org/abs/2211.12400 |
| 8.8 | Pix2Repair: Implicit Shape Restoration from Images | Song et al. | arXiv 2023 | https://arxiv.org/abs/2305.18273 |
| 8.9 | MatchMaker: Automated Asset Generation for Robotic Assembly | Wang et al. | ICRA 2025 | https://arxiv.org/abs/2503.05887 |
| 8.10 | Task-Driven 3D Printability Assistance via Geometry- and Knowledge-Grounded LLM Reasoning | Du et al. | arXiv 2026 | https://arxiv.org/abs/2608.22128 |
| 8.11 | Thingiverse "Spare Parts Challenge" and replacement-part tags; Printables `replacementpart` tag | Thingiverse / Prusa Printables | Community platforms | https://www.thingiverse.com/challenges/sparepartschallenge ; https://www.printables.com/tag/replacementpart |
| 8.12 | "3D Printing for Spares and Repairs" (case study) | iFixit | Blog / news | https://www.ifixit.com/News/115742/3d-printing-spare-parts-is-cool-but-not-a-silver-bullet |

- **8.1 AutoConnect:** Given virtual models of two objects, it synthesizes 3D-printable connectors from a database of parameterized holders, optimized for grip and material use. *Relation:* the closest prior on generating a printable part from the geometry of the objects it must attach to. It requires 3D models of the counterparts, uses templates rather than a generative code model, and targets new connectors, not replacement of a broken part.
- **8.2 Encore and 8.3 Reprise:** Design tools for printing attachments onto existing everyday objects (print-over, affix, interlock) and for fitting adaptations onto household objects. *Relation:* HCI prior on fit-to-existing-object fabrication. Both are interactive and model-driven, with no learning from photos.
- **8.4 PARTs:** Lets designers encode assertions (tests of design expectations) and integrators that generate geometry, so intent such as "this hole fits an M3 screw" survives reuse. *Relation:* prior art for representing fit-critical intent as checkable assertions, which Spare's verifiers mirror.
- **8.5 and 8.6 Style2Fab and Shaping Realities:** Segment Thingiverse models into functional and aesthetic parts so that generative stylization leaves the functional parts alone. The authors argue that generative 3D tools ignore physical and functional constraints. *Relation:* shared motivation, "function, not appearance". This line of work protects function during mesh stylization and does not generate fitting geometry.
- **8.7 and 8.8 DeepJoin and Pix2Repair:** Learn to generate the missing "restoration" piece of a fractured object, from a 3D scan (DeepJoin) or from images (Pix2Repair, with the Fantastic Breaks Imaged dataset of 11,653 images), so the patch can be 3D-printed and joined. *Relation:* the closest academic prior on photo-based replacement-part generation. Outputs are implicit or mesh shapes evaluated by Chamfer and normal consistency, not parametric programs with fit-class tolerances. The objects are mostly fractured household or ceramic items, not mechanical interfaces such as shafts and snaps.
- **8.9 MatchMaker:** Generates a mating second part for robotic assembly by constraining a 3D generative model with the first part's contact faces. *Relation:* generative creation of a mating part from counterpart geometry, which is close to Spare's "infer the mating counterpart". It is simulation-oriented and mesh-based, with no CadQuery, photos or print tolerances.
- **8.10 Task-driven printability assistance:** An LLM grounded in STL geometry and printer and material knowledge produces pre-print reports (printability, material, settings, risks). It was validated on 96 physical prints, and post-print feedback improved some recommendations. *Relation:* LLM plus printability knowledge plus physical-print feedback, but as advice on an existing STL, not generation of CAD.
- **8.11 and 8.12 Community and practice:** Thingiverse and Printables host user-designed replacement parts, and Thingiverse's 2026 Spare Parts Challenge asks for STL plus editable CAD plus an in-use photo. The iFixit case study documents real fit failures of printed spares, such as a retaining tab breaking during fitting. *Relation:* evidence of demand and of the fit problem, and a possible source of real evaluation cases (with licensing care).
- **Products, "photo to replacement part":** search found service bureaus that accept photos (goodprints3d, lyoniscorp "From Photo to Part") and generic photo-to-STL AI generators (Section 4). We did not open these pages, so their capability claims are **UNVERIFIED**. **We found no product or app that automatically produces a fit-toleranced parametric CAD replacement part from phone photos.** Searches returned only manual services and appearance-oriented mesh generators.
- **Agentic build123d design loop:** a GitHub project (`eschmitt88/agentic-3d-design`) reportedly runs an LLM verify-and-revise loop over build123d code-CAD that checks overhangs, thin walls and clearances. We saw it only in search results and did not open it, so it is **UNVERIFIED**. If confirmed, it is relevant prior art for printability checks inside an inference-time loop, without RL.

---

## UNVERIFIED or partly verified items (excluded from claims)

1. Venues of Zero-1-to-3, One-2-3-45, AutoMate and Skalse et al. The arXiv records were verified, but the conference venues were not re-checked, so they are listed as arXiv.
2. Author lists for Reprise (UIST 2016), the MDPI hole-accuracy paper (6.6), and the T-ASE FDM in-plane deviation paper (6.8). Title, venue and DOI were confirmed.
3. Style2Fab author list (the ACM page was found but not opened).
4. CADEvolve's use of RL and an IoU reward (seen only in a search snippet, not in the abstract).
5. Reward definitions for ReCAD, CME-CAD and FutureCAD (the abstracts do not specify them).
6. Vendor capability claims for Meshy, Hyper3D, PrintReady3D, goodprints3d, lyoniscorp and Backflip (current availability). Also the agentic-3d-design GitHub repo.
7. Slicer "XY hole compensation" features (e.g., OrcaSlicer). These were seen only via a blog in search results.

---

## What is and isn't new

**Already covered by prior work (Spare should not claim these):**

- **LLMs or VLMs that emit CadQuery or other CAD code from images or point clouds.** Examples: CAD-Recode, CAD-Coder (Doris et al.), cadrille, CADEvolve, GACO-CAD, ConCAD, Img2CAD. Photo-conditioned parametric CAD generation is established.
- **SFT on large procedurally generated or synthetic CadQuery pairs.** Examples: CAD-Recode (1M programs), CADEvolve (1.3M scripts), GenCAD-Code (more than 163k pairs).
- **RL (GRPO and variants) on CAD code with automatic verifier rewards.** This includes executability or valid-solid gates and geometric similarity to a ground truth (Chamfer or IoU). Examples: CAD-Coder (Guan et al., NeurIPS 2025, CadQuery with GRPO and Chamfer), cadrille (ICLR 2026, online GRPO), CAD-RL (CadQuery with exec, IoU and GPT-4o judge), ConCAD (CadQuery with GRPO, exec, code-constraint and IoU), Ortho2CAD (CadQuery with IoU RL), STEP-LLM, ReCAD, CME-CAD, FutureCAD, ToolCAD. Spare's "valid solid" reward and its geometric part of "critical-dimension match" overlap directly with this work.
- **Inference-time verification and refinement of CAD code** with VLM judges, executable tests or simulation. Examples: CADCodeVerify, CADTests, Seek-CAD, IterCAD, CADReasoner, FunCAD-Harness. FunCAD-Harness specifically targets *functional* correctness via simulated physical tests.
- **Generating a printable part from the geometry of the objects it attaches to.** Examples: AutoConnect (template-based, needs 3D models), Encore and Reprise (interactive). Separately, MatchMaker generates mating geometry from contact faces (mesh, robotics).
- **Learned restoration pieces for broken objects from images or scans.** Examples: Pix2Repair, DeepJoin. These produce mesh or implicit outputs, without tolerances.
- **Learned or model-based compensation of printed shape deviation.** Examples: Huang et al. 2015, the T-ASE FDM work, GraphCompNet. Also the HCI observation that user-made printed parts rarely fit first time (Kim et al., DIS 2017).
- **Encoding functional intent as checkable assertions.** Example: PARTs.

**Gaps where we found no prior work (state these carefully):**

- **Fit as an RL reward.** We found no prior work that trains a CAD-code generator with a reward computed against *separate mating geometry* (for example, simulated insertion or clearance of the generated part against an inferred shaft, pin, slot or snap counterpart). All verified CAD-RL rewards score similarity to a reference shape or program.
- **Printability, strength and per-fit-class tolerance as rewards.** We found no prior CAD-code RL work whose reward includes slicer-based printability, a strength or deflection check (for example, snap-hook strain per the Bayer or Klahn guidance), or tolerance bands per fit class and printer. CAD-RL and STEP-LLM explicitly list manufacturing constraints as future work.
- **Inferring the counterpart from photos and adding fit-class clearances.** We found no prior work that detects fit-critical features (D-shafts, press-fit faces, snap hooks, threads) in photos of a part *and* of its attachment context, infers the counterpart, and applies explicit clearance rules per fit class.
- **Print-test feedback mapped to parameter corrections.** We found no prior work that maps qualitative post-print fit reports ("too tight on shaft") to corrections of named CAD parameters, or that learns per-printer compensation at the level of parametric features rather than mesh or contour deviation. The closest are scan-based deviation compensation (6.7 to 6.9), parameter-localizing editing agents (IterCAD, CADReasoner), and post-print feedback for printability *advice* (8.10).
- **Product gap.** We found no product that produces a fit-toleranced, editable parametric replacement part from phone photos. Existing products either reconstruct meshes (Object Capture, Polycam, photo-to-STL generators) or need expert scan-to-CAD work (Design X, QuickSurface).

**Main overlap risk.** ConCAD (Sep 2026) and CAD-RL (2025) are the closest technical neighbours. Both train CadQuery generators with GRPO-style RL, using an execution gate plus IoU and either code-level constraint relations or an LLM judge. Spare's novelty must rest on *what is verified* (fit against counterpart geometry, printability, strength, tolerance, editability) and on the *physical print loop*, not on RL-with-geometric-reward for CadQuery. Because the field moves monthly (ConCAD and FunCAD appeared in Sep and Oct 2026), the search should be repeated before submission.

## Recommended claim wording

1. "Prior CAD-code generators trained with reinforcement learning, such as CAD-Coder, cadrille, CAD-RL and ConCAD, reward executability and geometric similarity to a reference shape. Spare instead rewards functional fit, measured against inferred mating geometry, together with printability, a simplified strength check and editability."
2. "In our review of the literature through October 2026, we found no prior work that trains a CAD-code model with rewards computed from simulated fit against separate mating geometry and from slicer-based printability." (Avoid unqualified "first" claims.)
3. "Spare builds on established image-to-CadQuery generation and GRPO-based post-training. Its contribution lies in function-first verification and in the print-test feedback loop, not in the RL algorithm or the CAD-code representation."
4. "Unlike shape-restoration methods such as Pix2Repair and DeepJoin, and connector-synthesis tools such as AutoConnect, Spare outputs an editable parametric program whose fit-critical dimensions carry explicit clearance rules per fit class and per printer."
5. "Prior compensation methods correct printed shape deviation from measured scans. Spare maps qualitative fit reports from a test print to corrections of named CAD parameters, and accumulates per-printer offsets."
