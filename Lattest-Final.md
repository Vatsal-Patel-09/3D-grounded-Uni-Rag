# FINAL RESEARCH PROPOSAL
## 2D-to-3D Scene Data Generation & Annotation Pipeline for 3D-Grounded Uni-RAG (AIED)
### Multi-Approach Synthesis & Mentor Presentation Document

**Citation Key:**
- [Synth-v2] = Kiro-SYNTHESIZED_3D_Pipeline_Proposal_v2.md
- [Perplexity] = perplexity.md
- [Proposal-B] = 3D-Pipeline-Research-Proposal.md
- [Claude] = claude-response.md
- [Kiro-Formal] = Kiro-research_proposal_2d3d_pipeline.md

---

## TABLE OF CONTENTS
1. [Problem Statement & Constraints](#1-problem-statement--constraints)
2. [Approach A: Pure Existing Dataset Leverage](#2-approach-a-pure-existing-dataset-leverage)
3. [Approach B: Pure Custom Reconstruction (No-LiDAR)](#3-approach-b-pure-custom-reconstruction-no-lidar)
4. [Approach C: Hybrid Synthesis (RECOMMENDED)](#4-approach-c-hybrid-synthesis-recommended)
5. [Approach D: API-First Rapid Prototyping](#5-approach-d-api-first-rapid-prototyping)
6. [Approach E: Synthetic Procedural Generation](#6-approach-e-synthetic-procedural-generation)
7. [Complete Unified Tool & Dataset Inventory](#7-complete-unified-tool--dataset-inventory)
8. [Comprehensive Cost Analysis](#8-comprehensive-cost-analysis)
9. [Complexity & Risk Matrix](#9-complexity--risk-matrix)
10. [Implementation Roadmap Options](#10-implementation-roadmap-options)
11. [Master Comparison Table](#11-master-comparison-table)
12. [Final Recommendation & Next Steps](#12-final-recommendation--next-steps)

---

## 1. PROBLEM STATEMENT & CONSTRAINTS

**Goal:** Build a training data pipeline for a 3D-grounded Uni-RAG system in AIED (Artificial Intelligence in Education) that can retrieve and generate interactive 3D visuals based on user queries (e.g., "show me the skeleton model relative to the teacher's desk"). [Perplexity, §System Role & Objective; Kiro-Formal, §1]

**Hard Constraints:**
- **No LiDAR hardware** (no iPhone Pro, no depth sensors) [Perplexity, §System Role & Objective; Proposal-B, §1; Kiro-Formal, §1]
- **Academic budget** (prefer free/open-source, minimal API spend) [Proposal-B, §6; Synth-v2, §6]
- **University lab setting** (must justify to mentor with concrete deliverables) [Claude, §5]
- **Need both 2D images AND 3D scene representations** with mathematical correspondence [Perplexity, §Pipeline architecture; Kiro-Formal, §3.3]
- **Rich spatial annotations** required for retrieval training [Proposal-B, §4; Kiro-Formal, §4]

**What the Pipeline Must Produce:**
1. 2D-3D paired data (image + camera pose + 3D representation) [Perplexity, §2D–3D pairing; Claude, §4]
2. Object-level spatial annotations (centroids, bounding boxes, relationships) [Proposal-B, §4C; Kiro-Formal, §4.2]
3. Natural language captions grounded in metric 3D space [Claude, §4; Proposal-B, §4B]
4. Training triplets for Uni-RAG: (query, scene, answer, 3D_asset) [Kiro-Formal, §5 Week 3]

---

## 2. APPROACH A: PURE EXISTING DATASET LEVERAGE

### 2.1 Philosophy
Use pre-existing, academically licensed datasets that **already contain** 2D-3D pairs, camera poses, meshes, and semantic labels. **Zero reconstruction code required for Day 1–3.** [Claude, §1 Executive Summary; Perplexity, §Core datasets]

### 2.2 Datasets Used

| Dataset | Scenes | Frames | 3D Format | License | Acquisition | AIED Relevance | Source |
|---------|--------|--------|-----------|---------|-------------|----------------|--------|
| **ScanNet v2** | 1,513 | 2.5M RGB-D | Mesh + point cloud + poses | CC BY 4.0 (academic) | 1–3 day approval | ★★★★★ Classroom-scale indoor | [Perplexity, §Recommended primary datasets; Claude, §Agent 1 Finding 1; Proposal-B, §2A; Kiro-Formal, §2.1] |
| **Matterport3D** | 90 buildings | 194K RGB-D | Semantic mesh + panoramas | Academic agreement | License agreement | ★★★★☆ Room-scale spatial reasoning | [Perplexity, §Recommended primary datasets; Claude, §Agent 1 Finding 1; Kiro-Formal, §2.1] |
| **HM3D** | 1,000 | ~5M views | Photorealistic 3D env | Apache 2.0 | Instant | ★★★★☆ Navigation-scale | [Perplexity, §Recommended primary datasets; Claude, §Agent 1 Finding 1; Kiro-Formal, §2.1] |
| **Replica** | 18 | Synthetic | Mesh + HDR + semantics | MIT | Instant (Git LFS) | ★★★☆☆ High fidelity, limited scale | [Perplexity, §Recommended primary datasets; Claude, §Agent 1 Finding 1; Proposal-B, §2A; Kiro-Formal, §2.1] |
| **HyperSim** | 461 rooms | 77K images | Dense depth + normals + labels | MIT | Instant | ★★★★☆ Photorealistic synthetic | [Perplexity, §Recommended primary datasets; Proposal-B, §2A; Synth-v2, §2.1; Kiro-Formal, §2.1] |
| **3RScan** | 1,482 | 2.4M RGB | Point cloud + change graphs | Academic | Registration | ★★★☆☆ Object re-localization | [Perplexity, §Recommended primary datasets; Proposal-B, §2A] |
| **ARKitScenes** | 5,047 | >200M frames | LiDAR mesh (pre-captured) | Apple Academic | Instant | ★★★★☆ High-quality indoor validation | [Perplexity, §Recommended primary datasets; Proposal-B, §2A; Synth-v2, §2.1] |
| **HSSD** | 211 | ~18K objects | Synthetic interior scenes | CC BY-NC 4.0 | Instant | ★★★★☆ Embodied AI compatible | [Perplexity, §Recommended primary datasets] |
| **SceneSplat-7K** | 7,916 | ~4.72M | Precomputed 3D Gaussian Splats | Non-commercial research | HuggingFace | ★★★★★ Ready-to-use 3DGS | [Perplexity, §Pre-splat scenes; Synth-v2, §2.1] |

### 2.3 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INGESTION                                                  │
│  ├── ScanNet .sens → SensReader.py → RGB + depth + poses   │  [Claude, §5 Day 1–2]
│  ├── Matterport3D → download_mp.py → region manifests      │  [Perplexity, §Ingestion layer]
│  ├── Replica/HM3D/HSSD → Habitat-compatible loaders        │  [Perplexity, §Ingestion layer]
│  └── SceneSplat-7K → HuggingFace → pre-built Gaussian scenes│  [Perplexity, §Pre-splat scenes]
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PROCESSING (No GPU required initially)                     │
│  ├── Frame sampling: 1 fps or keyframe-based               │  [Claude, §5 Day 2]
│  ├── Pose parsing: Load pre-computed T_i (4×4 camera-to-world)│ [Claude, §Agent 2]
│  ├── Mesh loading: .ply / .obj via Open3D                   │  [Claude, §5 Day 3]
│  └── 3D annotation read: semantic labels, instance IDs      │  [Perplexity, §Object-level pairing]
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  2D-3D PAIRING (Mathematical Core)                          │
│  ├── For each frame i:                                      │
│  │   K = intrinsic matrix (3×3)                             │  [Claude, §Agent 2; Kiro-Formal, §3.3]
│  │   T_i = pose matrix (4×4)                                │  [Claude, §Agent 2; Kiro-Formal, §3.3]
│  │   Project 3D bbox corners: p_2d = K × (T_i⁻¹ × P_world) │  [Claude, §Agent 2; Perplexity, §2D–3D pairing]
│  ├── Compute distances: ||centroid_3d - camera_pos||₂       │  [Perplexity, §Object-level pairing]
│  ├── Extract spatial relations: LEFT_OF, IN_FRONT_OF, etc.  │  [Perplexity, §Spatial relation extraction; Proposal-B, §3.2 Step 4]
│  └── Build metadata_json per frame                          │  [Claude, §4; Proposal-B, §4C]
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  VLM ANNOTATION                                             │
│  ├── Input: RGB frame + metadata_json + schema prompt      │  [Claude, §4; Proposal-B, §4C]
│  ├── Model: GPT-4o or Gemini 1.5 Pro                       │  [Perplexity, §VLM capabilities; Kiro-Formal, §2.3]
│  ├── Output: JSON with global_caption, objects, relations  │  [Claude, §4 JSON Schema; Kiro-Formal, §4.2]
│  └── Storage: JSONL/Parquet + SQLite index                 │  [Perplexity, §Tool stack summary; Proposal-B, §3.2]
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Tools Required
- **Open3D** / **PyTorch3D** / **trimesh** — 3D I/O and projection [Perplexity, §Tool stack summary; Claude, §5 Day 3]
- **pycolmap** — Python bindings for COLMAP formats [Claude, §5 Day 3]
- **ScanNet SDK** — SensReader.py for .sens decoding [Claude, §5 Day 1]
- **Habitat-sim** — For HM3D/HSSD/Replica scene loading [Perplexity, §Tool stack summary]
- **VLM API** — GPT-4o or Gemini 1.5 Pro [Perplexity, §VLM capabilities; Kiro-Formal, §2.3]

### 2.5 Complexity Analysis
| Factor | Level | Notes |
|--------|-------|-------|
| Setup | LOW | Download + SDK install only [Claude, §5 Day 1] |
| Compute | LOW–MEDIUM | CPU for pairing; GPU only for VLM batch [Claude, §Appendix] |
| Code | MEDIUM | Need unified dataset loaders + projection math [Claude, §5 Day 3] |
| Data Volume | HIGH | 2.5M+ frames available immediately [Claude, §Appendix] |
| Quality | HIGH | Ground-truth poses and geometry [Claude, §1] |

### 2.6 Cost
- **Datasets:** $0 (academic licenses) [Perplexity, §Dataset costs; Proposal-B, §6]
- **Compute:** $0 (laptop/CPU for pairing; lab GPU for optional rendering) [Claude, §5]
- **VLM Annotation:** ~$2,000–3,750 for full ScanNet (2.5M frames) at standard API rates; ~$1,000–1,875 with Batch API [Synth-v2, §6.2; Perplexity, §VLM Annotation cost estimates]
- **Storage:** ~2TB for ScanNet full; ~2.76TB for SceneSplat-7K [Perplexity, §Scalability and compute trade-off]

### 2.7 Pros & Cons
**Pros:**
- Immediate start (no waiting for reconstruction) [Claude, §1]
- Ground-truth accuracy for poses and geometry [Claude, §1]
- No calibration or scale ambiguity [Claude, §1]
- 2.5M+ training pairs possible from ScanNet alone [Claude, §Appendix]

**Cons:**
- Limited to existing scene types (mostly residential, not specifically educational) [Claude, §1]
- Cannot add custom lab/classroom environments [Claude, §1]
- Domain gap: synthetic (HyperSim) vs. real classrooms [Synth-v2, §8 Risk Register]
- Licensing restrictions (non-commercial) [Perplexity, §Recommended primary datasets]

---

## 3. APPROACH B: PURE CUSTOM RECONSTRUCTION (NO-LIDAR)

### 3.1 Philosophy
Generate 3D scenes entirely from **smartphone/DSLR imagery or video** using state-of-the-art no-LiDAR reconstruction. Full control over domain (educational environments). [Proposal-B, §1; Kiro-Formal, §1]

### 3.2 Input Requirements
- **Video:** 30–60s slow orbit, 1080p+, good lighting, avoid reflections [Proposal-B, §3.1 Track B; Kiro-Formal, §3.1 Path B]
- **Images:** 50–200 photos, >70% overlap, varying heights/angles [Kiro-Formal, §3.1 Path B]
- **Extraction:** `ffmpeg -i video.mp4 -vf "fps=3,scale=1280:720" frames/%06d.jpg` [Proposal-B, §3.1 Track B; Synth-v2, §3.1]

### 3.3 Reconstruction Stack Options

#### Option B1: DUSt3R-First Pipeline (RECOMMENDED within Approach B)
```
Input Images (uncalibrated, arbitrary order)
    │
    ▼
DUSt3R / MASt3R (CVPR 2024, Apache 2.0)
├── Dense pointmap + camera poses + confidence
├── NO calibration needed
└── 71.8% δ1 accuracy on ScanNet without poses
    │
    ▼
3D Gaussian Splatting (Nerfstudio Splatfacto + gsplat)
├── Real-time renderable representation
├── 30K iterations ≈ 15–30 min on RTX 4090
└── Export: PLY + transforms.json
    │
    ▼
SuperSplat (MIT, browser-based)
├── Remove floaters, crop ROI
└── Export cleaned PLY or SPZ (90% compression)
```
[Proposal-B, §2B; Synth-v2, §2.2, §3.2; Kiro-Formal, §3.2]

#### Option B2: COLMAP-First Pipeline (Fallback)
```
Input Images
    │
    ▼
COLMAP (SfM + MVS, BSD license)
├── Feature extraction (SIFT with GPU)
├── Sequential matching (for video)
├── Sparse reconstruction + bundle adjustment
└── Output: cameras.bin, images.bin, points3D.bin
    │
    ▼
OpenMVS (dense mesh reconstruction)
    │
    ▼
Nerfstudio (nerfacto / splatfacto)
├── ns-process-data converts COLMAP → transforms.json
└── Train Gaussian Splat or NeRF
```
[Perplexity, §3D reconstruction without LiDAR; Kiro-Formal, §3.2; Proposal-B, §2B]

#### Option B3: Feed-Forward Sparse Capture
- **MASt3R** — DUSt3R successor, pairwise dense matching, works with as few as 2 images [Synth-v2, §2.2; Proposal-B, §2B]
- Best for quick captures of small educational props [Synth-v2, §2.2]

### 3.4 Depth Estimation (Critical for Metric Scale)
Since custom captures lack ground-truth depth:

| Model | Role | Indoor Accuracy | Speed | License | Source |
|-------|------|-----------------|-------|---------|--------|
| **UniDepth v2 (Large)** | PRIMARY metric depth | SUN RGB-D: 96.4% δ1 | ~5 FPS on A100 | Apache 2.0 | [Synth-v2, §2.3; Proposal-B, §2B] |
| **Depth Anything v2 (Large)** | FALLBACK + relative depth | KITTI: 98.3% δ1 | ~10 FPS on A100 | MIT | [Synth-v2, §2.3; Proposal-B, §2B] |
| **Metric3D v2** | VALIDATION baseline | Competitive | Medium | Academic | [Synth-v2, §2.3] |

**Scale Recovery:** Use DUSt3R point cloud median depth to globally scale monocular depth predictions. [Synth-v2, §3.2 Step 2; Proposal-B, §2B]

### 3.5 Segmentation & Object Extraction
```
Keyframe (every 10th frame)
    │
    ▼
Grounding DINO (text-to-box detection)
├── Zero-shot object detection
└── Classes: whiteboard, desk, chair, lab_equipment, etc.
    │
    ▼
SAM2 (video-native segmentation, Apache 2.0)
├── Instance masks from detection boxes
└── Temporal consistency across frames
    │
    ▼
3D Unprojection
├── points_3d = unproject(mask_pixels, depth_map, K, [R|t])
├── centroid_3d = mean(points_3d)
└── bbox_3d = [min(points_3d), max(points_3d)]
    │
    ▼
Spatial Graph
├── Pairwise relations: LEFT_OF, IN_FRONT_OF, ABOVE, etc.
└── Metric distances between centroids
```
[Synth-v2, §2.4, §3.2 Step 3–4; Proposal-B, §3.2 Step 3–4]

**Quality Filters:**
- Discard masks with area < 0.5% of image [Synth-v2, §1 Dimension 4]
- Discard masks where mean depth confidence < 0.7 [Synth-v2, §1 Dimension 4]
- Filter at object boundaries to reduce noise [Synth-v2, §1 Dimension 4]

### 3.6 VLM Annotation Strategy

**Metadata-First Approach (Prevents Hallucination):**
- **Inputs to VLM:** RGB keyframe + Colorized depth map (jet colormap) + BEV composite + STRUCTURED METADATA STRING [Proposal-B, §4A; Synth-v2, §3.4]
- **Metadata String (ground truth facts):**
  ```
  SCENE: university_lecture_hall_001
  CAMERA_HEIGHT: 1.45m above floor
  FOV: 72.3° horizontal
  FLOOR_AREA: 8.2m × 12.5m
  OBJECTS: whiteboard_01 | centroid=(0.2, 1.5, 8.1)m | distance=8.2m | area=4.2m²
  SPATIAL_RELATIONSHIPS: desk_01 is 1.2m LEFT_OF chair_01
  ```
  [Proposal-B, §4C; Synth-v2, §3.4]
- **VLM Role:** "Semantic Translator" — converts metric facts to intuitive language ("arm's reach", "across the room") [Proposal-B, §4B; Synth-v2, §3.4]
- **Strict Rules:**
  - DO: Generate intuitive spatial descriptions [Synth-v2, §3.4]
  - DO: Identify educational context [Synth-v2, §3.4]
  - DO NOT: Invent distances or positions [Proposal-B, §4B; Synth-v2, §3.4]
  - DO NOT: Hallucinate objects not in metadata [Proposal-B, §4B; Synth-v2, §3.4]

### 3.7 Complexity Analysis
| Factor | Level | Notes |
|--------|-------|-------|
| Setup | MEDIUM | Install DUSt3R, Nerfstudio, SAM2, depth models [Proposal-B, §7 Day 1] |
| Compute | HIGH | 16GB+ VRAM for DUSt3R; 24GB ideal for large scenes [Synth-v2, §2.2] |
| Code | HIGH | Need reconstruction, depth, segmentation, unprojection pipelines [Proposal-B, §3.2] |
| Data Volume | LOW–MEDIUM | Limited by capture time (1 scene ≈ 1 hour processing) [Proposal-B, §2B] |
| Quality | MEDIUM–HIGH | Depends on capture quality; textureless surfaces problematic [Synth-v2, §8 Risk Register] |

### 3.8 Cost
- **Software:** $0 (all open-source) [Proposal-B, §6]
- **Compute:** RTX 4090 ($1,600 one-time) or Colab A100 (~$10/day) [Proposal-B, §2B; Synth-v2, §6.1]
- **Cloud API (Luma/Polycam fallback):** Luma 500 free splats/month; Polycam $20/month unlimited [Kiro-Formal, §2.2; Perplexity, §3D reconstruction without LiDAR]
- **VLM Annotation:** ~$400 for 500 scenes (tiered Flash/Pro strategy) [Synth-v2, §6.2]
- **Time:** ~45 min per scene on RTX 3090; ~20 min on A100 [Proposal-B, §2B]

### 3.9 Pros & Cons
**Pros:**
- Full domain control (scan actual classrooms, labs, anatomy models) [Proposal-B, §1; Kiro-Formal, §1]
- No dataset licensing restrictions on captured content [Kiro-Formal, §1]
- Scales to any educational environment [Proposal-B, §1]
- DUSt3R eliminates calibration barrier entirely [Proposal-B, §1; Synth-v2, §1 Dimension 1]

**Cons:**
- Reconstruction fails on textureless surfaces (whiteboards, bare walls) [Synth-v2, §8 Risk Register; Proposal-B, §8]
- Metric scale ambiguity without known reference objects [Synth-v2, §8 Risk Register]
- Computationally intensive per scene [Proposal-B, §2B]
- Requires good capture technique (overlap, lighting) [Kiro-Formal, §6 Risk Mitigation]

---

## 4. APPROACH C: HYBRID SYNTHESIS (RECOMMENDED)

### 4.1 Philosophy
Combine **Track A (Existing Datasets)** for immediate volume + **Track B (Custom Reconstruction)** for domain specificity + **Track C (Object Assets)** for populating scenes. This is the converged architecture from deep multi-agent analysis. [Synth-v2, §Executive Summary; Proposal-B, §1]

### 4.2 Three Parallel Tracks

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HYBRID PIPELINE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TRACK A: EXISTING DATASETS          TRACK B: CUSTOM CAPTURE           │
│  ├─ HyperSim (Day 1, MIT)            ├─ Smartphone video/photos        │  [Synth-v2, §3.1; Proposal-B, §1]
│  ├─ ScanNet (1–3 day approval)       ├─ FFmpeg extraction (fps=3)     │  [Synth-v2, §3.1; Proposal-B, §3.1]
│  ├─ Replica (instant)                ├─ DUSt3R → pointmap + poses      │  [Synth-v2, §3.2]
│  ├─ ARKitScenes (validation)         ├─ UniDepth v2 → metric depth    │  [Synth-v2, §3.2]
│  └─ SceneSplat-7K (instant 3DGS)     ├─ SAM2 + Grounding DINO         │  [Synth-v2, §2.4]
│                                      └─ Nerfstudio Splatfacto          │  [Synth-v2, §2.2]
│                                                                         │
│  TRACK C: OBJECT ASSETS                                                 │
│  └─ Objaverse-XL (10M+ objects, HuggingFace)                           │  [Synth-v2, §3.1; Kiro-Formal, §2.1]
│      → Filter by "education", "anatomy", "chemistry" tags               │  [Synth-v2, §3.1]
│      → Populate synthetic scenes with domain-specific props             │  [Synth-v2, §3.1]
│                                                                         │
│  UNIFIED PROCESSING LAYER                                               │
│  ├─ 2D-3D Pairing (projection math)                                    │  [Synth-v2, §3.3; Kiro-Formal, §3.3]
│  ├─ Spatial Graph Construction                                         │  [Synth-v2, §3.2 Step 4]
│  ├─ BEV Generation (top-down XZ projection)                            │  [Synth-v2, §1 Dimension 5; Proposal-B, §3.2 Step 4]
│  └─ Quality Gates (CLIP, depth coverage, pose confidence)              │  [Synth-v2, §1 Dimension 6; Proposal-B, §3.2 Quality Filter]
│                                                                         │
│  ANNOTATION LAYER                                                       │
│  ├─ Metadata-First VLM (Gemini 2.5 Pro / Flash)                        │  [Synth-v2, §3.4]
│  ├─ BEV as 4th input modality                                          │  [Synth-v2, §1 Dimension 5]
│  ├─ Tiered routing: Flash (80%) / Pro (20%)                            │  [Synth-v2, §6.2]
│  └─ Quality Gate: ±20% distance tolerance check                        │  [Synth-v2, §3.4]
│                                                                         │
│  OUTPUT                                                                 │
│  └─ Uni-RAG Training Corpus: (query, scene, answer, 3D_asset)          │  [Synth-v2, §7 Week 4]
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Corrected & Validated Tool Stack

#### Datasets
| Dataset | Scenes | License | Download | Role | Source |
|---------|--------|---------|----------|------|--------|
| **HyperSim** | 461 rooms, 77K images | MIT | Instant | PRIMARY — Day 1 validation, perfect ground truth | [Synth-v2, §2.1; Proposal-B, §2A] |
| **ScanNet v2** | 1,513 scans, 2.5M frames | Academic | 1–3 days | SECONDARY — real-world indoor diversity | [Synth-v2, §2.1; Proposal-B, §2A; Claude, §Agent 1] |
| **Replica** | 18 scenes | MIT | Fast (Git LFS) | TERTIARY — high-fidelity Habitat-compatible | [Synth-v2, §2.1; Proposal-B, §2A] |
| **ARKitScenes** | 5,047 scenes, >200M frames | Apple Academic | Moderate | VALIDATION — LiDAR ground truth benchmark | [Synth-v2, §2.1; Proposal-B, §2A] |
| **Objaverse-XL** | 10M+ objects | Open (CC variants) | Fast (HF) | OBJECT DIVERSITY — educational props/models | [Synth-v2, §2.1; Kiro-Formal, §2.1] |
| **SceneSplat-7K** | 7,916 scenes | Non-commercial | HuggingFace | INSTANT 3DGS — skip training | [Perplexity, §Pre-splat scenes; Synth-v2, §2.1] |

#### 3D Reconstruction (No LiDAR)
| Tool | Role | License | Input | Output | GPU | Source |
|------|------|---------|-------|--------|-----|--------|
| **DUSt3R** | PRIMARY reconstruction | Apache 2.0 | Arbitrary images | Pointmap + poses + confidence | 16GB+ VRAM | [Synth-v2, §2.2; Proposal-B, §2B] |
| **MASt3R** | PAIRWISE matching | Apache 2.0 | Image pairs | Dense matching + 3D points | 16GB+ VRAM | [Synth-v2, §2.2; Proposal-B, §2B] |
| **COLMAP** | FALLBACK reconstruction | BSD | Sequential images | Sparse point cloud + poses | 8GB+ VRAM | [Synth-v2, §2.2; Perplexity, §3D reconstruction] |
| **Nerfstudio (Splatfacto)** | Gaussian Splat training/export | Apache 2.0 | Images + poses | PLY + transforms.json | 8GB+ VRAM | [Synth-v2, §2.2; Kiro-Formal, §2.2] |
| **SuperSplat** | Post-processing editor | MIT | PLY | Cleaned PLY/SPZ | Browser | [Synth-v2, §2.2; Kiro-Formal, §2.2] |
| **gsplat** | CUDA backend (4× memory reduction) | MIT | Nerfstudio backend | Optimized training | CUDA | [Synth-v2, §2.2; Kiro-Formal, §2.2] |

**CRITICAL CORRECTION:** Luma AI has pivoted to video generation; free tier limited to 30 watermarked generations/month with no commercial use. **Removed as core stack.** Retained only as rapid prototyping fallback. [Synth-v2, §2.2 CRITICAL CORRECTION]

#### Depth Estimation
| Model | Role | Indoor Accuracy | Speed | Source |
|-------|------|-----------------|-------|--------|
| **UniDepth v2 (Large)** | PRIMARY metric depth | SUN RGB-D: 96.4% δ1 | ~5 FPS on A100 | [Synth-v2, §2.3; Proposal-B, §2B] |
| **Depth Anything v2 (Large)** | FALLBACK + relative depth | KITTI: 98.3% δ1 | ~10 FPS on A100 | [Synth-v2, §2.3; Proposal-B, §2B] |
| **Metric3D v2** | VALIDATION baseline | Competitive | Medium | [Synth-v2, §2.3] |

#### Segmentation
| Tool | Role | Notes | Source |
|------|------|-------|--------|
| **SAM2** | Instance segmentation | Video-native, released July 2024 | [Synth-v2, §2.4; Proposal-B, §2B] |
| **Grounding DINO** | Text-to-box detection | Zero-shot object detection | [Synth-v2, §2.4; Proposal-B, §2B] |
| **Grounded-SAM2** | Combined pipeline | Chain SAM2 + Grounding DINO if combined not released | [Synth-v2, §2.4] |

#### VLM Annotation
| Model | Role | Cost (Batch API) | Context | Structured Output | Source |
|-------|------|------------------|---------|-------------------|--------|
| **Gemini 2.5 Pro** | PRIMARY annotator | $0.625/MTok in, $5/MTok out | 1M tokens | Native JSON schema | [Synth-v2, §2.5] |
| **Gemini 2.5 Flash** | FAST annotator | $0.075/MTok in, $1.25/MTok out | 1M tokens | Native JSON schema | [Synth-v2, §2.5] |
| **GPT-4o** | FALLBACK | $2.50/MTok in, $10/MTok out | 128K tokens | response_format: json_object | [Synth-v2, §2.5; Kiro-Formal, §2.3] |
| **GPT-5.4** | Complex reasoning fallback | Higher | 128K tokens | JSON mode | [Synth-v2, §2.5] |

**CORRECTED COST:** ~$400 for 500 scenes (tiered: 80% Flash @ ~$160 + 20% Pro @ ~$240). NOT $0.50 as incorrectly claimed in early drafts. [Synth-v2, §2.5 CORRECTION; §6.2]

### 4.4 Quality Gates (Adopted from Proposal A)
- **CLIP score ≥ 0.25** — discard blurry/dark frames [Synth-v2, §1 Dimension 6; Proposal-B, §3.2 Quality Filter]
- **Depth coverage ≥ 70%** — discard frames where depth fails [Synth-v2, §1 Dimension 6; Proposal-B, §3.2 Quality Filter]
- **Pose confidence threshold** — filter DUSt3R low-confidence reconstructions [Synth-v2, §1 Dimension 6]
- **Deduplication** — cosine similarity < 0.95 between adjacent frames [Synth-v2, §1 Dimension 6; Proposal-B, §3.2 Quality Filter]
- **VLM consistency check** — verify VLM distances match metadata ±20% [Synth-v2, §3.4 Quality Gate]
- **Object completeness** — all metadata objects must appear in output [Synth-v2, §3.4 Quality Gate]

### 4.5 JSON Schema (Merged Best of All Proposals)

```json
{
  "scene_id": "string",
  "image_id": "string",
  "dataset": "string",

  "camera": {
    "position_world": [x, y, z],
    "rotation_world": [[...]],
    "rotation_quaternion": [qx, qy, qz, qw],
    "intrinsics": { "fx": 0, "fy": 0, "cx": 0, "cy": 0, "width": 0, "height": 0 },
    "height_from_floor_m": 0.0,
    "coordinate_frame": "right-handed_z_up"
  },

  "depth_stats": {
    "min_depth_m": 0, "max_depth_m": 0, "mean_depth_m": 0, "coverage_ratio": 0
  },

  "objects": [{
    "object_id": "string",
    "category": "string",
    "subtype": "string",
    "attributes": ["wooden", "brown"],
    "bbox_2d": [x, y, w, h],
    "bbox_3d": { "center_world": [x,y,z], "size_world": [dx,dy,dz], "orientation_world": [qx,qy,qz,qw] },
    "centroid_3d": [x, y, z],
    "distance_to_camera_m": 0.0,
    "height_above_floor_m": 0.0,
    "visibility": { "visible": true, "occlusion_ratio": 0.0 },
    "natural_caption": "string",
    "canonical_queries": ["teacher's desk", "front lab bench"],
    "mask_2d_rle": "string"
  }],

  "spatial_relationships": [{
    "subject_id": "string",
    "predicate": "LEFT_OF|RIGHT_OF|IN_FRONT_OF|BEHIND|ABOVE|BELOW|ADJACENT_TO|CONTAINS|ON_TOP_OF",
    "object_id": "string",
    "distance_m": 0.0,
    "frame_of_reference": "camera|world",
    "confidence": 0.0
  }],

  "spatial_caption": "Primary grounded spatial description",
  "detail_caption": "Richer visual/contextual description",
  "global_caption": "Detailed natural language description",

  "scene_tags": ["classroom", "lab", "anatomy"],

  "educational_context": {
    "subject_domain": "biology",
    "scene_type": "laboratory",
    "key_concepts": ["human_skeleton_structure"],
    "learning_objectives": ["identify bone positions"],
    "pedagogical_role": "teacher_area|student_workspace|demo_zone",
    "retrieval_queries": ["Where is the whiteboard?"]
  },

  "bev_map_path": "path/to/bev.png",
  "rgb_path": "path/to/rgb.jpg",
  "depth_path": "path/to/depth.png",
  "pointcloud_path": "path/to/pointcloud.ply",
  "gaussian_splat_path": "path/to/splat.ply",

  "quality_score": {
    "clip_score": 0.0,
    "depth_coverage": 0.0,
    "pose_confidence": 0.0
  },

  "annotation_meta": {
    "vlm_model": "gemini-2.5-pro",
    "vlm_prompt_version": "v1.0",
    "geometry_source": "dust3r|scannet_mesh|nerfstudio_splatfacto",
    "timestamp": "ISO-8601",
    "confidence": 0.0,
    "cost_usd": 0.0
  }
}
```
[Perplexity, §JSON schema for annotations; Claude, §4 JSON Schema; Proposal-B, §5; Kiro-Formal, §4.2; Synth-v2, §4 Merged JSON Schema]

### 4.6 Implementation File Map

```
3d-pipeline/
├── README.md
├── requirements.txt
├── config/
│   ├── pipeline.yaml
│   ├── edu_classes.txt
│   └── vlm_prompts/
│       ├── metadata_first_system.txt
│       ├── metadata_first_user.txt
│       └── few_shot_example.json
├── ingest/
│   ├── dataset_loader.py        (ScanNet/HyperSim/Replica/ARKitScenes)
│   ├── video_extractor.py       (ffmpeg wrapper)
│   └── object_downloader.py     (Objaverse-XL HF downloader)
├── process/
│   ├── dust3r_runner.py         (Primary reconstruction)
│   ├── colmap_runner.py         (Fallback reconstruction)
│   ├── depth_estimator.py       (UniDepth v2 + Depth Anything v2)
│   ├── gsam_segmenter.py        (SAM2 + Grounding DINO)
│   └── bev_generator.py         (Top-down XZ projection)
├── pair/
│   ├── projector.py             (3D→2D projection math)
│   ├── spatial_graph.py         (Pairwise relations)
│   └── index_builder.py         (SQLite/Parquet index)
├── annotate/
│   ├── vlm_annotator.py         (Gemini/OpenAI API clients)
│   ├── metadata_builder.py      (Spatial metadata string formatter)
│   ├── schema_validator.py      (Pydantic/JSON Schema validation)
│   └── quality_gate.py          (CLIP, depth, consistency checks)
├── store/
│   ├── hdf5_writer.py
│   ├── sqlite_index.py
│   ├── parquet_exporter.py
│   └── splat_converter.py       (SPZ compression)
├── validate/
│   ├── dust3r_vs_arkit.py       (LiDAR ground truth comparison)
│   ├── depth_accuracy.py
│   └── vlm_consistency.py
├── train_prep/
│   ├── unirag_formatter.py      (Training triplet generation)
│   └── negative_sampler.py
└── scripts/
    ├── setup_env.sh
    ├── download_hypersim.sh
    ├── run_pipeline.py
    └── cost_estimator.py
```
[Synth-v2, §5; Proposal-B, §3.2 Key implementation files]

### 4.7 Week-by-Week Execution Plan

**Week 1: Foundation**
- Day 1: Install core stack (PyTorch, Nerfstudio, DUSt3R, Depth Anything V2) [Synth-v2, §7 Day 1; Proposal-B, §7 Day 1]
- Day 2–3: Download HyperSim (instant, MIT) + request ScanNet [Synth-v2, §7 Day 1; Proposal-B, §7 Day 1]
- Day 4–5: Validate DUSt3R on HyperSim images vs. ground truth [Synth-v2, §7 Day 2–3]
- Day 6–7: End-to-end PoC on 1 HyperSim scene → validate JSON schema [Synth-v2, §7 Day 6–7]

**Week 2: Scale & Validation**
- Process 10 HyperSim scenes → 500 annotated frames [Synth-v2, §7 Week 2]
- Download ScanNet (should be approved) [Synth-v2, §7 Week 2]
- Process 5 ScanNet scenes → validate against ground truth poses [Synth-v2, §7 Week 2]
- Download ARKitScenes → run DUSt3R → compare to LiDAR ground truth [Synth-v2, §7 Week 2]
- Target: RMSE < 15cm for indoor scenes [Synth-v2, §7 Week 2]

**Week 3: VLM Integration & Cost Optimization**
- Set up Gemini API with Batch processing [Synth-v2, §7 Week 3]
- Implement metadata string builder + BEV generation [Synth-v2, §7 Week 3]
- Run annotation on 100 scenes (Flash vs. Pro comparison) [Synth-v2, §7 Week 3]
- Implement tiered routing: simple → Flash, complex → Pro [Synth-v2, §7 Week 3]

**Week 4: Uni-RAG Training Data**
- Convert annotations to training triplets [Synth-v2, §7 Week 4]
- Generate negative samples for contrastive learning [Synth-v2, §7 Week 4]
- Export to Parquet for PyTorch DataLoader [Synth-v2, §7 Week 4]
- Target: 10,000 training triplets across 50 scenes, 4 domains [Synth-v2, §7 Week 4]

---

## 5. APPROACH D: API-FIRST RAPID PROTOTYPING

### 5.1 Philosophy
Use managed cloud APIs for 3D reconstruction to achieve **same-day results** without local GPU infrastructure. Best for proof-of-concept demos when lab GPU is unavailable. [Kiro-Formal, §2.2 Secondary; Perplexity, §3D reconstruction without LiDAR]

### 5.2 Tools
| Service | Cost | Input | Output | Limitations | Source |
|---------|------|-------|--------|-------------|--------|
| **Luma AI** | 30 free generations/month (watermarked) | Smartphone video | GLB/USD/PLY Gaussian splat | Pivoted to video generation; NOT viable for academic research | [Synth-v2, §2.2 CRITICAL CORRECTION; Kiro-Formal, §2.2] |
| **Polycam** | $20/month unlimited | 20–200 images | GLTF/OBJ/PLY | API-dependent; ToS requires attribution for public display | [Perplexity, §3D reconstruction without LiDAR; Kiro-Formal, §2.2] |
| **Polyform** (Polycam OSS) | Free | Polycam exports | NeRF format conversion | MIT-licensed tooling | [Perplexity, §3D reconstruction without LiDAR] |

### 5.3 Pipeline
```
Smartphone Video
    │
    ▼
Luma AI / Polycam API
├── Upload video → Cloud reconstruction
└── Download: mesh/point cloud/NeRF asset
    │
    ▼
Nerfstudio Import
├── Polycam/RealityCapture formats native support
└── Continue with pairing + annotation layers
```
[Perplexity, §Processing and reconstruction; Kiro-Formal, §3.2]

### 5.4 Complexity & Cost
- **Setup:** MINIMAL (web upload) [Kiro-Formal, §5.4]
- **Compute:** NONE locally [Kiro-Formal, §5.4]
- **Time:** Minutes per scene [Kiro-Formal, §5.4]
- **Cost:** $20–50/month for unlimited Polycam; Luma free tier insufficient [Kiro-Formal, §5.4; Synth-v2, §2.2]
- **Risk:** API dependence, data privacy, ToS restrictions, watermarking [Kiro-Formal, §5.4; Synth-v2, §2.2]

### 5.5 When to Use
- Lab GPU unavailable during first week [Kiro-Formal, §5.5]
- Need mentor demo in <24 hours [Kiro-Formal, §5.5]
- Scanning sensitive educational environments where code setup is undesirable [Kiro-Formal, §5.5]
- **NOT recommended as core research infrastructure** [Synth-v2, §2.2 CRITICAL CORRECTION]

---

## 6. APPROACH E: SYNTHETIC PROCEDURAL GENERATION

### 6.1 Philosophy
Generate **unlimited synthetic data** using simulators and procedural engines. Addresses domain gap by creating idealized educational environments. [Perplexity, §Datasets; Kiro-Formal, §2.1 AI2-THOR]

### 6.2 Tools

| Tool | Type | License | Capability | Source |
|------|------|---------|------------|--------|
| **AI2-THOR** | Procedural simulator | Apache 2.0 | 120+ room types, programmatic viewpoint generation, physics | [Perplexity, §Recommended primary datasets; Kiro-Formal, §2.1] |
| **Hypersim** | Synthetic dataset (static) | MIT | 461 pre-rendered rooms with perfect ground truth | [Perplexity, §Recommended primary datasets; Synth-v2, §2.1] |
| **Habitat-sim** | Embodied AI simulator | Apache 2.0 | HM3D/HSSD/Replica integration, arbitrary camera trajectories | [Perplexity, §Tool stack summary] |
| **Blender + Objaverse-XL** | Manual/procedural | CC variants | Populate custom scenes with 10M+ educational objects | [Kiro-Formal, §2.1; Synth-v2, §3.1 Track C] |

### 6.3 Pipeline (AI2-THOR Example)
```python
from ai2thor.controller import Controller
ctrl = Controller(scene="FloorPlan212", gridSize=0.25)

# Generate unlimited viewpoints
event = ctrl.step(action="Teleport", position=dict(x=1.0, z=2.0), rotation=dict(y=45))
rgb = event.frame                                    # 2D image
depth = event.depth_frame                            # Ground-truth depth
objects = event.metadata["objects"]                  # 3D positions, bounding boxes

# Automatic annotation — no VLM needed for ground truth!
```
[Perplexity, §Datasets; Kiro-Formal, §2.1]

### 6.4 Complexity & Cost
- **Setup:** MEDIUM (simulator install) [Kiro-Formal, §6.4]
- **Compute:** MEDIUM (CPU for AI2-THOR; GPU for rendering) [Kiro-Formal, §6.4]
- **Data Volume:** UNLIMITED (programmatic generation) [Kiro-Formal, §6.4]
- **Quality:** HIGH (perfect ground truth) but domain gap to real images [Kiro-Formal, §6.5]

### 6.5 Pros & Cons
**Pros:**
- Infinite data generation [Kiro-Formal, §6.5]
- Perfect ground truth (no VLM needed for spatial annotations) [Kiro-Formal, §6.5]
- Zero API costs for annotations [Kiro-Formal, §6.5]
- Full control over educational content [Kiro-Formal, §6.5]

**Cons:**
- Domain gap: synthetic → real transfer required [Kiro-Formal, §6.5; Synth-v2, §8 Risk Register]
- Not photorealistic enough for some VLMs [Kiro-Formal, §6.5]
- Requires simulator expertise [Kiro-Formal, §6.5]
- May not impress mentor without real-world validation [Kiro-Formal, §6.5]

---

## 7. COMPLETE UNIFIED TOOL & DATASET INVENTORY

### 7.1 Every Dataset Mentioned Across All Proposals

| # | Dataset | Scenes | Frames | Format | License | Best Used In | Source |
|---|---------|--------|--------|--------|---------|--------------|--------|
| 1 | **ScanNet v2** | 1,513 | 2.5M | RGB-D + mesh + poses | CC BY 4.0 Academic | A, C | [Perplexity; Claude; Proposal-B; Kiro-Formal] |
| 2 | **ScanNet++** | Extended | Extended | Enhanced annotations | Academic | A, C | [Perplexity] |
| 3 | **Matterport3D** | 90 | 194K | RGB-D + panoramas + mesh | Academic agreement | A, C | [Perplexity; Claude; Kiro-Formal] |
| 4 | **HM3D** | 1,000 | ~5M | Photorealistic 3D env | Apache 2.0 | A, C, E | [Perplexity; Claude; Kiro-Formal] |
| 5 | **Replica** | 18 | Synthetic | Mesh + HDR + semantics | MIT | A, C, E | [Perplexity; Claude; Proposal-B; Kiro-Formal] |
| 6 | **HyperSim** | 461 | 77K | Depth + normals + labels | MIT | A, C, E | [Perplexity; Proposal-B; Synth-v2; Kiro-Formal] |
| 7 | **3RScan** | 1,482 | 2.4M | Point cloud + change graphs | Academic | A, C | [Perplexity; Proposal-B] |
| 8 | **ARKitScenes** | 5,047 | >200M | LiDAR mesh + poses | Apple Academic | A, C (validation) | [Perplexity; Proposal-B; Synth-v2] |
| 9 | **HSSD** | 211 | ~18K objects | Synthetic interiors | CC BY-NC 4.0 | A, C, E | [Perplexity] |
| 10 | **AI2-THOR** | 120+ room types | Unlimited | Procedural simulator | Apache 2.0 | E | [Perplexity; Kiro-Formal] |
| 11 | **RealEstate10K** | 10M frames | 10M | YouTube video poses | Research | A (supplementary) | [Claude] |
| 12 | **Objaverse-XL** | 10M+ objects | — | 3D object models | Open (CC variants) | C, E | [Kiro-Formal; Synth-v2] |
| 13 | **SceneSplat-7K** | 7,916 | ~4.72M | Precomputed 3DGS | Non-commercial research | A, C (instant) | [Perplexity; Synth-v2] |

### 7.2 Every Reconstruction Tool Mentioned

| # | Tool | Method | Input | License | GPU | Approach | Source |
|---|------|--------|-------|---------|-----|----------|--------|
| 1 | **DUSt3R** | Feed-forward 3D | Arbitrary images | Apache 2.0 | 16GB+ | B, C | [Synth-v2; Proposal-B] |
| 2 | **MASt3R** | Pairwise matching | Image pairs | Apache 2.0 | 16GB+ | B, C | [Synth-v2; Proposal-B] |
| 3 | **COLMAP** | SfM + MVS | Any images | BSD | 8GB+ | B, C (fallback) | [Perplexity; Kiro-Formal; Proposal-B] |
| 4 | **OpenMVS** | Dense mesh | COLMAP output | AGPL | CPU/GPU | B, C | [Claude] |
| 5 | **Nerfstudio** | NeRF/3DGS framework | Images + poses | Apache 2.0 | 8GB+ | B, C, D | [Perplexity; Kiro-Formal; Proposal-B; Synth-v2] |
| 6 | **gsplat** | CUDA rasterizer | Nerfstudio backend | MIT | CUDA | B, C | [Synth-v2; Kiro-Formal] |
| 7 | **3D Gaussian Splatting** (official) | Gaussian primitives | Images + poses | Inria academic | 8GB+ | B, C | [Claude; Kiro-Formal] |
| 8 | **SuperSplat** | Web editor | PLY files | MIT | Browser | B, C | [Synth-v2; Kiro-Formal] |
| 9 | **Luma AI** | Cloud NeRF | Video | Commercial | Cloud | D (deprecated) | [Kiro-Formal; Perplexity; Synth-v2] |
| 10 | **Polycam** | Cloud photogrammetry | Images | Commercial | Cloud | D | [Perplexity; Kiro-Formal] |
| 11 | **Polyform** | Format converter | Polycam exports | MIT | CPU | D | [Perplexity] |

### 7.3 Every Depth Estimation Model

| # | Model | Type | Indoor δ1 | License | Role | Approach | Source |
|---|-------|------|-----------|---------|------|----------|--------|
| 1 | **UniDepth v2 (Large)** | Metric depth | 96.4% (SUN RGB-D) | Apache 2.0 | PRIMARY | B, C | [Synth-v2; Proposal-B] |
| 2 | **Depth Anything v2 (Large)** | Relative/metric | 98.3% (KITTI) | MIT | FALLBACK | B, C | [Synth-v2; Proposal-B] |
| 3 | **Metric3D v2** | Metric depth | Competitive | Academic | VALIDATION | B, C | [Synth-v2] |
| 4 | **Dataset GT** | Ground truth | 100% | — | PRIMARY | A | [Claude; Perplexity] |

### 7.4 Every Segmentation Tool

| # | Tool | Function | License | Approach | Source |
|---|------|----------|---------|----------|--------|
| 1 | **SAM2** | Video-native instance segmentation | Apache 2.0 | B, C | [Synth-v2; Proposal-B] |
| 2 | **Grounding DINO** | Text-to-box zero-shot detection | Apache 2.0 | B, C | [Synth-v2; Proposal-B] |
| 3 | **Grounded-SAM2** | Combined detection + segmentation | Apache 2.0 | B, C | [Synth-v2] |
| 4 | **Open3D-ML** | 3D instance segmentation | MIT | B, C (fallback) | [Perplexity] |

### 7.5 Every VLM Model & Strategy

| # | Model | Provider | Context | JSON Output | Cost (Batch) | Role | Approach | Source |
|---|-------|----------|---------|-------------|--------------|------|----------|--------|
| 1 | **Gemini 2.5 Pro** | Google | 1M tokens | Native schema | $0.625/$5 per MTok | PRIMARY | C | [Synth-v2] |
| 2 | **Gemini 2.5 Flash** | Google | 1M tokens | Native schema | $0.075/$1.25 per MTok | FAST / 80% scenes | C | [Synth-v2] |
| 3 | **Gemini 1.5 Pro** | Google | ~2M tokens | SDK schema | Standard rate | LEGACY / PoC | A, B, D | [Perplexity; Kiro-Formal; Proposal-B] |
| 4 | **GPT-4o** | OpenAI | 128K tokens | response_format | $2.50/$10 per MTok | FALLBACK | A, B, C | [Perplexity; Kiro-Formal; Claude; Proposal-B] |
| 5 | **GPT-4o-mini** | OpenAI | 128K tokens | response_format | Cheaper | Simple scenes | A, B | [Kiro-Formal] |
| 6 | **GPT-5.4** | OpenAI | 128K tokens | JSON mode | Higher | Complex reasoning | C (fallback) | [Synth-v2] |
| 7 | **GPT4Scene paradigm** | Research (HKU) | BEV + frames | Structured | N/A (research) | SOTA benchmark | B, C | [Kiro-Formal] |

### 7.6 Infrastructure & Storage

| Category | Options | Best For | Source |
|----------|---------|----------|--------|
| **Orchestration** | Apache Airflow, Prefect, Python scripts | Pipeline scheduling | [Perplexity, §Tool stack summary; Claude, §2 Tool Stack] |
| **3D Operations** | Open3D, PyTorch3D, trimesh, pycolmap | Projection, mesh I/O | [Perplexity, §Tool stack summary; Claude, §5 Day 3] |
| **Storage** | HDF5, SQLite, PostgreSQL (JSONB), MongoDB, Parquet, JSONL | Annotation records | [Perplexity, §Tool stack summary; Proposal-B, §3.2; Kiro-Formal, §3.4] |
| **Compute** | Local RTX 4090, Google Colab A100, Lambda Labs A100 ($1.10/hr) | Training/reconstruction | [Synth-v2, §6.1; Proposal-B, §2B; Kiro-Formal, §5] |
| **Object Store** | Lab NFS, S3, GCS | Raw dataset hosting (TB scale) | [Perplexity, §Tool stack summary] |

---

## 8. COMPREHENSIVE COST ANALYSIS

### 8.1 Compute Costs

| Resource | Spec | Upfront | Hourly | Best For | Source |
|----------|------|---------|--------|----------|--------|
| **Workstation GPU** | RTX 4090 24GB | $1,600 | — | Permanent lab setup | [Synth-v2, §6.1; Proposal-B, §2B] |
| **Google Colab** | A100 40GB | — | ~$10/day | PoC, short experiments | [Synth-v2, §6.1; Proposal-B, §6] |
| **Lambda Labs** | A100 40GB | — | $1.10/hr | Scale processing | [Synth-v2, §6.1] |
| **CPU Server** | 16+ cores | Lab existing | — | Dataset pairing, VLM batch prep | [Claude, §Appendix] |

### 8.2 VLM API Costs (Per 500 Scenes)

| Strategy | Model Mix | Input Cost | Output Cost | Tokens/Scene | Total | Source |
|----------|-----------|-----------|-------------|--------------|-------|--------|
| **Premium Only** | Gemini 2.5 Pro standard | $1.25/MTok | $10/MTok | ~2K in / ~500 out | ~$3,750 | [Synth-v2, §6.2] |
| **Batch Only (Pro)** | Gemini 2.5 Pro Batch | $0.625/MTok | $5/MTok | ~2K in / ~500 out | ~$1,875 | [Synth-v2, §6.2] |
| **Flash Only** | Gemini 2.5 Flash Batch | $0.075/MTok | $1.25/MTok | ~2K in / ~500 out | ~$200 | [Synth-v2, §6.2] |
| **TIERED (RECOMMENDED)** | 80% Flash + 20% Pro (Batch) | Mixed | Mixed | Mixed | **~$400** | [Synth-v2, §6.2] |
| **GPT-4o Fallback** | GPT-4o standard | $2.50/MTok | $10/MTok | ~2K in / ~500 out | ~$7,500 | [Synth-v2, §6.2] |

**Breakdown of Tiered Strategy:**
- 400 simple scenes (classrooms, offices) → Flash Batch: ~$160
- 100 complex scenes (labs, anatomy) → Pro Batch: ~$240
- **Total: ~$400 for 500 scenes** [Synth-v2, §6.2 TIERED STRATEGY]

### 8.3 Dataset Costs

| Dataset | Cost | Notes | Source |
|---------|------|-------|--------|
| HyperSim | $0 | MIT, instant | [Synth-v2, §6.3; Proposal-B, §6] |
| ScanNet | $0 | Academic, 1–3 day approval | [Synth-v2, §6.3; Proposal-B, §6] |
| Replica | $0 | MIT, Git LFS bandwidth only | [Synth-v2, §6.3; Proposal-B, §6] |
| ARKitScenes | $0 | Apple academic | [Synth-v2, §6.3] |
| HM3D | $0 | Apache 2.0 | [Proposal-B, §6] |
| HSSD | $0 | CC BY-NC 4.0 | [Proposal-B, §6] |
| Objaverse-XL | $0 | Open license, HuggingFace | [Synth-v2, §6.3; Kiro-Formal] |
| SceneSplat-7K | $0 | HuggingFace | [Synth-v2, §6.3] |
| AI2-THOR | $0 | Apache 2.0 | [Kiro-Formal] |

### 8.4 Software Costs

All core software is **free and open-source**:
- DUSt3R, MASt3R, COLMAP, Nerfstudio, gsplat, SuperSplat: **$0** [Synth-v2, §2.2; Kiro-Formal, §2.2]
- SAM2, Grounding DINO: **$0** [Synth-v2, §2.4]
- UniDepth v2, Depth Anything v2: **$0** [Synth-v2, §2.3]
- Open3D, PyTorch3D: **$0** [Perplexity, §Tool stack summary]

### 8.5 Total Estimated Budget (First Month)

| Approach | GPU | Cloud Compute | API | Storage | Total | Source |
|----------|-----|---------------|-----|---------|-------|--------|
| **A (Dataset-only)** | $0 (lab CPU) | $0 | $1,000–1,875 | $50 (external drive) | **$1,050–1,925** | [Synth-v2, §6.2; Claude, §Appendix] |
| **B (Custom-only)** | $1,600 (RTX 4090) | $50 (Colab backup) | $400 | $50 | **$2,100** | [Synth-v2, §6.1–6.2; Proposal-B, §6] |
| **C (Hybrid)** | $1,600 | $100 | $400 | $100 | **$2,200** | [Synth-v2, §6.1–6.3] |
| **D (API-first)** | $0 | $0 | $400 + $20/mo Polycam | $50 | **$470** | [Kiro-Formal, §5.4] |
| **E (Synthetic)** | $0 | $0 | $0 | $50 | **$50** | [Kiro-Formal, §6.4] |

---

## 9. COMPLEXITY & RISK MATRIX

### 9.1 Risk Register (All Approaches)

| Risk | Probability | Impact | Mitigation | Source |
|------|-------------|--------|------------|--------|
| DUSt3R fails on textureless scenes (whiteboards, bare walls) | Medium | High | Fallback to COLMAP + ArUco markers; filter scenes with <30% textured surfaces | [Synth-v2, §8 Risk Register; Proposal-B, §8] |
| UniDepth v2 scale ambiguity on custom captures | Medium | High | Use DUSt3R point cloud median for global scale; validate against known object sizes | [Synth-v2, §8 Risk Register] |
| ScanNet approval delayed/rejected | Low | Medium | HyperSim provides immediate alternative; 77K images sufficient for PoC | [Synth-v2, §8 Risk Register; Proposal-B, §8] |
| Gemini API rate limits at scale | Medium | Medium | Use Batch API; exponential backoff; shard across multiple projects | [Synth-v2, §8 Risk Register; Proposal-B, §8] |
| SAM2 misses small educational objects (test tubes, beakers) | Medium | High | Use 2× upsampling for small objects; refine category list | [Synth-v2, §8 Risk Register] |
| VLM spatial hallucination despite metadata | Low | High | Quality gate: ±20% distance tolerance; discard non-compliant outputs | [Synth-v2, §8 Risk Register; Proposal-B, §8] |
| GPU memory insufficient for large scenes | Medium | Medium | Use gsplat backend (4× memory reduction); process in overlapping chunks | [Synth-v2, §8 Risk Register; Kiro-Formal, §6] |
| Domain gap: HyperSim synthetic → real classrooms | High | Medium | Fine-tune UniDepth v2 on ScanNet real depth; domain randomization | [Synth-v2, §8 Risk Register] |
| COLMAP fails on low-texture custom video | Medium | High | Ensure >70% overlap; add textured markers; use DUSt3R primary | [Kiro-Formal, §6 Risk Mitigation] |
| Luma AI API discontinuation/shutdown | High | High | **Do NOT depend on Luma; use open-source stack** | [Synth-v2, §2.2 CRITICAL CORRECTION] |
| API cost overrun | Medium | Medium | Tiered Flash/Pro routing; batch processing; start with 100-scene validation | [Synth-v2, §6.2] |
| Data storage overflow (>5TB) | Medium | Medium | Store only keyframes; compress depth maps; use SPZ for splats | [Synth-v2, §8 Risk Register] |

### 9.2 Complexity Comparison by Layer

| Layer | Approach A | Approach B | Approach C | Approach D | Approach E | Source |
|-------|------------|------------|------------|------------|------------|--------|
| **Ingestion** | LOW | MEDIUM | MEDIUM | LOW | MEDIUM | [Synthesized from all docs] |
| **Reconstruction** | NONE | HIGH | MEDIUM | NONE | LOW | [Synthesized from all docs] |
| **Depth Estimation** | NONE | HIGH | MEDIUM | NONE | NONE | [Synthesized from all docs] |
| **Segmentation** | LOW | HIGH | MEDIUM | LOW | NONE | [Synthesized from all docs] |
| **2D-3D Pairing** | MEDIUM | HIGH | MEDIUM | MEDIUM | LOW | [Synthesized from all docs] |
| **VLM Annotation** | MEDIUM | MEDIUM | MEDIUM | MEDIUM | NONE | [Synthesized from all docs] |
| **Quality Control** | LOW | HIGH | MEDIUM | LOW | LOW | [Synthesized from all docs] |
| **Overall** | LOW | HIGH | MEDIUM | LOW | LOW | [Synthesized from all docs] |

---

## 10. IMPLEMENTATION ROADMAP OPTIONS

### 10.1 Fast Track (Mentor Demo in 1 Week)
**Best for:** Securing approval, proving feasibility
1. **Day 1:** Download HyperSim (instant) [Synth-v2, §7 Day 1; Proposal-B, §7 Day 1]
2. **Day 2:** Extract 100 frames + poses from 1 scene [Claude, §5 Day 2]
3. **Day 3:** Build pairing script (project 3D labels to 2D) [Claude, §5 Day 3]
4. **Day 4:** Annotate 10 frames via GPT-4o or Gemini (free tier) [Claude, §5 Day 4]
5. **Day 5:** Validate JSON schema + create visualization [Claude, §5 Day 5]
6. **Day 6:** Prepare 5-slide summary [Claude, §5 Day 5]
7. **Day 7:** **Mentor presentation** with live demo [Claude, §5 Day 5]

**Cost:** ~$0–10 (API free tier) [Claude, §5]  
**Deliverable:** 100 annotated frames from existing dataset [Claude, §5]

### 10.2 Standard Track (PoC in 1 Month)
**Best for:** Solid foundation, first training data
- **Week 1:** Environment + HyperSim validation [Synth-v2, §7 Week 1]
- **Week 2:** DUSt3R on custom video + ScanNet processing [Synth-v2, §7 Week 2]
- **Week 3:** VLM integration + cost optimization [Synth-v2, §7 Week 3]
- **Week 4:** Uni-RAG training corpus (10K triplets) [Synth-v2, §7 Week 4]

**Cost:** ~$400 (tiered VLM) + GPU time [Synth-v2, §6.2]  
**Deliverable:** 10,000 query-scene-answer triplets across 50 scenes [Synth-v2, §7 Week 4]

### 10.3 Full Research Track (3 Months)
**Best for:** Publication-quality dataset, comprehensive evaluation
- **Month 1:** All datasets ingested, reconstruction pipeline solid [Synth-v2, §7]
- **Month 2:** Scale to 500 scenes, validate against ARKitScenes LiDAR [Synth-v2, §7 Week 2]
- **Month 3:** Full Uni-RAG training, ablation studies, paper draft [Synth-v2, §7 Week 4]

**Cost:** ~$2,000–2,500 total [Synth-v2, §6]  
**Deliverable:** 100K+ annotated pairs, benchmark results, paper [Synth-v2, §9 Expected Deliverables]

---

## 11. MASTER COMPARISON TABLE

| Dimension | Approach A:<br>Dataset-First | Approach B:<br>Custom Recon | Approach C:<br>Hybrid<br>(RECOMMENDED) | Approach D:<br>API-First | Approach E:<br>Synthetic | Source |
|-----------|:---------------------------:|:---------------------------:|:--------------------------------------:|:------------------------:|:------------------------:|--------|
| **Start Time** | Day 1 | Day 3–5 | Day 1 | Day 1 | Day 2 | [Claude, §1; Proposal-B, §7; Kiro-Formal, §5] |
| **Initial Cost** | $0 | $1,600 (GPU) | $1,600 (GPU) | $20–50/mo | $0 | [Synth-v2, §6; Kiro-Formal, §5.4] |
| **Per-Scene Cost** | ~$2–4 (VLM only) | ~$0.80 (compute) + $0.80 (VLM) | ~$0.80 (compute) + $0.80 (VLM) | ~$0.50 (API) | $0 | [Synth-v2, §6.2; Kiro-Formal, §5.4] |
| **Data Volume** | 2.5M+ frames | Limited by capture | Unlimited (hybrid) | Limited by API quota | Unlimited | [Claude, §Appendix; Synth-v2, §3.1; Kiro-Formal, §6.4] |
| **Domain Control** | ❌ Fixed | ✅ Full | ✅ Full | ✅ Full | ✅ Full | [Claude, §1; Proposal-B, §1] |
| **Ground Truth Quality** | ★★★★★ Perfect | ★★★☆☆ Estimated | ★★★★☆ Mixed | ★★★☆☆ Cloud-dependent | ★★★★★ Perfect | [Synth-v2, §1; Kiro-Formal, §6.5] |
| **Real-World Realism** | ★★★★★ Real | ★★★★★ Real | ★★★★★ Real | ★★★★☆ Good | ★★☆☆☆ Synthetic | [Kiro-Formal, §6.5] |
| **Educational Specificity** | ★★☆☆☆ Generic | ★★★★★ Custom | ★★★★★ Custom | ★★★★★ Custom | ★★★★★ Custom | [Claude, §1; Proposal-B, §1] |
| **Reconstruction Complexity** | NONE | HIGH | MEDIUM | NONE | LOW | [Synthesized] |
| **Annotation Reliability** | HIGH (GT-backed) | MEDIUM (metadata-first) | HIGH (hybrid validation) | MEDIUM | HIGHEST (no VLM needed) | [Synth-v2, §1 Dimension 5; Kiro-Formal, §6.5] |
| **Scalability** | ★★★★★ Instant | ★★☆☆☆ Slow per scene | ★★★★☆ Good | ★★★☆☆ API-limited | ★★★★★ Unlimited | [Claude, §Appendix; Synth-v2, §3.1] |
| **Mentor Impressiveness** | ★★★☆☆ Standard | ★★★★★ Novel | ★★★★★ Comprehensive | ★★★☆☆ Quick but shallow | ★★★☆☆ Synthetic concern | [Synthesized] |
| **Best For** | Immediate volume | Domain-specific scenes | **Balanced research** | Quick demo | Ablation / augmentation | [Synthesized] |
| **Risk Level** | LOW | HIGH | MEDIUM | MEDIUM | LOW | [Synth-v2, §8; Kiro-Formal, §6] |
| **Uni-RAG Ready** | ✅ Yes | ✅ Yes | ✅✅ Best | ⚠️ Partial | ⚠️ Domain gap | [Synthesized] |

---

## 12. FINAL RECOMMENDATION & NEXT STEPS

### 12.1 Recommended Path: Approach C (Hybrid Synthesis)

After analyzing all 5 source documents, **Approach C is definitively recommended** for the following reasons:

1. **Eliminates single points of failure:** If ScanNet is delayed, HyperSim provides Day 1 data. If DUSt3R fails on a textureless wall, COLMAP + ArUco markers serve as fallback. [Synth-v2, §8 Risk Register; Proposal-B, §8]
2. **Optimizes cost:** The tiered VLM strategy (80% Flash / 20% Pro) reduces annotation costs by 5× compared to premium-only. [Synth-v2, §6.2]
3. **Validates quality:** ARKitScenes LiDAR ground truth allows quantification of no-LiDAR pipeline error (target RMSE < 15cm). [Synth-v2, §7 Week 2]
4. **Scales across domains:** Existing datasets provide generic indoor diversity; custom capture adds educational specificity; Objaverse-XL adds object-level detail. [Synth-v2, §3.1; Kiro-Formal, §2.1]
5. **Prevents hallucination:** The metadata-first VLM strategy (geometry from math, semantics from VLM) is validated by the Real-3DQA benchmark showing 3D-LLMs collapse on multi-view rotation tasks. [Synth-v2, §1 Dimension 5]

### 12.2 Key Differentiators to Highlight to Mentor

1. **DUSt3R over COLMAP as primary** — eliminates calibration barrier (core constraint) [Synth-v2, §1 Dimension 1; Proposal-B, §1]
2. **UniDepth v2 for metric depth** — first proposal to solve "metric scale without sensors" [Synth-v2, §1 Dimension 2]
3. **Metadata-first VLM** — eliminates spatial hallucination risk [Synth-v2, §1 Dimension 5; Proposal-B, §4B]
4. **BEV as 4th input modality** — global context augmentation [Synth-v2, §1 Dimension 5; Proposal-B, §3.2 Step 4]
5. **HyperSim-first dataset strategy** — Day 1 validation without approval delays [Synth-v2, §1 Dimension 3; Proposal-B, §2A]
6. **Corrected cost analysis** — realistic $400 for 500 scenes (not $0.50) [Synth-v2, §2.5 CORRECTION; §6.2]
7. **Validation against ARKitScenes** — LiDAR ground truth quantifies error [Synth-v2, §1 Dimension 3; §7 Week 2]
8. **SPZ compression** — 90% size reduction for web deployment [Synth-v2, §2.2; Kiro-Formal, §3.2 Step 4]
9. **Tiered VLM routing** — Flash for simple, Pro for complex [Synth-v2, §6.2]
10. **Explicit quality gates** — CLIP, depth coverage, pose confidence, VLM consistency [Synth-v2, §1 Dimension 6; Proposal-B, §3.2 Quality Filter]

### 12.3 Immediate Action Items (Today)

```bash
# 1. Install core stack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install nerfstudio
pip install git+https://github.com/naver/dust3r.git
pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git
pip install transformers accelerate
```
[Synth-v2, §7 Day 1; Proposal-B, §7 Day 1]

```bash
# 2. Download HyperSim (instant, MIT license)
git clone https://github.com/apple/ml-hypersim.git
python ml-hypersim/code/python/tools/dataset_download.py \
  --scene_names ai_001_001 ai_001_002 ai_001_003 \
  --downloads_dir /data/hypersim
```
[Synth-v2, §7 Day 1; Proposal-B, §7 Day 1]

```bash
# 3. Request ScanNet (start approval process)
# Visit: http://www.scan-net.org/ → "Get Access"
```
[Synth-v2, §7 Day 1; Proposal-B, §7 Day 1]

```bash
# 4. Record test video (2-minute slow orbit of lab/office)
# Extract frames: ffmpeg -i video.mp4 -vf fps=3 frames/%06d.jpg
```
[Synth-v2, §3.1 Track B; Proposal-B, §3.1 Track B]

```bash
# 5. Set up Gemini API key for batch annotation testing
# https://aistudio.google.com/app/apikey
```
[Synth-v2, §7 Week 3]

### 12.4 Expected Deliverables for Mentor Review

| # | Deliverable | Timeline | Metric | Source |
|---|-------------|----------|--------|--------|
| 1 | Pipeline Demo Notebook | Week 1 | Video → 3D splat → JSON in <15 min | [Synth-v2, §9] |
| 2 | Reconstruction Accuracy Report | Week 2 | DUSt3R vs. ARKitScenes LiDAR: RMSE < 15cm | [Synth-v2, §9; §7 Week 2] |
| 3 | Annotated Dataset v1 | Week 3 | 1,000 frames across 20 scenes, 3 domains | [Synth-v2, §9] |
| 4 | Cost Analysis Report | Week 3 | Validated $/scene with tiered VLM strategy | [Synth-v2, §9] |
| 5 | Uni-RAG Training Corpus | Week 4 | 10,000 query-scene-answer triplets with 3D coordinates | [Synth-v2, §9; §7 Week 4] |
| 6 | Qualitative Results Gallery | Week 4 | Side-by-side: reconstruction, BEV, annotated frames | [Synth-v2, §9] |

---

*This document synthesizes 5 independent research proposals into a single decision framework. Every tool, dataset, cost estimate, and architectural decision has been preserved with inline citations for comprehensive mentor discussion.*

**Sources Referenced:**
- [Synth-v2] Kiro-SYNTHESIZED_3D_Pipeline_Proposal_v2.md
- [Perplexity] perplexity.md
- [Proposal-B] 3D-Pipeline-Research-Proposal.md
- [Claude] claude-response.md
- [Kiro-Formal] Kiro-research_proposal_2d3d_pipeline.md
