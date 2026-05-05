# FINAL RESEARCH PROPOSAL
## 2D-to-3D Scene Data Generation & Annotation Pipeline for 3D-Grounded Uni-RAG (AIED)
### Multi-Approach Synthesis & Mentor Presentation Document
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

**Goal:** Build a training data pipeline for a 3D-grounded Uni-RAG system in AIED (Artificial Intelligence in Education) that can retrieve and generate interactive 3D visuals based on user queries (e.g., "show me the skeleton model relative to the teacher's desk").

**Hard Constraints:**
- **No LiDAR hardware** (no iPhone Pro, no depth sensors)
- **Academic budget** (prefer free/open-source, minimal API spend)
- **University lab setting** (must justify to mentor with concrete deliverables)
- **Need both 2D images AND 3D scene representations** with mathematical correspondence
- **Rich spatial annotations** required for retrieval training

**What the Pipeline Must Produce:**
1. 2D-3D paired data (image + camera pose + 3D representation)
2. Object-level spatial annotations (centroids, bounding boxes, relationships)
3. Natural language captions grounded in metric 3D space
4. Training triplets for Uni-RAG: (query, scene, answer, 3D_asset)

---

## 2. APPROACH A: PURE EXISTING DATASET LEVERAGE

### 2.1 Philosophy
Use pre-existing, academically licensed datasets that **already contain** 2D-3D pairs, camera poses, meshes, and semantic labels. **Zero reconstruction code required for Day 1–3.**

### 2.2 Datasets Used

| Dataset | Scenes | Frames | 3D Format | License | Acquisition | AIED Relevance |
|---------|--------|--------|-----------|---------|-------------|----------------|
| **ScanNet v2** | 1,513 | 2.5M RGB-D | Mesh + point cloud + poses | CC BY 4.0 (academic) | 1–3 day approval | ★★★★★ Classroom-scale indoor |
| **Matterport3D** | 90 buildings | 194K RGB-D | Semantic mesh + panoramas | Academic agreement | License agreement | ★★★★☆ Room-scale spatial reasoning |
| **HM3D** | 1,000 | ~5M views | Photorealistic 3D env | Apache 2.0 | Instant | ★★★★☆ Navigation-scale |
| **Replica** | 18 | Synthetic | Mesh + HDR + semantics | MIT | Instant (Git LFS) | ★★★☆☆ High fidelity, limited scale |
| **HyperSim** | 461 rooms | 77K images | Dense depth + normals + labels | MIT | Instant | ★★★★☆ Photorealistic synthetic |
| **3RScan** | 1,482 | 2.4M RGB | Point cloud + change graphs | Academic | Registration | ★★★☆☆ Object re-localization |
| **ARKitScenes** | 5,047 | >200M frames | LiDAR mesh (pre-captured) | Apple Academic | Instant | ★★★★☆ High-quality indoor validation |
| **HSSD** | 211 | ~18K objects | Synthetic interior scenes | CC BY-NC 4.0 | Instant | ★★★★☆ Embodied AI compatible |
| **SceneSplat-7K** | 7,916 | ~4.72M | Precomputed 3D Gaussian Splats | Non-commercial research | HuggingFace | ★★★★★ Ready-to-use 3DGS |

### 2.3 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INGESTION                                                  │
│  ├── ScanNet .sens → SensReader.py → RGB + depth + poses   │
│  ├── Matterport3D → download_mp.py → region manifests      │
│  ├── Replica/HM3D/HSSD → Habitat-compatible loaders        │
│  └── SceneSplat-7K → HuggingFace → pre-built Gaussian scenes│
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PROCESSING (No GPU required initially)                     │
│  ├── Frame sampling: 1 fps or keyframe-based               │
│  ├── Pose parsing: Load pre-computed T_i (4×4 camera-to-world)│
│  ├── Mesh loading: .ply / .obj via Open3D                   │
│  └── 3D annotation read: semantic labels, instance IDs      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  2D-3D PAIRING (Mathematical Core)                          │
│  ├── For each frame i:                                      │
│  │   K = intrinsic matrix (3×3)                             │
│  │   T_i = pose matrix (4×4)                                │
│  │   Project 3D bbox corners: p_2d = K × (T_i⁻¹ × P_world) │
│  ├── Compute distances: ||centroid_3d - camera_pos||₂       │
│  ├── Extract spatial relations: LEFT_OF, IN_FRONT_OF, etc.  │
│  └── Build metadata_json per frame                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  VLM ANNOTATION                                             │
│  ├── Input: RGB frame + metadata_json + schema prompt      │
│  ├── Model: GPT-4o or Gemini 1.5 Pro                       │
│  ├── Output: JSON with global_caption, objects, relations  │
│  └── Storage: JSONL/Parquet + SQLite index                 │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Tools Required
- **Open3D** / **PyTorch3D** / **trimesh** — 3D I/O and projection
- **pycolmap** — Python bindings for COLMAP formats
- **ScanNet SDK** — SensReader.py for .sens decoding
- **Habitat-sim** — For HM3D/HSSD/Replica scene loading
- **VLM API** — GPT-4o or Gemini 1.5 Pro

### 2.5 Complexity Analysis
| Factor | Level | Notes |
|--------|-------|-------|
| Setup | LOW | Download + SDK install only |
| Compute | LOW–MEDIUM | CPU for pairing; GPU only for VLM batch |
| Code | MEDIUM | Need unified dataset loaders + projection math |
| Data Volume | HIGH | 2.5M+ frames available immediately |
| Quality | HIGH | Ground-truth poses and geometry |

### 2.6 Cost
- **Datasets:** $0 (academic licenses)
- **Compute:** $0 (laptop/CPU for pairing; lab GPU for optional rendering)
- **VLM Annotation:** ~$2,000–3,750 for full ScanNet (2.5M frames) at standard API rates; ~$1,000–1,875 with Batch API
- **Storage:** ~2TB for ScanNet full; ~2.76TB for SceneSplat-7K

### 2.7 Pros & Cons
**Pros:**
- Immediate start (no waiting for reconstruction)
- Ground-truth accuracy for poses and geometry
- No calibration or scale ambiguity
- 2.5M+ training pairs possible from ScanNet alone

**Cons:**
- Limited to existing scene types (mostly residential, not specifically educational)
- Cannot add custom lab/classroom environments
- Domain gap: synthetic (HyperSim) vs. real classrooms
- Licensing restrictions (non-commercial)

---

## 3. APPROACH B: PURE CUSTOM RECONSTRUCTION (NO-LIDAR)

### 3.1 Philosophy
Generate 3D scenes entirely from **smartphone/DSLR imagery or video** using state-of-the-art no-LiDAR reconstruction. Full control over domain (educational environments).

### 3.2 Input Requirements
- **Video:** 30–60s slow orbit, 1080p+, good lighting, avoid reflections
- **Images:** 50–200 photos, >70% overlap, varying heights/angles
- **Extraction:** `ffmpeg -i video.mp4 -vf "fps=3,scale=1280:720" frames/%06d.jpg`

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

#### Option B3: Feed-Forward Sparse Capture
- **MASt3R** — DUSt3R successor, pairwise dense matching, works with as few as 2 images
- Best for quick captures of small educational props

### 3.4 Depth Estimation (Critical for Metric Scale)
Since custom captures lack ground-truth depth:

| Model | Role | Indoor Accuracy | Speed | License |
|-------|------|-----------------|-------|---------|
| **UniDepth v2 (Large)** | PRIMARY metric depth | SUN RGB-D: 96.4% δ1 | ~5 FPS on A100 | Apache 2.0 |
| **Depth Anything v2 (Large)** | FALLBACK + relative depth | KITTI: 98.3% δ1 | ~10 FPS on A100 | MIT |
| **Metric3D v2** | VALIDATION baseline | Competitive | Medium | Academic |

**Scale Recovery:** Use DUSt3R point cloud median depth to globally scale monocular depth predictions.

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

**Quality Filters:**
- Discard masks with area < 0.5% of image
- Discard masks where mean depth confidence < 0.7
- Filter at object boundaries to reduce noise

### 3.6 VLM Annotation Strategy

**Metadata-First Approach (Prevents Hallucination):**
- **Inputs to VLM:** RGB keyframe + Colorized depth map (jet colormap) + BEV composite + STRUCTURED METADATA STRING
- **Metadata String (ground truth facts):**
  ```
  SCENE: university_lecture_hall_001
  CAMERA_HEIGHT: 1.45m above floor
  FOV: 72.3° horizontal
  FLOOR_AREA: 8.2m × 12.5m
  OBJECTS: whiteboard_01 | centroid=(0.2, 1.5, 8.1)m | distance=8.2m | area=4.2m²
  SPATIAL_RELATIONSHIPS: desk_01 is 1.2m LEFT_OF chair_01
  ```
- **VLM Role:** "Semantic Translator" — converts metric facts to intuitive language ("arm's reach", "across the room")
- **Strict Rules:**
  - DO: Generate intuitive spatial descriptions
  - DO: Identify educational context
  - DO NOT: Invent distances or positions
  - DO NOT: Hallucinate objects not in metadata

### 3.7 Complexity Analysis
| Factor | Level | Notes |
|--------|-------|-------|
| Setup | MEDIUM | Install DUSt3R, Nerfstudio, SAM2, depth models |
| Compute | HIGH | 16GB+ VRAM for DUSt3R; 24GB ideal for large scenes |
| Code | HIGH | Need reconstruction, depth, segmentation, unprojection pipelines |
| Data Volume | LOW–MEDIUM | Limited by capture time (1 scene ≈ 1 hour processing) |
| Quality | MEDIUM–HIGH | Depends on capture quality; textureless surfaces problematic |

### 3.8 Cost
- **Software:** $0 (all open-source)
- **Compute:** RTX 4090 ($1,600 one-time) or Colab A100 (~$10/day)
- **Cloud API (Luma/Polycam fallback):** Luma 500 free splats/month; Polycam $20/month unlimited
- **VLM Annotation:** ~$400 for 500 scenes (tiered Flash/Pro strategy)
- **Time:** ~45 min per scene on RTX 3090; ~20 min on A100

### 3.9 Pros & Cons
**Pros:**
- Full domain control (scan actual classrooms, labs, anatomy models)
- No dataset licensing restrictions on captured content
- Scales to any educational environment
- DUSt3R eliminates calibration barrier entirely

**Cons:**
- Reconstruction fails on textureless surfaces (whiteboards, bare walls)
- Metric scale ambiguity without known reference objects
- Computationally intensive per scene
- Requires good capture technique (overlap, lighting)

---

## 4. APPROACH C: HYBRID SYNTHESIS (RECOMMENDED)

### 4.1 Philosophy
Combine **Track A (Existing Datasets)** for immediate volume + **Track B (Custom Reconstruction)** for domain specificity + **Track C (Object Assets)** for populating scenes. This is the converged architecture from deep multi-agent analysis.

### 4.2 Three Parallel Tracks

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HYBRID PIPELINE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TRACK A: EXISTING DATASETS          TRACK B: CUSTOM CAPTURE           │
│  ├─ HyperSim (Day 1, MIT)            ├─ Smartphone video/photos        │
│  ├─ ScanNet (1–3 day approval)       ├─ FFmpeg extraction (fps=3)     │
│  ├─ Replica (instant)                ├─ DUSt3R → pointmap + poses      │
│  ├─ ARKitScenes (validation)         ├─ UniDepth v2 → metric depth    │
│  └─ SceneSplat-7K (instant 3DGS)     ├─ SAM2 + Grounding DINO         │
│                                      └─ Nerfstudio Splatfacto          │
│                                                                         │
│  TRACK C: OBJECT ASSETS                                                 │
│  └─ Objaverse-XL (10M+ objects, HuggingFace)                           │
│      → Filter by "education", "anatomy", "chemistry" tags               │
│      → Populate synthetic scenes with domain-specific props             │
│                                                                         │
│  UNIFIED PROCESSING LAYER                                               │
│  ├─ 2D-3D Pairing (projection math)                                    │
│  ├─ Spatial Graph Construction                                         │
│  ├─ BEV Generation (top-down XZ projection)                            │
│  └─ Quality Gates (CLIP, depth coverage, pose confidence)              │
│                                                                         │
│  ANNOTATION LAYER                                                       │
│  ├─ Metadata-First VLM (Gemini 2.5 Pro / Flash)                        │
│  ├─ BEV as 4th input modality                                          │
│  ├─ Tiered routing: Flash (80%) / Pro (20%)                            │
│  └─ Quality Gate: ±20% distance tolerance check                        │
│                                                                         │
│  OUTPUT                                                                 │
│  └─ Uni-RAG Training Corpus: (query, scene, answer, 3D_asset)          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Corrected & Validated Tool Stack

#### Datasets
| Dataset | Scenes | License | Download | Role |
|---------|--------|---------|----------|------|
| **HyperSim** | 461 rooms, 77K images | MIT | Instant | PRIMARY — Day 1 validation, perfect ground truth |
| **ScanNet v2** | 1,513 scans, 2.5M frames | Academic | 1–3 days | SECONDARY — real-world indoor diversity |
| **Replica** | 18 scenes | MIT | Fast (Git LFS) | TERTIARY — high-fidelity Habitat-compatible |
| **ARKitScenes** | 5,047 scenes, >200M frames | Apple Academic | Moderate | VALIDATION — LiDAR ground truth benchmark |
| **Objaverse-XL** | 10M+ objects | Open (CC variants) | Fast (HF) | OBJECT DIVERSITY — educational props/models |
| **SceneSplat-7K** | 7,916 scenes | Non-commercial | HuggingFace | INSTANT 3DGS — skip training |

#### 3D Reconstruction (No LiDAR)
| Tool | Role | License | Input | Output | GPU |
|------|------|---------|-------|--------|-----|
| **DUSt3R** | PRIMARY reconstruction | Apache 2.0 | Arbitrary images | Pointmap + poses + confidence | 16GB+ VRAM |
| **MASt3R** | PAIRWISE matching | Apache 2.0 | Image pairs | Dense matching + 3D points | 16GB+ VRAM |
| **COLMAP** | FALLBACK reconstruction | BSD | Sequential images | Sparse point cloud + poses | 8GB+ VRAM |
| **Nerfstudio (Splatfacto)** | Gaussian Splat training/export | Apache 2.0 | Images + poses | PLY + transforms.json | 8GB+ VRAM |
| **SuperSplat** | Post-processing editor | MIT | PLY | Cleaned PLY/SPZ | Browser |
| **gsplat** | CUDA backend (4× memory reduction) | MIT | Nerfstudio backend | Optimized training | CUDA |

**CRITICAL CORRECTION:** Luma AI has pivoted to video generation; free tier limited to 30 watermarked generations/month with no commercial use. **Removed as core stack.** Retained only as rapid prototyping fallback.

#### Depth Estimation
| Model | Role | Indoor Accuracy | Speed |
|-------|------|-----------------|-------|
| **UniDepth v2 (Large)** | PRIMARY metric depth | SUN RGB-D: 96.4% δ1 | ~5 FPS on A100 |
| **Depth Anything v2 (Large)** | FALLBACK + relative depth | KITTI: 98.3% δ1 | ~10 FPS on A100 |
| **Metric3D v2** | VALIDATION baseline | Competitive | Medium |

#### Segmentation
| Tool | Role | Notes |
|------|------|-------|
| **SAM2** | Instance segmentation | Video-native, released July 2024 |
| **Grounding DINO** | Text-to-box detection | Zero-shot object detection |
| **Grounded-SAM2** | Combined pipeline | Chain SAM2 + Grounding DINO if combined not released |

#### VLM Annotation
| Model | Role | Cost (Batch API) | Context | Structured Output |
|-------|------|------------------|---------|-------------------|
| **Gemini 2.5 Pro** | PRIMARY annotator | $0.625/MTok in, $5/MTok out | 1M tokens | Native JSON schema |
| **Gemini 2.5 Flash** | FAST annotator | $0.075/MTok in, $1.25/MTok out | 1M tokens | Native JSON schema |
| **GPT-4o** | FALLBACK | $2.50/MTok in, $10/MTok out | 128K tokens | response_format: json_object |
| **GPT-5.4** | Complex reasoning fallback | Higher | 128K tokens | JSON mode |

**CORRECTED COST:** ~$400 for 500 scenes (tiered: 80% Flash @ ~$160 + 20% Pro @ ~$240). NOT $0.50 as incorrectly claimed in early drafts.

### 4.4 Quality Gates (Adopted from Proposal A)
- **CLIP score ≥ 0.25** — discard blurry/dark frames
- **Depth coverage ≥ 70%** — discard frames where depth fails
- **Pose confidence threshold** — filter DUSt3R low-confidence reconstructions
- **Deduplication** — cosine similarity < 0.95 between adjacent frames
- **VLM consistency check** — verify VLM distances match metadata ±20%
- **Object completeness** — all metadata objects must appear in output

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

### 4.7 Week-by-Week Execution Plan

**Week 1: Foundation**
- Day 1: Install core stack (PyTorch, Nerfstudio, DUSt3R, Depth Anything V2)
- Day 2–3: Download HyperSim (instant, MIT) + request ScanNet
- Day 4–5: Validate DUSt3R on HyperSim images vs. ground truth
- Day 6–7: End-to-end PoC on 1 HyperSim scene → validate JSON schema

**Week 2: Scale & Validation**
- Process 10 HyperSim scenes → 500 annotated frames
- Download ScanNet (should be approved)
- Process 5 ScanNet scenes → validate against ground truth poses
- Download ARKitScenes → run DUSt3R → compare to LiDAR ground truth
- Target: RMSE < 15cm for indoor scenes

**Week 3: VLM Integration & Cost Optimization**
- Set up Gemini API with Batch processing
- Implement metadata string builder + BEV generation
- Run annotation on 100 scenes (Flash vs. Pro comparison)
- Implement tiered routing: simple → Flash, complex → Pro

**Week 4: Uni-RAG Training Data**
- Convert annotations to training triplets
- Generate negative samples for contrastive learning
- Export to Parquet for PyTorch DataLoader
- Target: 10,000 training triplets across 50 scenes, 4 domains

---

## 5. APPROACH D: API-FIRST RAPID PROTOTYPING

### 5.1 Philosophy
Use managed cloud APIs for 3D reconstruction to achieve **same-day results** without local GPU infrastructure. Best for proof-of-concept demos when lab GPU is unavailable.

### 5.2 Tools
| Service | Cost | Input | Output | Limitations |
|---------|------|-------|--------|-------------|
| **Luma AI** | 30 free generations/month (watermarked) | Smartphone video | GLB/USD/PLY Gaussian splat | Pivoted to video generation; NOT viable for academic research |
| **Polycam** | $20/month unlimited | 20–200 images | GLTF/OBJ/PLY | API-dependent; ToS requires attribution for public display |
| **Polyform** (Polycam OSS) | Free | Polycam exports | NeRF format conversion | MIT-licensed tooling |

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

### 5.4 Complexity & Cost
- **Setup:** MINIMAL (web upload)
- **Compute:** NONE locally
- **Time:** Minutes per scene
- **Cost:** $20–50/month for unlimited Polycam; Luma free tier insufficient
- **Risk:** API dependence, data privacy, ToS restrictions, watermarking

### 5.5 When to Use
- Lab GPU unavailable during first week
- Need mentor demo in <24 hours
- Scanning sensitive educational environments where code setup is undesirable
- **NOT recommended as core research infrastructure**

---

## 6. APPROACH E: SYNTHETIC PROCEDURAL GENERATION

### 6.1 Philosophy
Generate **unlimited synthetic data** using simulators and procedural engines. Addresses domain gap by creating idealized educational environments.

### 6.2 Tools

| Tool | Type | License | Capability |
|------|------|---------|------------|
| **AI2-THOR** | Procedural simulator | Apache 2.0 | 120+ room types, programmatic viewpoint generation, physics |
| **Hypersim** | Synthetic dataset (static) | MIT | 461 pre-rendered rooms with perfect ground truth |
| **Habitat-sim** | Embodied AI simulator | Apache 2.0 | HM3D/HSSD/Replica integration, arbitrary camera trajectories |
| **Blender + Objaverse-XL** | Manual/procedural | CC variants | Populate custom scenes with 10M+ educational objects |

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

### 6.4 Complexity & Cost
- **Setup:** MEDIUM (simulator install)
- **Compute:** MEDIUM (CPU for AI2-THOR; GPU for rendering)
- **Data Volume:** UNLIMITED (programmatic generation)
- **Quality:** HIGH (perfect ground truth) but domain gap to real images

### 6.5 Pros & Cons
**Pros:**
- Infinite data generation
- Perfect ground truth (no VLM needed for spatial annotations)
- Zero API costs for annotations
- Full control over educational content

**Cons:**
- Domain gap: synthetic → real transfer required
- Not photorealistic enough for some VLMs
- Requires simulator expertise
- May not impress mentor without real-world validation

---

## 7. COMPLETE UNIFIED TOOL & DATASET INVENTORY

### 7.1 Every Dataset Mentioned Across All Proposals

| # | Dataset | Scenes | Frames | Format | License | Best Used In |
|---|---------|--------|--------|--------|---------|--------------|
| 1 | **ScanNet v2** | 1,513 | 2.5M | RGB-D + mesh + poses | CC BY 4.0 Academic | A, C |
| 2 | **ScanNet++** | Extended | Extended | Enhanced annotations | Academic | A, C |
| 3 | **Matterport3D** | 90 | 194K | RGB-D + panoramas + mesh | Academic agreement | A, C |
| 4 | **HM3D** | 1,000 | ~5M | Photorealistic 3D env | Apache 2.0 | A, C, E |
| 5 | **Replica** | 18 | Synthetic | Mesh + HDR + semantics | MIT | A, C, E |
| 6 | **HyperSim** | 461 | 77K | Depth + normals + labels | MIT | A, C, E |
| 7 | **3RScan** | 1,482 | 2.4M | Point cloud + change graphs | Academic | A, C |
| 8 | **ARKitScenes** | 5,047 | >200M | LiDAR mesh + poses | Apple Academic | A, C (validation) |
| 9 | **HSSD** | 211 | ~18K objects | Synthetic interiors | CC BY-NC 4.0 | A, C, E |
| 10 | **AI2-THOR** | 120+ room types | Unlimited | Procedural simulator | Apache 2.0 | E |
| 11 | **RealEstate10K** | 10M frames | 10M | YouTube video poses | Research | A (supplementary) |
| 12 | **Objaverse-XL** | 10M+ objects | — | 3D object models | Open (CC variants) | C, E |
| 13 | **SceneSplat-7K** | 7,916 | ~4.72M | Precomputed 3DGS | Non-commercial research | A, C (instant) |

### 7.2 Every Reconstruction Tool Mentioned

| # | Tool | Method | Input | License | GPU | Approach |
|---|------|--------|-------|---------|-----|----------|
| 1 | **DUSt3R** | Feed-forward 3D | Arbitrary images | Apache 2.0 | 16GB+ | B, C |
| 2 | **MASt3R** | Pairwise matching | Image pairs | Apache 2.0 | 16GB+ | B, C |
| 3 | **COLMAP** | SfM + MVS | Any images | BSD | 8GB+ | B, C (fallback) |
| 4 | **OpenMVS** | Dense mesh | COLMAP output | AGPL | CPU/GPU | B, C |
| 5 | **Nerfstudio** | NeRF/3DGS framework | Images + poses | Apache 2.0 | 8GB+ | B, C, D |
| 6 | **gsplat** | CUDA rasterizer | Nerfstudio backend | MIT | CUDA | B, C |
| 7 | **3D Gaussian Splatting** (official) | Gaussian primitives | Images + poses | Inria academic | 8GB+ | B, C |
| 8 | **SuperSplat** | Web editor | PLY files | MIT | Browser | B, C |
| 9 | **Luma AI** | Cloud NeRF | Video | Commercial | Cloud | D (deprecated) |
| 10 | **Polycam** | Cloud photogrammetry | Images | Commercial | Cloud | D |
| 11 | **Polyform** | Format converter | Polycam exports | MIT | CPU | D |

### 7.3 Every Depth Estimation Model

| # | Model | Type | Indoor δ1 | License | Role | Approach |
|---|-------|------|-----------|---------|------|----------|
| 1 | **UniDepth v2 (Large)** | Metric depth | 96.4% (SUN RGB-D) | Apache 2.0 | PRIMARY | B, C |
| 2 | **Depth Anything v2 (Large)** | Relative/metric | 98.3% (KITTI) | MIT | FALLBACK | B, C |
| 3 | **Metric3D v2** | Metric depth | Competitive | Academic | VALIDATION | B, C |
| 4 | **Dataset GT** | Ground truth | 100% | — | PRIMARY | A |

### 7.4 Every Segmentation Tool

| # | Tool | Function | License | Approach |
|---|------|----------|---------|----------|
| 1 | **SAM2** | Video-native instance segmentation | Apache 2.0 | B, C |
| 2 | **Grounding DINO** | Text-to-box zero-shot detection | Apache 2.0 | B, C |
| 3 | **Grounded-SAM2** | Combined detection + segmentation | Apache 2.0 | B, C |
| 4 | **Open3D-ML** | 3D instance segmentation | MIT | B, C (fallback) |

### 7.5 Every VLM Model & Strategy

| # | Model | Provider | Context | JSON Output | Cost (Batch) | Role | Approach |
|---|-------|----------|---------|-------------|--------------|------|----------|
| 1 | **Gemini 2.5 Pro** | Google | 1M tokens | Native schema | $0.625/$5 per MTok | PRIMARY | C |
| 2 | **Gemini 2.5 Flash** | Google | 1M tokens | Native schema | $0.075/$1.25 per MTok | FAST / 80% scenes | C |
| 3 | **Gemini 1.5 Pro** | Google | ~2M tokens | SDK schema | Standard rate | LEGACY / PoC | A, B, D |
| 4 | **GPT-4o** | OpenAI | 128K tokens | response_format | $2.50/$10 per MTok | FALLBACK | A, B, C |
| 5 | **GPT-4o-mini** | OpenAI | 128K tokens | response_format | Cheaper | Simple scenes | A, B |
| 6 | **GPT-5.4** | OpenAI | 128K tokens | JSON mode | Higher | Complex reasoning | C (fallback) |
| 7 | **GPT4Scene paradigm** | Research (HKU) | BEV + frames | Structured | N/A (research) | SOTA benchmark | B, C |

### 7.6 Infrastructure & Storage

| Category | Options | Best For |
|----------|---------|----------|
| **Orchestration** | Apache Airflow, Prefect, Python scripts | Pipeline scheduling |
| **3D Operations** | Open3D, PyTorch3D, trimesh, pycolmap | Projection, mesh I/O |
| **Storage** | HDF5, SQLite, PostgreSQL (JSONB), MongoDB, Parquet, JSONL | Annotation records |
| **Compute** | Local RTX 4090, Google Colab A100, Lambda Labs A100 ($1.10/hr) | Training/reconstruction |
| **Object Store** | Lab NFS, S3, GCS | Raw dataset hosting (TB scale) |

---

## 8. COMPREHENSIVE COST ANALYSIS

### 8.1 Compute Costs

| Resource | Spec | Upfront | Hourly | Best For |
|----------|------|---------|--------|----------|
| **Workstation GPU** | RTX 4090 24GB | $1,600 | — | Permanent lab setup |
| **Google Colab** | A100 40GB | — | ~$10/day | PoC, short experiments |
| **Lambda Labs** | A100 40GB | — | $1.10/hr | Scale processing |
| **CPU Server** | 16+ cores | Lab existing | — | Dataset pairing, VLM batch prep |

### 8.2 VLM API Costs (Per 500 Scenes)

| Strategy | Model Mix | Input Cost | Output Cost | Tokens/Scene | Total |
|----------|-----------|-----------|-------------|--------------|-------|
| **Premium Only** | Gemini 2.5 Pro standard | $1.25/MTok | $10/MTok | ~2K in / ~500 out | ~$3,750 |
| **Batch Only (Pro)** | Gemini 2.5 Pro Batch | $0.625/MTok | $5/MTok | ~2K in / ~500 out | ~$1,875 |
| **Flash Only** | Gemini 2.5 Flash Batch | $0.075/MTok | $1.25/MTok | ~2K in / ~500 out | ~$200 |
| **TIERED (RECOMMENDED)** | 80% Flash + 20% Pro (Batch) | Mixed | Mixed | Mixed | **~$400** |
| **GPT-4o Fallback** | GPT-4o standard | $2.50/MTok | $10/MTok | ~2K in / ~500 out | ~$7,500 |

**Breakdown of Tiered Strategy:**
- 400 simple scenes (classrooms, offices) → Flash Batch: ~$160
- 100 complex scenes (labs, anatomy) → Pro Batch: ~$240
- **Total: ~$400 for 500 scenes**

### 8.3 Dataset Costs

| Dataset | Cost | Notes |
|---------|------|-------|
| HyperSim | $0 | MIT, instant |
| ScanNet | $0 | Academic, 1–3 day approval |
| Replica | $0 | MIT, Git LFS bandwidth only |
| ARKitScenes | $0 | Apple academic |
| HM3D | $0 | Apache 2.0 |
| HSSD | $0 | CC BY-NC 4.0 |
| Objaverse-XL | $0 | Open license, HuggingFace |
| SceneSplat-7K | $0 | HuggingFace |
| AI2-THOR | $0 | Apache 2.0 |

### 8.4 Software Costs

All core software is **free and open-source**:
- DUSt3R, MASt3R, COLMAP, Nerfstudio, gsplat, SuperSplat: **$0**
- SAM2, Grounding DINO: **$0**
- UniDepth v2, Depth Anything v2: **$0**
- Open3D, PyTorch3D: **$0**

### 8.5 Total Estimated Budget (First Month)

| Approach | GPU | Cloud Compute | API | Storage | Total |
|----------|-----|---------------|-----|---------|-------|
| **A (Dataset-only)** | $0 (lab CPU) | $0 | $1,000–1,875 | $50 (external drive) | **$1,050–1,925** |
| **B (Custom-only)** | $1,600 (RTX 4090) | $50 (Colab backup) | $400 | $50 | **$2,100** |
| **C (Hybrid)** | $1,600 | $100 | $400 | $100 | **$2,200** |
| **D (API-first)** | $0 | $0 | $400 + $20/mo Polycam | $50 | **$470** |
| **E (Synthetic)** | $0 | $0 | $0 | $50 | **$50** |

---

## 9. COMPLEXITY & RISK MATRIX

### 9.1 Risk Register (All Approaches)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| DUSt3R fails on textureless scenes (whiteboards, bare walls) | Medium | High | Fallback to COLMAP + ArUco markers; filter scenes with <30% textured surfaces |
| UniDepth v2 scale ambiguity on custom captures | Medium | High | Use DUSt3R point cloud median for global scale; validate against known object sizes |
| ScanNet approval delayed/rejected | Low | Medium | HyperSim provides immediate alternative; 77K images sufficient for PoC |
| Gemini API rate limits at scale | Medium | Medium | Use Batch API; exponential backoff; shard across multiple projects |
| SAM2 misses small educational objects (test tubes, beakers) | Medium | High | Use 2× upsampling for small objects; refine category list |
| VLM spatial hallucination despite metadata | Low | High | Quality gate: ±20% distance tolerance; discard non-compliant outputs |
| GPU memory insufficient for large scenes | Medium | Medium | Use gsplat backend (4× memory reduction); process in overlapping chunks |
| Domain gap: HyperSim synthetic → real classrooms | High | Medium | Fine-tune UniDepth v2 on ScanNet real depth; domain randomization |
| COLMAP fails on low-texture custom video | Medium | High | Ensure >70% overlap; add textured markers; use DUSt3R primary |
| Luma AI API discontinuation/shutdown | High | High | **Do NOT depend on Luma; use open-source stack** |
| API cost overrun | Medium | Medium | Tiered Flash/Pro routing; batch processing; start with 100-scene validation |
| Data storage overflow (>5TB) | Medium | Medium | Store only keyframes; compress depth maps; use SPZ for splats |

### 9.2 Complexity Comparison by Layer

| Layer | Approach A | Approach B | Approach C | Approach D | Approach E |
|-------|------------|------------|------------|------------|------------|
| **Ingestion** | LOW | MEDIUM | MEDIUM | LOW | MEDIUM |
| **Reconstruction** | NONE | HIGH | MEDIUM | NONE | LOW |
| **Depth Estimation** | NONE | HIGH | MEDIUM | NONE | NONE |
| **Segmentation** | LOW | HIGH | MEDIUM | LOW | NONE |
| **2D-3D Pairing** | MEDIUM | HIGH | MEDIUM | MEDIUM | LOW |
| **VLM Annotation** | MEDIUM | MEDIUM | MEDIUM | MEDIUM | NONE |
| **Quality Control** | LOW | HIGH | MEDIUM | LOW | LOW |
| **Overall** | LOW | HIGH | MEDIUM | LOW | LOW |

---

## 10. IMPLEMENTATION ROADMAP OPTIONS

### 10.1 Fast Track (Mentor Demo in 1 Week)
**Best for:** Securing approval, proving feasibility
1. **Day 1:** Download HyperSim (instant)
2. **Day 2:** Extract 100 frames + poses from 1 scene
3. **Day 3:** Build pairing script (project 3D labels to 2D)
4. **Day 4:** Annotate 10 frames via GPT-4o or Gemini (free tier)
5. **Day 5:** Validate JSON schema + create visualization
6. **Day 6:** Prepare 5-slide summary
7. **Day 7:** **Mentor presentation** with live demo

**Cost:** ~$0–10 (API free tier)  
**Deliverable:** 100 annotated frames from existing dataset

### 10.2 Standard Track (PoC in 1 Month)
**Best for:** Solid foundation, first training data
- **Week 1:** Environment + HyperSim validation
- **Week 2:** DUSt3R on custom video + ScanNet processing
- **Week 3:** VLM integration + cost optimization
- **Week 4:** Uni-RAG training corpus (10K triplets)

**Cost:** ~$400 (tiered VLM) + GPU time  
**Deliverable:** 10,000 query-scene-answer triplets across 50 scenes

### 10.3 Full Research Track (3 Months)
**Best for:** Publication-quality dataset, comprehensive evaluation
- **Month 1:** All datasets ingested, reconstruction pipeline solid
- **Month 2:** Scale to 500 scenes, validate against ARKitScenes LiDAR
- **Month 3:** Full Uni-RAG training, ablation studies, paper draft

**Cost:** ~$2,000–2,500 total  
**Deliverable:** 100K+ annotated pairs, benchmark results, paper

---

## 11. MASTER COMPARISON TABLE

| Dimension | Approach A:<br>Dataset-First | Approach B:<br>Custom Recon | Approach C:<br>Hybrid<br>(RECOMMENDED) | Approach D:<br>API-First | Approach E:<br>Synthetic |
|-----------|:---------------------------:|:---------------------------:|:--------------------------------------:|:------------------------:|:------------------------:|
| **Start Time** | Day 1 | Day 3–5 | Day 1 | Day 1 | Day 2 |
| **Initial Cost** | $0 | $1,600 (GPU) | $1,600 (GPU) | $20–50/mo | $0 |
| **Per-Scene Cost** | ~$2–4 (VLM only) | ~$0.80 (compute) + $0.80 (VLM) | ~$0.80 (compute) + $0.80 (VLM) | ~$0.50 (API) | $0 |
| **Data Volume** | 2.5M+ frames | Limited by capture | Unlimited (hybrid) | Limited by API quota | Unlimited |
| **Domain Control** | ❌ Fixed | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| **Ground Truth Quality** | ★★★★★ Perfect | ★★★☆☆ Estimated | ★★★★☆ Mixed | ★★★☆☆ Cloud-dependent | ★★★★★ Perfect |
| **Real-World Realism** | ★★★★★ Real | ★★★★★ Real | ★★★★★ Real | ★★★★☆ Good | ★★☆☆☆ Synthetic |
| **Educational Specificity** | ★★☆☆☆ Generic | ★★★★★ Custom | ★★★★★ Custom | ★★★★★ Custom | ★★★★★ Custom |
| **Reconstruction Complexity** | NONE | HIGH | MEDIUM | NONE | LOW |
| **Annotation Reliability** | HIGH (GT-backed) | MEDIUM (metadata-first) | HIGH (hybrid validation) | MEDIUM | HIGHEST (no VLM needed) |
| **Scalability** | ★★★★★ Instant | ★★☆☆☆ Slow per scene | ★★★★☆ Good | ★★★☆☆ API-limited | ★★★★★ Unlimited |
| **Mentor Impressiveness** | ★★★☆☆ Standard | ★★★★★ Novel | ★★★★★ Comprehensive | ★★★☆☆ Quick but shallow | ★★★☆☆ Synthetic concern |
| **Best For** | Immediate volume | Domain-specific scenes | **Balanced research** | Quick demo | Ablation / augmentation |
| **Risk Level** | LOW | HIGH | MEDIUM | MEDIUM | LOW |
| **Uni-RAG Ready** | ✅ Yes | ✅ Yes | ✅✅ Best | ⚠️ Partial | ⚠️ Domain gap |

---

## 12. FINAL RECOMMENDATION & NEXT STEPS

### 12.1 Recommended Path: Approach C (Hybrid Synthesis)

After analyzing all 5 source documents, **Approach C is definitively recommended** for the following reasons:

1. **Eliminates single points of failure:** If ScanNet is delayed, HyperSim provides Day 1 data. If DUSt3R fails on a textureless wall, COLMAP + ArUco markers serve as fallback.
2. **Optimizes cost:** The tiered VLM strategy (80% Flash / 20% Pro) reduces annotation costs by 5× compared to premium-only.
3. **Validates quality:** ARKitScenes LiDAR ground truth allows quantification of no-LiDAR pipeline error (target RMSE < 15cm).
4. **Scales across domains:** Existing datasets provide generic indoor diversity; custom capture adds educational specificity; Objaverse-XL adds object-level detail.
5. **Prevents hallucination:** The metadata-first VLM strategy (geometry from math, semantics from VLM) is validated by the Real-3DQA benchmark showing 3D-LLMs collapse on multi-view rotation tasks.

### 12.2 Key Differentiators to Highlight to Mentor

1. **DUSt3R over COLMAP as primary** — eliminates calibration barrier (core constraint)
2. **UniDepth v2 for metric depth** — first proposal to solve "metric scale without sensors"
3. **Metadata-first VLM** — eliminates spatial hallucination risk
4. **BEV as 4th input modality** — global context augmentation
5. **HyperSim-first dataset strategy** — Day 1 validation without approval delays
6. **Corrected cost analysis** — realistic $400 for 500 scenes (not $0.50)
7. **Validation against ARKitScenes** — LiDAR ground truth quantifies error
8. **SPZ compression** — 90% size reduction for web deployment
9. **Tiered VLM routing** — Flash for simple, Pro for complex
10. **Explicit quality gates** — CLIP, depth coverage, pose confidence, VLM consistency

### 12.3 Immediate Action Items (Today)

```bash
# 1. Install core stack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install nerfstudio
pip install git+https://github.com/naver/dust3r.git
pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git
pip install transformers accelerate

# 2. Download HyperSim (instant, MIT license)
git clone https://github.com/apple/ml-hypersim.git
python ml-hypersim/code/python/tools/dataset_download.py   --scene_names ai_001_001 ai_001_002 ai_001_003   --downloads_dir /data/hypersim

# 3. Request ScanNet (start approval process)
# Visit: http://www.scan-net.org/ → "Get Access"

# 4. Record test video (2-minute slow orbit of lab/office)
# Extract frames: ffmpeg -i video.mp4 -vf fps=3 frames/%06d.jpg

# 5. Set up Gemini API key for batch annotation testing
# https://aistudio.google.com/app/apikey
```

### 12.4 Expected Deliverables for Mentor Review

| # | Deliverable | Timeline | Metric |
|---|-------------|----------|--------|
| 1 | Pipeline Demo Notebook | Week 1 | Video → 3D splat → JSON in <15 min |
| 2 | Reconstruction Accuracy Report | Week 2 | DUSt3R vs. ARKitScenes LiDAR: RMSE < 15cm |
| 3 | Annotated Dataset v1 | Week 3 | 1,000 frames across 20 scenes, 3 domains |
| 4 | Cost Analysis Report | Week 3 | Validated $/scene with tiered VLM strategy |
| 5 | Uni-RAG Training Corpus | Week 4 | 10,000 query-scene-answer triplets with 3D coordinates |
| 6 | Qualitative Results Gallery | Week 4 | Side-by-side: reconstruction, BEV, annotated frames |

---

*This document synthesizes 5 independent research proposals into a single decision framework. Every tool, dataset, cost estimate, and architectural decision has been preserved for comprehensive mentor discussion.*
