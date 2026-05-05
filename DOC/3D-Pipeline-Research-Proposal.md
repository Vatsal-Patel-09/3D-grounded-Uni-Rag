# Formal Research Proposal
## 2D-to-3D Scene Data Generation & Annotation Pipeline for 3D-Grounded Uni-RAG (AIED)

**Prepared by:** AI Architect Swarm (3 agents)
**Date:** 2026-04-25
**Target:** Lab Mentor Presentation

---

## 1. Executive Summary

**Chosen Hybrid Approach: Datasets-First + Software Reconstruction for Custom Content**

Do not attempt to solve everything from scratch. The optimal path uses two parallel tracks:

| Track | When | Output |
|---|---|---|
| **Track A — Leverage Existing** | Day 1–30 | ScanNet/HM3D/HyperSim scenes pre-labeled with 3D geometry |
| **Track B — Custom Reconstruction** | Day 31–90 | DUSt3R + 3D Gaussian Splatting on smartphone video of real educational spaces |

Track A gives 100K+ annotated 2D-3D pairs immediately. Track B adds domain-specific scenes (actual classrooms, lab benches, anatomy models). Together they cover the diversity needed to train a robust retrieval model.

**Core Insight:** The LiDAR gap is solved by **DUSt3R** (CVPR 2024, Apache 2.0) — it performs dense 3D reconstruction from arbitrary, uncalibrated image collections with no special hardware. This is the critical enabling technology.

---

## 2. Agent 1 Findings — Datasets & Tools

### 2A. Existing 3D Indoor Datasets (Academic License, No LiDAR Required from You)

| Dataset | Scenes | 2D Frames | 3D Format | License | AIED Relevance |
|---|---|---|---|---|---|
| **ScanNet v2** | 1,513 scans | 2.5M RGB-D frames | Mesh + point cloud + poses | Free academic | ★★★★★ Indoor rooms, classroom-scale |
| **HM3D (Habitat-Matterport)** | 1,000 buildings | ~5M views | Semantic mesh | Academic | ★★★★☆ Large-scale navigation |
| **Replica** | 18 scenes | Synthetic | Mesh + HDR | MIT | ★★★☆☆ High fidelity, limited scale |
| **HyperSim** | 461 rooms | 77K images | Dense depth + normals + labels | MIT | ★★★★☆ Photorealistic synthetic, rich labels |
| **3RScan** | 1,482 scans | 2.4M RGB | Point cloud + graph | Academic | ★★★☆☆ Object-change detection |
| **ARKitScenes** | 5,047 scenes | >200M frames | LiDAR mesh (pre-captured) | Apple Academic | ★★★★☆ High quality indoor |

**Recommended download order:**
1. HyperSim first — MIT license, Python-friendly API, synthetic so no ethical issues
2. ScanNet — request at `scannet.org`, approval in 1–3 days
3. Replica — instant via `git lfs` from Facebook Research GitHub

### 2B. Software Reconstruction Stack (No LiDAR Hardware)

**Tier 1 — Core Pipeline:**

```
Input Images/Video
       │
       ▼
  DUSt3R / MASt3R          ← Dense point map, camera poses, NO calibration needed
  (github.com/naver/dust3r)
       │
       ▼
  3D Gaussian Splatting     ← Radiance field from DUSt3R poses + images
  (github.com/graphdeco-inria/gaussian-splatting)
       │
       ▼
  Nerfstudio Export         ← Clean PLY point cloud + mesh export
  (nerf.studio)
```

**Tier 2 — Depth Augmentation (when video is available but reconstruction is slow):**

- **Depth Anything v2** (`github.com/DepthAnything/Depth-Anything-V2`) — SOTA monocular depth, MIT license. Run on every frame → metric-scale depth map. Compute: single A100 processes ~1,500 frames/hour.
- **UniDepth v2** — metric depth from single images with camera intrinsic prediction. Alternative when focal length is unknown.

**Tier 3 — Managed APIs (for rapid prototyping, budget permitting):**

- **Luma AI** (`lumalabs.ai/luma-api`) — 500 free NeRF splats/month. Upload iPhone video, get Gaussian splat + point cloud back via REST API. Best for quick wins.
- **Polycam** — Similar, $20/month unlimited. Outputs GLTF/OBJ/PLY.

**Compute Budget:**
- DUSt3R + 3DGS on 100-image scene: ~45 min on RTX 3090 (Google Colab A100 ~20 min, ~$2)
- Depth Anything v2 on 1K frames: ~15 min on A100

---

## 3. Agent 2 Findings — Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                                                   │
│                                                                                     │
│  [Track A] Existing Datasets          [Track B] Custom Capture                     │
│  HyperSim / ScanNet / Replica         Smartphone video of real classrooms           │
│       │                                      │                                      │
│       ▼                                      ▼                                      │
│  dataset_loader.py                    FFmpeg frame extraction                       │
│  (standardize to common format)       fps=3, resolution=1280x720                    │
└──────────────────────┬──────────────────────┬──────────────────────────────────────┘
                       │                      │
                       ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                                                  │
│                                                                                     │
│  Step 1: Camera Pose Estimation                                                     │
│  ┌─────────────────────────────────────────────────────────┐                       │
│  │  Track A: Poses already in dataset (load .npz / .json)  │                       │
│  │  Track B: Run DUSt3R on frame batch → poses + pointmap  │                       │
│  └─────────────────────────────────────────────────────────┘                       │
│                                                                                     │
│  Step 2: Depth Map Generation                                                       │
│  ┌─────────────────────────────────────────────────────────┐                       │
│  │  Track A: Load ground-truth depth (aligned RGB-D)        │                       │
│  │  Track B: Depth Anything v2 → per-frame depth .png       │                       │
│  │           Scale to metric using DUSt3R point cloud avg   │                       │
│  └─────────────────────────────────────────────────────────┘                       │
│                                                                                     │
│  Step 3: Scene Graph Extraction                                                     │
│  ┌─────────────────────────────────────────────────────────┐                       │
│  │  Run Grounded-SAM2 on each keyframe → instance masks    │                       │
│  │  Unproject masks using depth + camera pose → 3D centroids│                      │
│  │  Build per-frame object list: {label, centroid_3d, bbox_3d, distance}           │
│  └─────────────────────────────────────────────────────────┘                       │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    2D-3D PAIRING LAYER  (Critical)                                  │
│                                                                                     │
│  For each frame i with camera pose [R_i | t_i] and intrinsics K:                   │
│                                                                                     │
│  1. Project global 3D points into frame i:                                         │
│     p_2d = K @ (R_i @ p_3d + t_i)                                                  │
│     → Filter: z > 0, p_2d inside image bounds                                      │
│                                                                                     │
│  2. Compute per-object metric distances:                                            │
│     dist = ||p_centroid_3d - camera_position||₂  (meters)                          │
│                                                                                     │
│  3. Extract Bird's Eye View (BEV):                                                  │
│     Project XZ plane → 2D BEV map → encode spatial layout                          │
│     (LEFT-OF, BEHIND, IN-FRONT, ADJACENT-TO with metric offsets)                  │
│                                                                                     │
│  4. Store pairing record (see JSON schema in Section 5)                            │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    ANNOTATION LAYER (VLM)                                           │
│  See Agent 3 output (Section 4)                                                     │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    QUALITY FILTER + INDEX                                           │
│  - CLIP score ≥ 0.25 (discard blurry/dark frames)                                  │
│  - Depth coverage ≥ 70% (discard frames where depth fails)                          │
│  - Pose confidence filter (DUSt3R confidence map threshold)                         │
│  - Deduplicate: cosine similarity < 0.95 between adjacent frames                   │
│  → Write to SQLite index + HDF5 data store                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Key implementation files:**

| File | Purpose |
|---|---|
| `ingest/dataset_loader.py` | Unified reader for ScanNet/HyperSim/Replica formats |
| `ingest/video_extractor.py` | FFmpeg wrapper, keyframe selection |
| `process/dust3r_runner.py` | Batch DUSt3R reconstruction |
| `process/depth_estimator.py` | Depth Anything v2 inference |
| `process/gsam_segmenter.py` | Grounded-SAM2 instance extraction |
| `pair/projector.py` | 3D→2D projection, BEV computation, metric distance |
| `annotate/vlm_annotator.py` | Gemini/GPT-4o batch annotation |
| `store/hdf5_writer.py` | Data serialization |

---

## 4. Agent 3 Findings — Annotation Strategy

### 4A. VLM Input Package (per frame)

Feed Gemini 1.5 Pro **three inputs simultaneously** using its 2M context multimodal capability:

1. **RGB image** (the 2D frame)
2. **Colorized depth map** (jet colormap, 16-bit normalized to 0-255)
3. **Structured metadata string** (pre-computed 3D spatial facts)

The metadata string removes spatial hallucination — the VLM annotates *how*, not *what the geometry is*. Geometry comes from math; semantics come from the VLM.

### 4B. System Prompt

```
You are a 3D spatial scene annotator for an educational AI system.
You will receive:
  (1) An RGB image of an indoor educational environment
  (2) Its depth map (blue=near, red=far)
  (3) Pre-computed spatial metadata from 3D reconstruction

Your task is to generate spatially grounded captions that are ACCURATE to the
provided metadata. Do NOT invent distances or positions. Use the metadata as
ground truth. Convert all distances to intuitive human language
(e.g., "arm's reach" for < 0.8m, "across the room" for > 4m).

Output ONLY valid JSON matching the provided schema. No markdown. No explanation.
```

### 4C. User Prompt Template

```
SPATIAL METADATA (ground truth, treat as authoritative):
Camera height: {camera_height_m:.2f}m above floor
Camera FOV: {fov_deg:.1f}°

Detected objects (3D centroids in camera coordinate frame):
{object_list_formatted}
  → Example line: "whiteboard | centroid=(0.0, 0.2, 3.1)m | distance=3.1m | area=2.4m²"

Spatial relationships (pre-computed):
{relationship_list}
  → Example: "desk is 1.2m LEFT of chair, 0.4m CLOSER to camera"

Scene type: {scene_type}  (e.g., "university_lecture_hall", "biology_lab", "library")
Educational domain: {edu_domain}  (e.g., "anatomy", "chemistry", "general")

---
Generate the annotation JSON now.
```

### 4D. Few-Shot Example (include 1 in every API call)

```json
// Example input metadata:
// Objects: whiteboard(3.1m), desk(1.8m), chair(1.2m, left-of desk)
// Scene: biology_lab

// Expected output:
{
  "spatial_caption": "A biology lab viewed from standing height. A whiteboard dominates the far wall approximately 3 meters away. A wooden desk sits in the mid-ground at roughly 2 meters, with a chair positioned to its left at arm's reach distance.",
  "detail_caption": "The scene is a compact biology laboratory. The whiteboard spans the full width of the rear wall and appears to have written content. The desk surface is clear and unoccupied. Natural lighting enters from the left side.",
  "educational_context": {
    "subject_domain": "biology",
    "key_concepts": ["laboratory workspace", "instructional display area"],
    "spatial_relationships": [
      {"subject": "chair", "predicate": "LEFT_OF", "object": "desk", "metric": "0.6m"},
      {"subject": "desk", "predicate": "IN_FRONT_OF", "object": "whiteboard", "metric": "1.3m"}
    ]
  }
}
```

---

## 5. Annotation JSON Schema (Full)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema",
  "title": "SceneAnnotation",
  "type": "object",
  "required": ["scene_id", "frame_id", "camera_pose", "objects_3d", "spatial_caption"],
  "properties": {
    "scene_id": { "type": "string", "description": "Unique scene identifier, e.g. scan0001_00" },
    "frame_id": { "type": "string", "description": "Frame within scene, e.g. frame_000420" },
    "source_dataset": { "type": "string", "enum": ["scannet", "hypersim", "replica", "hm3d", "custom_dust3r"] },
    "timestamp_sec": { "type": "number" },

    "camera_pose": {
      "type": "object",
      "properties": {
        "position_xyz": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
        "rotation_matrix": { "type": "array", "items": { "type": "array" } },
        "height_from_floor_m": { "type": "number" },
        "intrinsics": {
          "type": "object",
          "properties": {
            "fx": { "type": "number" }, "fy": { "type": "number" },
            "cx": { "type": "number" }, "cy": { "type": "number" },
            "width_px": { "type": "integer" }, "height_px": { "type": "integer" }
          }
        }
      }
    },

    "depth_stats": {
      "type": "object",
      "properties": {
        "min_depth_m": { "type": "number" },
        "max_depth_m": { "type": "number" },
        "mean_depth_m": { "type": "number" },
        "coverage_ratio": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },

    "objects_3d": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "object_id":           { "type": "string" },
          "label":               { "type": "string" },
          "confidence":          { "type": "number" },
          "centroid_xyz":        { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3 },
          "bbox_3d_min":         { "type": "array", "items": { "type": "number" } },
          "bbox_3d_max":         { "type": "array", "items": { "type": "number" } },
          "distance_from_cam_m": { "type": "number" },
          "visible_area_m2":     { "type": "number" },
          "mask_2d_rle":         { "type": "string", "description": "RLE-encoded 2D instance mask" }
        }
      }
    },

    "spatial_relationships": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "subject_id": { "type": "string" },
          "predicate":  { "type": "string", "enum": ["LEFT_OF","RIGHT_OF","IN_FRONT_OF","BEHIND","ABOVE","BELOW","ADJACENT_TO","CONTAINS","ON_TOP_OF"] },
          "object_id":  { "type": "string" },
          "metric_m":   { "type": "number" },
          "axis":       { "type": "string", "enum": ["x","y","z","euclidean"] }
        }
      }
    },

    "spatial_caption":  { "type": "string", "description": "Primary grounded spatial description, 1-3 sentences" },
    "detail_caption":   { "type": "string", "description": "Richer visual/contextual description" },

    "educational_context": {
      "type": "object",
      "properties": {
        "subject_domain":    { "type": "string" },
        "scene_type":        { "type": "string" },
        "key_concepts":      { "type": "array", "items": { "type": "string" } },
        "retrieval_queries": { "type": "array", "items": { "type": "string" },
          "description": "5-10 natural language queries this scene should answer" }
      }
    },

    "bev_map_path":     { "type": "string", "description": "Path to Bird's Eye View PNG" },
    "rgb_path":         { "type": "string" },
    "depth_path":       { "type": "string" },
    "pointcloud_path":  { "type": "string" },

    "annotation_meta": {
      "type": "object",
      "properties": {
        "model":      { "type": "string", "enum": ["gemini-1.5-pro", "gpt-4o", "gemini-2.5-pro"] },
        "version":    { "type": "string" },
        "timestamp":  { "type": "string" },
        "confidence": { "type": "number" },
        "cost_usd":   { "type": "number" }
      }
    }
  }
}
```

---

## 6. Final Tool Stack

| Layer | Tool | License / Cost |
|---|---|---|
| Datasets | HyperSim | MIT (free) |
| Datasets | ScanNet v2 | Academic (free) |
| Datasets | Replica | MIT (free) |
| 3D Reconstruction | DUSt3R / MASt3R | Apache 2.0 (free) |
| 3D Reconstruction | 3D Gaussian Splatting | Inria license (free academic) |
| 3D Reconstruction | Nerfstudio | Apache 2.0 (free) |
| Depth | Depth Anything v2 (Large) | MIT (free) |
| Depth | UniDepth v2 | Apache 2.0 (free) |
| Segmentation | Grounded-SAM2 | Apache 2.0 (free) |
| Camera Poses | DUSt3R (custom video) | Apache 2.0 (free) |
| Camera Poses | COLMAP (fallback) | BSD (free) |
| VLM Annotation | Gemini 1.5 Pro (primary) | $0.00125/1K tokens (~$0.01/scene) |
| VLM Annotation | GPT-4o (fallback) | $0.0025/1K tokens |
| Rapid Prototyping | Luma AI API | 500 free splats/month |
| Storage | HDF5 + SQLite index | Free |
| Compute | Google Colab A100 / L4 | ~$10 for PoC batch |

---

## 7. Immediate Next Steps (Today / This Week)

### Day 1 — Environment Setup + First Data

```bash
# 1. Install core tools
pip install nerfstudio
pip install git+https://github.com/naver/dust3r.git
pip install git+https://github.com/IDEA-Research/Grounded-Segment-Anything.git

# 2. Download HyperSim (MIT, no approval wait)
# https://github.com/apple/ml-hypersim
python -m hypersim_download --scenes ai_001_001 ai_001_002 ai_001_003

# 3. Request ScanNet (takes 1-3 days)
# Visit: http://www.scan-net.org/ → click "Get Access"
```

### Day 2 — PoC Reconstruction (Track B)

```bash
# Record 2-minute smartphone video of a classroom/lab
# Extract frames
ffmpeg -i classroom.mp4 -vf fps=3 frames/%06d.jpg

# Run DUSt3R
python dust3r/demo.py \
  --input frames/ \
  --output recon/ \
  --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt

# Visualize in Nerfstudio
ns-train gaussian-splatting --data recon/
```

### Day 3 — First Annotation Run

```python
# Quick PoC: annotate 10 HyperSim frames via Gemini API
import google.generativeai as genai
genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel("gemini-1.5-pro")

# Load frame + depth + metadata → send multimodal request → save JSON
# Target: validate JSON schema with real output before scaling
```

### Day 4–7 — Integration + Scale

- Wire `ingest/ → process/ → pair/ → annotate/` into single `python pipeline.py --scene ai_001_001`
- Run on 10 HyperSim scenes → produce ~500 annotated 2D-3D pairs
- Validate schema integrity with JSON Schema validator (`jsonschema` Python library)
- Compute annotation cost: target < $0.50 for 500 scenes

### First 30 Days — Scale Target

| Metric | Target |
|---|---|
| Annotated 2D-3D pairs | 10,000 |
| Unique scenes | 50 |
| Domains covered | 4 (general indoor, biology lab, anatomy, library) |
| Annotation cost | < $15 total |
| Pipeline runtime | < 4 hrs/100 scenes on A100 |

---

## 8. Risk Register

| Risk | Mitigation |
|---|---|
| ScanNet approval delayed | Start with HyperSim (MIT, instant) |
| DUSt3R pose quality poor for custom video | Fallback to COLMAP + OpenMVS; use Luma AI API |
| Gemini rate limits at scale | Batch with exponential backoff; use Vertex AI batch prediction API (cheaper + higher quota) |
| Depth scale ambiguity for Track B | Use DUSt3R confidence maps to scale Depth Anything output; discard frames with coverage < 70% |
| VLM spatial hallucination | Ground truth geometry always from math, not VLM; VLM only writes prose from pre-computed facts |

---

*The pipeline is fully achievable without LiDAR. DUSt3R + existing datasets + Gemini batch annotation removes all hardware blockers. PoC executable within one week on a single GPU.*