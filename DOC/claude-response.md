# Formal Research Proposal: 2D-to-3D Scene Data Generation & Annotation Pipeline for 3D-Grounded Uni-RAG

---

## Swarm Execution Log

### Agent 1 — 3D Computer Vision Researcher

**Finding 1: Dataset Audit**

After surveying the landscape, three tiers of datasets emerge based on richness, licensing, and relevance to AIED:

**Tier 1 — Primary Candidates (use immediately)**
- **ScanNet v2** (`http://www.scan-net.org`) — 1,513 RGB-D scans of indoor scenes (classrooms, offices, labs). Provides aligned 2D frames, depth maps, 3D mesh, semantic + instance labels, and camera poses. CC BY 4.0 for academic use. This is your strongest starting point.
- **Matterport3D** (`https://niessner.github.io/Matterport`) — 90 large building-scale environments, 194,400 RGB-D images, skybox panoramas. Provides surface normals, depth, region and object labels, and mesh. Free for academic use (license agreement required). Best for room-scale spatial reasoning.
- **HM3D (Habitat-Matterport 3D)** (`https://aihabitat.org/datasets/hm3d`) — 1,000 photorealistic 3D environments, specifically designed for navigation and simulation tasks. Compatible with Meta's Habitat simulator for synthetic view generation. Apache 2.0.

**Tier 2 — Supplementary / Domain-Specific**
- **Hypersim** (`https://github.com/apple/ml-hypersim`) — 461 synthetic indoor scenes from Autodesk 3ds Max with physically-based rendering. Every pixel has ground-truth depth, normals, and semantic labels. No hardware required. MIT License. Especially strong for educational interior environments.
- **AI2-THOR** (`https://ai2thor.allenai.org`) — A procedural 3D environment simulator with 120+ room types. Generate unlimited 2D frames from arbitrary viewpoints programmatically. Apache 2.0. Strong choice for synthetic data augmentation.
- **RealEstate10K** (`https://google.github.io/realestate10k`) — 10 million frames from YouTube real estate videos with camera poses. For monocular video reconstruction pipelines.

**Finding 2: Software-Based Reconstruction Tools (No LiDAR)**

| Tool | Method | Input | Quality | GPU Req. | Cost |
|------|---------|-------|---------|----------|------|
| **Nerfstudio** | NeRF variants (Nerfacto, Instant-NGP) | 30–200 images | High novel view synthesis | 1× A100 (24GB) | Free/OSS |
| **3D Gaussian Splatting (3DGS)** | Gaussian primitives | 50–300 images | Excellent real-time rendering | 1× RTX 3090 | Free/OSS |
| **Luma AI** | Neural Radiance Field | Short video (2–5 min) | Very high | Cloud (free tier) | Free/Paid |
| **Polycam** | Photogrammetry + NeRF | 20–200 images | High | Cloud | Free/Paid |
| **COLMAP** | Structure-from-Motion | Any image set | Sparse/dense PC | CPU/GPU | Free/OSS |
| **OpenMVS** | Multi-View Stereo | COLMAP output | Dense mesh | CPU/GPU | Free/OSS |
| **MASt3R / DUSt3R** | Feed-forward 3D | Image pairs | High | 1× A40 | Free/OSS |

**Recommendation from Agent 1:** The optimal no-LiDAR strategy is a **hybrid pipeline** — use ScanNet/Matterport3D for ground-truth 3D pairs immediately, use Hypersim for domain-controlled synthetic data, and add Nerfstudio + COLMAP for any custom environments you wish to reconstruct from your own footage.

---

### Agent 2 — Data Pipeline Engineer

**Full pipeline architecture designed — see diagram below.**

**Ingestion Layer:**
- ScanNet `.sens` files → decoded via `SensReader.py` (official SDK) to extract RGB frames + depth maps + camera intrinsics/extrinsics at every keyframe.
- Matterport3D `.zip` files → structured JSON manifests providing image-to-region-to-mesh mappings out of the box.
- Custom video → COLMAP SfM → Nerfstudio for novel view synthesis.

**Processing Layer:**
- Frame extraction at 1 fps (or keyframe-based) using ffmpeg / ScanNet SDK.
- COLMAP-computed camera poses bundled per scene into a `transforms.json` (Nerfstudio native format).
- 3D mesh stored as `.ply` or `.obj` with UV texture maps.

**Pairing Layer (the critical mathematical link):**
- Each 2D frame `i` is associated with a `4×4 camera-to-world matrix` **T_i** from COLMAP/ScanNet poses.
- For each annotated 3D object bounding box `(x, y, z, w, h, d)` in the scene, project its 8 corners through `K × T_i^{-1}` (where `K` is the camera intrinsic matrix) to get the 2D pixel bounding box in frame `i`.
- Store the round-trip: `frame_id → {camera_pose, projected_2d_boxes, 3d_world_coords, depth_map_path}`.

---

### Agent 3 — Multimodal Prompt Engineer

**VLM Annotation Strategy:**

The annotation call uses a two-stage approach:

**Stage 1 (Geometry-Aware Context Injection):** Before the VLM sees the image, pre-compute spatial metadata from the 3D scene (object centroids, pairwise distances, height above floor, surface normals). Inject this as a structured preamble into the prompt.

**Stage 2 (Grounded Caption Generation):** The VLM is instructed to use the spatial metadata to anchor every spatial claim in its caption, producing metric-grounded language suitable for RAG retrieval.

---

Now let me render the full architecture diagram and then the complete formal proposal document.---

## Formal Research Proposal

**Title:** A Hybrid 2D-to-3D Scene Data Generation and Annotation Pipeline for 3D-Grounded Uni-RAG in AI Education

**Submitted by:** [Your Name], [Lab Name], [University]

---

### 1. Executive Summary

The proposed approach is a **three-track hybrid strategy** that eliminates the LiDAR hardware dependency by combining (1) existing high-quality 3D datasets with verified camera-scene correspondences, (2) software-only 3D reconstruction from video using Nerfstudio and 3D Gaussian Splatting, and (3) synthetic procedural generation via AI2-THOR and Hypersim. These three tracks feed a unified pairing layer that links 2D viewpoints to metric 3D coordinates via camera projection matrices. A final VLM annotation pass using Gemini 1.5 Pro (with spatial metadata injection) generates the grounded, distance-anchored captions required to train the Uni-RAG retrieval model.

The use of ScanNet v2 as the primary source means your first training batch can be constructed **without writing any reconstruction code at all** — the 2D-3D pairs are pre-built and academically licensed. Reconstruction pipelines are then introduced incrementally to generate domain-specific educational content.

---

### 2. Tool Stack

**Datasets**

| Dataset | Link | License | Use in Pipeline |
|---|---|---|---|
| ScanNet v2 | `scan-net.org` | CC BY 4.0 | Primary RGB-D + 3D mesh pairs |
| Matterport3D | `niessner.github.io/Matterport` | Academic agreement | Room-scale spatial reasoning |
| HM3D | `aihabitat.org/datasets/hm3d` | Apache 2.0 | Habitat simulator integration |
| Hypersim | `github.com/apple/ml-hypersim` | MIT | Synthetic ground-truth depth |
| AI2-THOR | `ai2thor.allenai.org` | Apache 2.0 | Procedural view generation |

**Reconstruction & Processing**

- `COLMAP` (v3.9+) — Structure-from-Motion, camera pose recovery
- `Nerfstudio` (v1.x) — Nerfacto / Instant-NGP for novel view synthesis
- `3D Gaussian Splatting` (official repo: `github.com/graphdeco-inria/gaussian-splatting`) — real-time renderable 3D scene representation
- `OpenMVS` — dense mesh reconstruction from COLMAP sparse point cloud
- `MASt3R / DUSt3R` (`github.com/naver/mast3r`) — feed-forward 3D reconstruction from as few as 2 images; excellent for quick custom captures

**VLM Annotation**

- Gemini 1.5 Pro via Google AI Studio API (`aistudio.google.com`)
- GPT-4o via OpenAI API (fallback / cross-validation)

**Orchestration**

- Python 3.11, PyTorch 2.x
- `open3d` (point cloud and mesh manipulation)
- `pycolmap` (Python bindings for COLMAP)
- `nerfstudio` CLI
- Apache Airflow (or simpler: Prefect) for pipeline scheduling

---

### 3. Pipeline Architecture

The full data flow proceeds in five stages (illustrated in the diagram above):

**Stage 1 — Ingestion.** ScanNet `.sens` files are decoded using the official `SensReader.py` to extract RGB frames, depth maps, and camera intrinsics/extrinsics. For Matterport3D, the provided `download_mp.py` script fetches structured region manifests. Custom lab footage is ingested as raw `.mp4` files. All sources normalize to a shared intermediate format: a directory of `.jpg` frames plus a `transforms.json` following the Nerfstudio/NeRF convention.

**Stage 2 — Processing.** FFmpeg extracts frames at 1 fps for video sources (or uses ScanNet's pre-defined keyframes). COLMAP's `automatic_reconstructor` computes sparse point clouds and recovers `4×4` camera-to-world poses for each frame. For dataset sources (ScanNet, Matterport3D), poses are read directly from their bundled metadata — no reconstruction needed.

**Stage 3 — 3D Reconstruction.** Three paths run in parallel based on source track: (a) ScanNet/Matterport3D scenes use pre-built `.ply` meshes directly; (b) custom captured footage is passed through Nerfstudio (`ns-train nerfacto`) and rendered to produce a 3D mesh via `ns-export poisson`; (c) Hypersim/AI2-THOR synthetic scenes export mesh and depth ground truth natively. For very sparse captures (2–10 images), MASt3R provides a feed-forward alternative that requires no iterative training.

**Stage 4 — 2D–3D Correspondence Pairing.** This is the mathematical core of the pipeline. For each frame `i` with camera intrinsic matrix `K` and pose `T_i`:

```python
# Project 3D world point P_world into 2D pixel coordinate
P_cam = T_i_inv @ P_world_homogeneous     # world → camera
P_img = K @ P_cam[:3]                     # camera → image plane
px, py = P_img[0]/P_img[2], P_img[1]/P_img[2]  # normalize
```

Every 3D annotated object bounding box has its 8 corners projected through this transform to produce a 2D bounding box in the corresponding frame. The resulting pair record is stored as a JSON object (schema in Section 4). This stage also computes inter-object Euclidean distances in world coordinates and each object's height above the floor plane — both values are injected into the VLM annotation prompt.

**Stage 5 — VLM Annotation.** Each 2D frame is sent to Gemini 1.5 Pro with a two-part payload: the image itself plus a structured spatial preamble derived from Stage 4 metadata. The model returns a spatially grounded caption. Full prompt template and output schema follow in Section 4.

---

### 4. Annotation Strategy

#### VLM Prompt Template (Gemini 1.5 Pro / GPT-4o)

```
SYSTEM:
You are a spatial scene annotator for an AI education system. You will be given:
  1. A 2D image of an indoor scene.
  2. Structured spatial metadata extracted from the corresponding 3D reconstruction.

Your task is to generate a highly detailed, spatially grounded scene description
suitable for 3D-aware retrieval. Every spatial claim MUST use the metric distances
provided in the metadata. Do not invent distances.

Output ONLY valid JSON matching the schema below. No prose outside the JSON object.

SPATIAL METADATA (pre-computed from 3D scene):
- Scene type: {scene_type}
- Camera height above floor: {cam_height_m:.2f} m
- Detected objects and world positions:
{object_list_formatted}
  Example row: "object_id": "obj_003", "label": "whiteboard",
               "centroid_xyz": [1.2, 0.0, 3.4], "height_m": 1.8,
               "distance_from_camera_m": 3.6

- Pairwise object distances (nearest 5 pairs):
{pairwise_distances_formatted}
  Example: "desk → chair: 0.82 m", "whiteboard → projector: 1.1 m"

USER:
[IMAGE ATTACHED]

Analyze this scene and generate the annotation JSON.
```

#### JSON Schema

```json
{
  "scene_id": "scannet_scene0050_00_frame_00423",
  "source_dataset": "scannet_v2",
  "camera_pose": {
    "position_xyz": [0.0, 1.2, 0.0],
    "rotation_quaternion": [0.0, 0.0, 0.0, 1.0],
    "height_above_floor_m": 1.2
  },
  "scene_type": "classroom",
  "global_caption": "A classroom viewed from a standing position near the back wall. A whiteboard spanning approximately 3 meters in width is mounted 1.1 meters above the floor on the far wall, 4.2 meters from the camera. A wooden desk occupies the foreground at roughly 1.0 meter distance, with a laptop open on its surface.",
  "objects": [
    {
      "object_id": "obj_007",
      "label": "whiteboard",
      "confidence": 0.97,
      "bbox_2d": {"x1": 102, "y1": 88, "x2": 534, "y2": 310},
      "centroid_3d_world": [0.2, 0.9, 4.2],
      "dimensions_m": {"width": 3.1, "height": 1.8, "depth": 0.05},
      "distance_from_camera_m": 4.2,
      "height_above_floor_m": 1.1,
      "spatial_caption": "A large whiteboard mounted on the far wall, approximately 4.2 meters from the camera and 1.1 meters above the floor, spanning the full width of the visible wall surface.",
      "relative_positions": [
        {"reference_object": "obj_003", "label": "desk", "relation": "in front of", "distance_m": 3.4},
        {"reference_object": "obj_011", "label": "projector", "relation": "below", "distance_m": 1.1}
      ]
    }
  ],
  "spatial_relationships": [
    "The teacher's desk is positioned 1.0 meter directly in front of the camera.",
    "Three student desks are arranged in a row 2.5 meters from the whiteboard.",
    "The projector is mounted 0.3 meters above and 1.1 meters to the right of the whiteboard center."
  ],
  "retrieval_tags": ["classroom", "whiteboard", "desk", "educational", "indoor", "frontal_view"],
  "reconstruction_method": "scannet_rgbd_native",
  "annotation_model": "gemini-1.5-pro",
  "annotation_timestamp": "2025-07-14T09:32:11Z"
}
```

---

### 5. Immediate Next Steps — Proof of Concept

Execute these in order over the next 3–5 days:

**Day 1 — Data acquisition**

```bash
# 1. Request ScanNet access at scan-net.org (approval is ~24hrs for academics)
# 2. Once approved, download 5 small scenes for PoC:
python download-scannet.py -o ./data/scannet --id scene0050_00
python download-scannet.py -o ./data/scannet --id scene0084_00

# 3. Clone the SensReader decoder
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet/SensReader/python && pip install -r requirements.txt
```

**Day 2 — Frame extraction and pose parsing**

```bash
# Extract RGB frames, depth maps, and camera poses from a .sens file
python reader.py --filename scene0050_00.sens \
                 --output_path ./data/scene0050_00 \
                 --export_color_images \
                 --export_depth_images \
                 --export_poses \
                 --export_intrinsics
```

**Day 3 — 2D–3D pairing script**

Write `pair_frames.py` that:
- Loads `intrinsic_color.txt` → `K` matrix
- Loads per-frame `pose/*.txt` → list of `T_i` matrices
- Loads `scene0050_00_vh_clean_2.labels.ply` (ScanNet semantic mesh) via `open3d`
- Projects labeled 3D bounding boxes into each frame
- Writes `pairs.jsonl` (one record per frame matching the schema above, minus the VLM caption)

```bash
pip install open3d pycolmap numpy tqdm
python pair_frames.py --scene_dir ./data/scene0050_00 \
                      --output ./pairs/scene0050_00_pairs.jsonl
```

**Day 4 — VLM annotation loop**

```python
# annotate.py — batch annotate frames via Gemini API
import google.generativeai as genai
import json, base64, pathlib

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-pro")

for record in load_jsonl("pairs/scene0050_00_pairs.jsonl"):
    img_bytes = pathlib.Path(record["frame_path"]).read_bytes()
    spatial_preamble = format_spatial_metadata(record)  # format dict → string
    
    response = model.generate_content([
        spatial_preamble,
        {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}
    ])
    record["global_caption"] = parse_vlm_json(response.text)
    append_jsonl("annotated/scene0050_00_annotated.jsonl", record)
```

**Day 5 — Validation and mentor demo**

- Spot-check 20 random records: verify that stated distances in captions match the numeric metadata
- Compute CLIP embedding cosine similarity between caption and frame as a proxy for annotation quality (target > 0.28)
- Prepare a 5-slide summary: dataset volume, sample annotated frames with projected bounding boxes, one full JSON record, quality metric score

---

### Appendix: Scalability Notes

Once the PoC validates on 5 ScanNet scenes, scale by: (a) downloading all 1,513 ScanNet scenes (~2TB, processable on a university HPC cluster), (b) integrating AI2-THOR to generate unlimited synthetic classroom views via its Python API (`ai2thor.Controller`), and (c) adding Nerfstudio reconstruction for any custom lab or anatomy-lab footage you shoot with a standard smartphone. Total estimated annotated pairs from ScanNet alone: approximately 2.5 million frame-scene records at 1 fps keyframe density.

---

This proposal is self-contained and ready to present. The strongest opening argument for your mentor is that **Day 1–3 requires zero GPU** — ScanNet already provides pre-computed camera poses and 3D meshes, so the pairing layer can be prototyped entirely on a laptop before any reconstruction or VLM API budget is spent.