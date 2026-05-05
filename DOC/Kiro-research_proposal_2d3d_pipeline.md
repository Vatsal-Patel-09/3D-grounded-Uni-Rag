# Formal Research Proposal: 2D-to-3D Image-Scene Data Generation and Annotation Pipeline
## For: 3D-Grounded Uni-RAG System in Artificial Intelligence in Education (AIED)

---

## 1. Executive Summary

### The Chosen Approach: Hybrid Strategy (Datasets + Reconstruction)

To overcome the **no-LiDAR hardware constraint** while building a robust 3D-grounded retrieval-augmented generation (Uni-RAG) system for education, we propose a **hybrid pipeline** combining:

1. **Foundation via Existing Datasets:** Leverage large-scale, academically licensed 3D indoor datasets (ScanNet, Replica, Habitat-Matterport3D) that already contain registered 2D-3D pairs, semantic annotations, and camera poses. This provides immediate, high-quality ground truth without hardware costs.

2. **Scalability via Software-Based Reconstruction:** Use open-source 3D Gaussian Splatting (via Nerfstudio/Splatfacto) and photogrammetry pipelines (COLMAP) to reconstruct novel educational environments (classrooms, labs, anatomy models) from commodity smartphone videos or image collections. This requires **zero specialized hardware**—only an NVIDIA GPU (8GB+ VRAM) and Python.

3. **Intelligence via VLM Annotation:** Deploy GPT-4o or Gemini 1.5 Pro with structured JSON output schemas to automatically generate rich, spatially-aware captions from the 2D viewpoints and 3D metadata, creating training data for the Uni-RAG model.

**Key Advantage:** This approach decouples data acquisition from expensive LiDAR scanning, enabling rapid iteration and domain-specific dataset construction for AIED applications.

---

## 2. Tool Stack

### 2.1 Datasets (Pre-existing 2D-3D Pairs)

| Dataset | Scale | License | Key Features | Download |
|---------|-------|---------|--------------|----------|
| **ScanNet** | 1,500 scenes, 2.5M views | Academic (free with registration) | RGB-D video, reconstructed meshes, semantic/instance segmentation, camera poses | http://www.scan-net.org |
| **Replica** | 18 high-fidelity indoor scenes | CC-BY (permissive) | Dense geometry, HDR textures, semantic segmentation, navigation meshes, Habitat-compatible | https://github.com/facebookresearch/Replica-Dataset [^13^] |
| **Habitat-Matterport3D** | 90 building-scale scenes | Academic (Matterport TOS) | Photorealistic renders, semantic annotations, embodied AI navigation graphs | https://aihabitat.org/datasets/hm3d/ |
| **Objaverse-XL** | 10M+ 3D objects | Open (CC variants) | Annotated 3D objects, Blender-compatible, ideal for object-level educational content | https://objaverse.allenai.org [^19^] |

### 2.2 3D Reconstruction Software (No LiDAR Required)

| Tool | Type | Cost | Input | Output | GPU Requirement |
|------|------|------|-------|--------|----------------|
| **Nerfstudio** (Splatfacto method) | Open-source | Free | Images/video | Gaussian Splat PLY, camera poses | NVIDIA 8GB+ VRAM [^16^] |
| **COLMAP** | Open-source | Free | Images | Sparse point cloud, camera intrinsics/extrinsics | CPU (SfM), GPU optional for feature extraction [^11^] |
| **SuperSplat** | Web-based editor | Free (MIT) | PLY files | Cleaned/optimized PLY, cropped scenes | Browser (WebGL) [^16^] |
| **gsplat** | CUDA backend | Free | Nerfstudio backend | 4x memory reduction, 10% faster training | NVIDIA CUDA [^16^] |
| **Luma AI** | Cloud API | Free tier | Smartphone video | GLB/USD/PLY | Cloud (no local GPU) [^2^] |

**Recommended Stack for Lab:**
- **Primary:** Nerfstudio + COLMAP (full control, no API limits, academic freedom)
- **Secondary:** Luma AI API for rapid prototyping when local GPU is unavailable
- **Cleanup:** SuperSplat for post-processing (remove floaters, crop backgrounds)

### 2.3 VLM Annotation APIs

| Model | Provider | Structured Output | Multi-image Input | Spatial Reasoning |
|-------|----------|-------------------|-------------------|-------------------|
| **GPT-4o** | OpenAI | `response_format: {type: "json_object"}` + function calling [^23^] | Yes (multiple image_url in single message) | Strong with visual prompting (3DAxisPrompt) [^5^] |
| **Gemini 1.5 Pro** | Google | Native `response_json_schema` with Pydantic/Zod support [^21^][^24^] | Yes (up to 3,600 images in context) | Excellent long-context spatial tracking |
| **GPT4Scene paradigm** | Research (HKU/SH AI Lab) | BEV image + marked frames for 3D understanding [^1^] | Video frames + BEV composite | SOTA on ScanQA/SQA3D benchmarks |

---

## 3. Pipeline Architecture

### 3.1 Data Ingestion Layer

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA INGESTION                             │
├─────────────────────────────────────────────────────────────┤
│  Path A: Existing Datasets                                    │
│    ├── ScanNet: Download RGB-D sequences + mesh + poses   │
│    ├── Replica: Download Habitat-compatible GLB scenes      │
│    └── Objaverse-XL: Download 3D objects via HuggingFace  │
│                                                              │
│  Path B: Novel Capture (No LiDAR)                          │
│    ├── Input: Smartphone video (30-60s, slow orbit)         │
│    ├── Extract frames: ffmpeg -i video.mp4 -vf fps=4 frames/%04d.jpg
│    └── Quality check: Minimum 50 frames, >70% overlap     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Processing Layer: 3D Reconstruction

```
┌─────────────────────────────────────────────────────────────┐
│              3D RECONSTRUCTION PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│  Step 1: Camera Pose Estimation (COLMAP)                    │
│    ├── Feature extraction: SIFT with GPU acceleration       │
│    ├── Matching: Sequential matching for video sequences    │
│    ├── SfM: Sparse reconstruction + bundle adjustment       │
│    └── Output: cameras.bin, images.bin, points3D.bin        │
│                                                              │
│  Step 2: Data Conversion (Nerfstudio)                       │
│    ├── ns-process-data images --data ./frames --output-dir ./processed
│    ├── Converts COLMAP poses → Nerfstudio format (transforms.json)
│    └── Inverts world-to-camera → camera-to-world poses     │
│                                                              │
│  Step 3: Gaussian Splatting Training                        │
│    ├── ns-train splatfacto --data ./processed/              │
│    ├── Iterations: 30,000 (≈15-30 min on RTX 4090)        │
│    └── Export: ns-export gaussian-splat --load-config ...   │
│                                                              │
│  Step 4: Post-Processing (SuperSplat)                        │
│    ├── Upload PLY to superspl.at/editor                     │
│    ├── Remove floaters, crop to region of interest          │
│    └── Export compressed PLY (70-90% size reduction)        │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Pairing Layer: 2D-to-3D Correspondence

This is the **critical mathematical linkage** for Uni-RAG training:

```python
# Core Concept: Each 2D frame has a known camera pose in the 3D scene coordinate system
# From COLMAP/Nerfstudio, we obtain for each frame i:
#   - Camera-to-world transformation matrix T_wc_i (4x4)
#   - Intrinsics matrix K (3x3)
#   - Image file path

# For any 3D point P = [x, y, z, 1]^T in the Gaussian Splat:
#   p_2d = K * [R | t] * P  (projected to image plane)
#   where [R | t] = T_wc_i[:3, :] (first 3 rows of 4x4 matrix)

# This enables:
# 1. Given a 2D pixel, ray-cast into 3D to find intersected Gaussians
# 2. Given a 3D object, project its bounding box to all 2D frames where it's visible
# 3. Compute visibility: dot(camera_ray, surface_normal) > threshold
```

**Implementation Strategy:**
- Store `transforms.json` (from Nerfstudio) alongside each scene
- For dataset samples: Use ScanNet's `axisAlignment` matrices + `pose` files
- Build an index: `{scene_id: {frame_id: {image_path, pose_matrix, depth_path, timestamp}}}`

### 3.4 Annotation Layer: VLM Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              VLM ANNOTATION PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│  Input Assembly (per scene):                                │
│    ├── Select N=8 keyframes (evenly spaced temporally)      │
│    ├── Generate BEV (Bird's Eye View) composite image      │
│    │   └── Project 3D point cloud top-down, color by height │
│    ├── Overlay object markers (SAM + ID labels) on frames  │
│    └── Package: [BEV_image] + [frame_1, ..., frame_N]       │
│                                                              │
│  VLM Prompt (GPT-4o / Gemini 1.5 Pro):                     │
│    ├── System: "You are a 3D scene understanding expert..." │
│    ├── Visual: BEV + marked frames (GPT4Scene style) [^1^]│
│    └── Task: Generate spatial caption + object graph         │
│                                                              │
│  Output: Structured JSON (see Section 4)                   │
│    ├── Scene-level description                              │
│    ├── Object inventory with 3D coordinates                  │
│    └── Spatial relationship graph                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Annotation Strategy

### 4.1 VLM Prompt Template

```markdown
## SYSTEM PROMPT

You are an expert 3D scene understanding and spatial reasoning agent. 
Your task is to analyze indoor educational environments and produce 
structured, metrically-grounded descriptions suitable for training 
a 3D-aware retrieval model.

You will receive:
1. A BIRD'S EYE VIEW (BEV) image showing the top-down layout of the scene
2. N perspective frames from a video walkthrough, with colored markers (C1, C2, ...) 
   indicating detected objects

## RULES
- Use metric units (meters) for all distances and dimensions
- Describe spatial relationships using precise directional terms 
  ("2.5m to the left of", "adjacent to", "facing toward")
- Identify educational objects specifically (whiteboard, projector, lab bench, skeleton model)
- If uncertain about a measurement, provide a range (e.g., "approximately 1.5-2.0m")
- Do NOT hallucinate objects not visible in the frames

## OUTPUT FORMAT
Return ONLY a JSON object matching the schema below. No markdown, no explanations.
```

### 4.2 JSON Schema for Uni-RAG Training Data

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "title": "SpatialSceneAnnotation",
  "required": ["scene_id", "scene_type", "global_description", "coordinate_system", "objects", "spatial_relationships", "viewpoints"],
  "properties": {
    "scene_id": {
      "type": "string",
      "description": "Unique identifier for the scene (e.g., 'classroom_001')"
    },
    "scene_type": {
      "type": "string",
      "enum": ["classroom", "laboratory", "library", "lecture_hall", "anatomy_lab", "virtual_museum"],
      "description": "Educational environment category"
    },
    "global_description": {
      "type": "string",
      "description": "2-3 sentence holistic description of the scene's purpose and layout"
    },
    "coordinate_system": {
      "type": "object",
      "required": ["origin", "unit", "up_axis"],
      "properties": {
        "origin": {
          "type": "string",
          "description": "Definition of coordinate origin (e.g., 'room_center', 'entrance_door')"
        },
        "unit": {
          "type": "string",
          "enum": ["meters", "centimeters"],
          "description": "Spatial unit of measurement"
        },
        "up_axis": {
          "type": "string",
          "enum": ["y", "z"],
          "description": "Which axis points upward in the 3D coordinate system"
        }
      }
    },
    "objects": {
      "type": "array",
      "description": "All semantically meaningful objects in the scene",
      "items": {
        "type": "object",
        "required": ["object_id", "category", "description", "bbox_3d", "centroid_3d", "visible_in_views"],
        "properties": {
          "object_id": {
            "type": "string",
            "description": "Unique object identifier (e.g., 'obj_001')"
          },
          "category": {
            "type": "string",
            "enum": ["whiteboard", "desk", "chair", "projector", "bookshelf", "lab_equipment", "skeleton_model", "poster", "window", "door", "computer", "podium"],
            "description": "Semantic category of the object"
          },
          "description": {
            "type": "string",
            "description": "Detailed visual description (color, material, state)"
          },
          "bbox_3d": {
            "type": "object",
            "required": ["min", "max"],
            "properties": {
              "min": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
                "description": "[x, y, z] minimum corner in world coordinates"
              },
              "max": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
                "description": "[x, y, z] maximum corner in world coordinates"
              }
            }
          },
          "centroid_3d": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
            "description": "[x, y, z] center point of the object"
          },
          "visible_in_views": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of frame_ids where this object is visible"
          }
        }
      }
    },
    "spatial_relationships": {
      "type": "array",
      "description": "Pairwise spatial relationships between objects",
      "items": {
        "type": "object",
        "required": ["subject_id", "object_id", "relationship", "distance_m", "confidence"],
        "properties": {
          "subject_id": {
            "type": "string",
            "description": "ID of the reference object"
          },
          "object_id": {
            "type": "string",
            "description": "ID of the target object"
          },
          "relationship": {
            "type": "string",
            "enum": ["left_of", "right_of", "in_front_of", "behind", "above", "below", "adjacent_to", "facing", "contained_in", "supported_by"],
            "description": "Spatial predicate from subject to object"
          },
          "distance_m": {
            "type": "number",
            "description": "Estimated metric distance between centroids"
          },
          "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "VLM confidence score for this relationship"
          }
        }
      }
    },
    "viewpoints": {
      "type": "array",
      "description": "Camera viewpoints used for this annotation",
      "items": {
        "type": "object",
        "required": ["view_id", "image_path", "camera_pose", "timestamp"],
        "properties": {
          "view_id": {
            "type": "string",
            "description": "Frame identifier"
          },
          "image_path": {
            "type": "string",
            "description": "Relative path to the 2D image file"
          },
          "camera_pose": {
            "type": "object",
            "required": ["position", "rotation"],
            "properties": {
              "position": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
                "description": "[x, y, z] camera position in world coordinates"
              },
              "rotation": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "[qx, qy, qz, qw] quaternion orientation"
              }
            }
          },
          "timestamp": {
            "type": "number",
            "description": "Optional temporal offset in seconds"
          }
        }
      }
    }
  }
}
```

### 4.3 API Implementation (Python Pseudocode)

```python
# GPT-4o Implementation
import openai
import json

client = openai.OpenAI(api_key="YOUR_KEY")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": f"Analyze this educational scene. BEV and {N} frames follow."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{bev_base64}"}},
            *[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame}"}} for frame in frames_base64]
        ]}
    ],
    response_format={"type": "json_object"},  # Enforces valid JSON [^23^]
    temperature=0.2  # Low variance for consistent structure
)

annotation = json.loads(response.choices[0].message.content)
```

```python
# Gemini 1.5 Pro Implementation
from google import genai
from pydantic import BaseModel

class SpatialSceneAnnotation(BaseModel):
    scene_id: str
    scene_type: str
    global_description: str
    # ... (full schema as Pydantic model)

client = genai.Client(api_key="YOUR_KEY")

response = client.models.generate_content(
    model="gemini-1.5-pro-002",
    contents=[prompt_text, bev_image, *frame_images],
    config={
        "response_mime_type": "application/json",
        "response_json_schema": SpatialSceneAnnotation.model_json_schema()  # Native schema enforcement [^21^]
    }
)

annotation = SpatialSceneAnnotation.model_validate_json(response.text)
```

---

## 5. Immediate Next Steps (Proof-of-Concept Sprint)

### Week 1: Environment & Dataset Setup

**Day 1-2: Hardware & Software Installation**
```bash
# 1. Verify NVIDIA GPU (minimum RTX 3060 8GB, recommended RTX 4090 24GB)
nvidia-smi

# 2. Install Nerfstudio (Python 3.10 required)
pip install nerfstudio

# 3. Install COLMAP (Ubuntu/Windows WSL)
sudo apt-get install colmap

# 4. Verify installation
ns-train --help
ns-process-data --help
```

**Day 3-4: Download Foundation Dataset**
```bash
# Download Replica dataset (18 scenes, CC-BY license, no registration)
# Mirror: https://sourceforge.net/projects/replica-dataset.mirror/
git clone https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
# Follow README to download 3D meshes and Habitat configs

# Alternative: ScanNet (requires academic registration)
# Visit http://www.scan-net.org and request download credentials
```

**Day 5-7: First Reconstruction from Video**
```bash
# Capture a 30-second slow orbit video of your lab/office with smartphone
# Extract frames at 4 FPS
ffmpeg -i lab_walkthrough.mp4 -vf "fps=4,scale=1920:1080" frames/%04d.jpg

# Process with Nerfstudio pipeline
ns-process-data images --data ./frames --output-dir ./processed_lab
ns-train splatfacto --data ./processed_lab/ --max-num-iterations 30000

# Export Gaussian Splat
ns-export gaussian-splat --load-config outputs/*/config.yml --output-dir ./exports/
```

### Week 2: Annotation Pipeline Prototype

**Day 8-10: VLM API Setup**
- Register for OpenAI API (GPT-4o) or Google AI Studio (Gemini 1.5 Pro free tier)
- Implement base64 image encoding for frame batches
- Test structured JSON output with simple scene descriptions

**Day 11-12: BEV Generation**
- Write Python script to project exported PLY point cloud to top-down view
- Color-code by height (z-coordinate) for visual salience
- Overlay 2D object bounding boxes from projected 3D annotations

**Day 13-14: End-to-End Integration**
- Connect: Video → Nerfstudio → PLY → SuperSplat cleanup → BEV + Frames → VLM → JSON
- Store outputs in MongoDB or PostgreSQL with JSONB columns
- Validate schema compliance using Pydantic models

### Week 3: Uni-RAG Data Format Preparation

**Goal:** Transform annotations into training pairs for your retrieval model:

```json
// Training example for Uni-RAG
{
  "query": "Where is the whiteboard relative to the teacher's desk?",
  "positive_scene": "classroom_001",
  "positive_view": "frame_012",
  "spatial_context": {
    "whiteboard": {"centroid": [2.1, 1.5, -0.8]},
    "teacher_desk": {"centroid": [0.0, 0.0, 2.5]}
  },
  "answer": "The whiteboard is 2.1 meters to the left and 3.3 meters in front of the teacher's desk.",
  "3d_asset_path": "gs://bucket/classroom_001/splat.ply"
}
```

---

## 6. Risk Mitigation & Alternatives

| Risk | Mitigation |
|------|-----------|
| **COLMAP fails on textureless scenes** | Add ArUco markers or patterned calibration objects to the scene [^11^] |
| **GPU memory insufficient for large scenes** | Use `gsplat` backend (4x memory reduction) or process in chunks [^16^] |
| **VLM hallucinates spatial relationships** | Use GPT4Scene visual prompting (BEV + marked frames) to ground predictions [^1^] |
| **Dataset licensing restrictions** | Prioritize Replica (CC-BY) and Objaverse-XL (open licenses) over ScanNet |
| **Reconstruction quality poor** | Ensure >70% frame overlap, good lighting, avoid reflective surfaces |

---

## 7. Expected Deliverables for Lab Mentor Review

1. **Dataset Inventory:** Table of 50+ annotated scenes (mix of Replica/ScanNet + lab reconstructions)
2. **Pipeline Demo:** Jupyter notebook showing video → 3D splat → JSON annotation in <10 minutes
3. **Annotation Corpus:** 1,000+ spatial relationship triplets with 3D coordinates
4. **Qualitative Results:** Side-by-side renders of reconstructed educational environments
5. **Uni-RAG Training Data:** 500 query-scene-answer triplets ready for model fine-tuning

---

*Proposal prepared by: Chief AI Architect & Orchestrator*
*Date: April 30, 2026*
*Target: Top-tier University AIED Research Lab*
