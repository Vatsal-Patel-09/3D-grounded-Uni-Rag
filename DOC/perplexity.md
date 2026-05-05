<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# **System Role \& Objective**

You are the **Chief AI Architect \& Orchestrator**. Your objective is to autonomously research, design, and architect a highly efficient "2D-to-3D Image-Scene Data Generation and Annotation Pipeline."

**Context**
I am conducting research at a top-tier university lab focused on building a "3D-grounded Uni-RAG" system for Artificial Intelligence in Education (AIED). The goal is to retrieve and generate interactive 3D visuals/videos in spatial environments based on user queries (e.g., teaching human anatomy or exploring a virtual classroom).
**The Constraint:** I do not have access to LiDAR hardware (like an iPhone Pro).
**The Task:** I need to build a pipeline that generates diverse 2D image and 3D scene pairs (either synthetically, via open-source datasets, or through 2D video/image reconstruction) and then automatically annotates them with rich, spatial captions using a Vision-Language Model (VLM) like Gemini or GPT-4o.

**Agent Swarm Initialization**
To accomplish this, you must simulate an autonomous swarm of three specialized sub-agents. You will coordinate their research and synthesize their findings into a final, actionable proposal.

**Agent 1: The 3D Computer Vision Researcher**
**Role:** Identify the best state-of-the-art, open-source, or low-cost tools and datasets to solve the hardware constraint.
**Tasks:**
1. Research existing large-scale 3D indoor/educational datasets (e.g., ScanNet, Matterport3D, Habitat) that already contain 2D-3D pairs and check their licensing for academic use.
2. Research software-based 3D reconstruction techniques that do not require LiDAR (e.g., Nerfstudio, 3D Gaussian Splatting, photogrammetry APIs like Luma AI or Polycam).
3. Evaluate these options based on scalability, compute requirements, and quality of spatial output.

**Agent 2: The Data Pipeline Engineer**
**Role:** Design the step-by-step technical architecture of the pipeline.
**Tasks:**
1. Define the ingestion step (Where do the raw 2D videos/images or dataset files come from?).
2. Define the processing step (How are frames extracted? How are they batched for 3D reconstruction?).
3. Define the pairing step (How do we mathematically or programmatically link specific 2D viewpoints to the generated 3D scene coordinates?).

**Agent 3: The Multimodal Prompt Engineer**
**Role:** Develop the automated annotation strategy.
**Tasks:**
1. Design a robust VLM prompt framework (for Gemini 1.5 Pro or GPT-4o) that takes the 2D images (and ideally 3D spatial metadata) and generates highly detailed, spatial captions (e.g., "A wooden desk is positioned 2 meters in front of a whiteboard...").
2. Define the exact JSON schema for how these annotations should be stored for training a Unified Retrieval model.

**Execution Protocol \& Deliverables**
Execute the swarm's tasks step-by-step. Do not provide generic advice. Provide concrete tool names, dataset links, and architectural decisions.

Synthesize the final output into a **Formal Research Proposal** that I can present to my lab mentor. The proposal must include:

1. **Executive Summary:** The chosen approach (Datasets vs. Reconstruction).
2. **Tool Stack:** Exact software, APIs, and models to be used.
3. **Pipeline Architecture:** Step-by-step data flow.
4. **Annotation Strategy:** The VLM prompt template and JSON schema.
5. **Immediate Next Steps:** What I need to code or download today to build the Proof of Concept.

Here is a concrete, lab-ready proposal for a 2D↔3D data generation and spatial annotation pipeline that avoids any LiDAR requirement and is optimized for Uni-RAG training in AIED.

***

## Executive summary

You will combine two complementary strategies:

1. **Leverage existing 3D indoor datasets with 2D–3D pairs** (ScanNet, Matterport3D, Replica, Hypersim, HSSD/HM3D, AI2-THOR/Habitat) under their non‑commercial academic licenses.[^1][^2][^3][^4][^5][^6][^7][^8]
2. **Generate new 3D scenes from standard RGB images/videos** using a NeRF/3D Gaussian Splatting pipeline (Nerfstudio + COLMAP + gsplat), plus optional low-friction photogrammetry services (Luma AI, Polycam) for quick scanning of your own educational environments.[^9][^10][^11][^12][^13][^14][^15][^16]

On top of these 2D–3D pairs (images + camera poses + 3D scene), you will run a **two-stage annotation pipeline**: (a) extract structured geometric metadata from 3D (object positions, distances, relations), then (b) feed each image + metadata into a VLM (Gemini 1.5 Pro or GPT‑4o) with a strict JSON output format to produce rich spatial captions for Uni‑RAG training.[^17][^18][^19][^20][^21][^22][^23][^24][^25]

***

## Core datasets and 3D resources (Agent 1)

### Recommended primary datasets

These give you large-scale indoor 2D–3D pairs with academic-friendly licensing (non‑commercial, research/education):

- **ScanNet / ScanNet++** – RGB‑D video dataset with ~2.5M views across 1500+ scans, including 3D camera poses, surface reconstructions, and instance-level semantic segmentations; per-scan assets include .sens RGB‑D streams, reconstructed meshes, and projected 2D labels.[^2][^26]
- **Matterport3D \& HM3D (Habitat-Matterport 3D)** – Building-scale RGB‑D scans with panoramic RGB images, depth, global surface reconstructions, and 2D/3D semantic labels; Matterport data for Meta’s HM3D release is explicitly licensed for academic, non‑commercial use.[^27][^3][^28]
- **Replica** – 18 highly photorealistic room- and building-scale 3D reconstructions with dense meshes, HDR textures, and semantic class/instance annotations, natively compatible with Habitat.[^5][^29]
- **Hypersim** – Synthetic dataset with 77,400 images of 461 indoor scenes, each with complete scene geometry, materials, lighting, dense per-pixel semantics, and full camera information.[^4][^30]
- **3RScan** – 1482 RGB‑D scans of 478 environments captured at multiple time steps with semantic annotations and 6‑DoF mappings for changing objects, designed for 3D object re-localization and scene understanding.[^31][^32]
- **ARKitScenes** – 5,047 LiDAR-based captures of 1,661 indoor scenes with camera poses, surface reconstructions, and oriented 3D bounding boxes for furniture; included in SceneSplat‑7K and released for non‑commercial research.[^33][^34][^1]
- **HSSD \& HM3D-compatible synthetic scenes** – Habitat Synthetic Scenes Dataset (HSSD‑200) has 211 high‑quality synthetic interior scenes and ~18k object models under CC BY‑NC 4.0, designed specifically for embodied navigation and generalization to real indoor environments.[^6][^7][^8]

All of these datasets restrict use to **non‑commercial research and education**, which aligns with a university lab; most require agreeing to dataset-specific terms and proper attribution.[^1][^27][^6]

### Pre-splat scenes: SceneSplat‑7K

To bypass much of the 3D reconstruction cost, you can use **SceneSplat‑7K**, a large-scale 3D Gaussian Splatting dataset aggregating 7916 precomputed 3DGS scenes derived from ScanNet, ScanNet++, Replica, Hypersim, 3RScan, ARKitScenes, and Matterport3D, with ~4.72M RGB frames.[^35][^36][^1]

- SceneSplat‑7K provides ready-to-use Gaussian scenes (≈1.4M Gaussians per scene on average) with PSNR ~29.6 for high‑quality rendering, built from the underlying datasets listed above, all under non‑commercial research licenses.[^36][^35][^1]
- Using this, you can focus on **2D–3D alignment and annotation** without first training NeRF/GS models yourself for those scenes.


### 3D reconstruction without LiDAR

For your own custom educational scenes (classrooms, labs, anatomy models), or for datasets that only have posed RGB frames, you can construct 3D representations using commodity cameras:

- **COLMAP (SfM + MVS)** – General-purpose 3D reconstruction pipeline (Structure-from-Motion + Multi-View Stereo) with GUI/CLI, BSD-licensed, supports ordered/unordered image collections, and widely used as a standard for image-based recon.[^15][^37]
- **Nerfstudio** – A modular NeRF/3DGS framework that takes posed images (e.g., from COLMAP, Polycam, or mobile capture apps) and optimizes a neural 3D representation; it supports NeRF and 3D Gaussian Splatting via modules like splatfacto and advanced variants (EGGS, EVolSplat4D).[^14][^16]
- **gsplat** – CUDA-accelerated Gaussian Splatting rasterizer used inside Nerfstudio, optimized for speed and memory efficiency; enables real-time rendering at 60+ FPS and training in minutes–hours rather than days.[^38][^9][^14]
- **Luma AI** – Multi-platform photogrammetry app and web service that builds 3D scenes from commodity phone cameras (no LiDAR required); evaluations emphasize its educational potential and the fact that expensive dedicated hardware is no longer necessary for high-quality photogrammetry.[^10][^12]
- **Polycam** – Mobile and web capture app; their open-source `polyform` tools (MIT-licensed) facilitate export to NeRF formats, and their ToS allow copying, modifying, and commercial use of models you generate as long as you attribute Polycam for public display, with API access governed by documented usage limits.[^11][^13][^16]

**Scalability and compute trade-off**

- **Pre-existing 3D datasets (ScanNet, Matterport3D, SceneSplat‑7K)** give you thousands of scenes at scale with high-quality geometry and pose, at the cost of some dataset-specific tooling and storage (SceneSplat‑7K alone is ~2.76 TB).[^26][^35]
- **Nerfstudio + COLMAP** scales well per scene on 1 modern GPU (e.g., 12–24 GB VRAM), with Gaussian Splatting training in minutes to a couple hours per scene, particularly when using splatfacto and gsplat.[^9][^38][^14]
- **Luma / Polycam** offload reconstruction to their infrastructure and offer the fastest workflow for scanning bespoke educational spaces, but at the cost of API dependence and ToS constraints, making them best suited as a *complement* rather than the core research dataset.

***

## Pipeline architecture (Agent 2)

Below is the step-by-step technical architecture, from ingestion to 2D–3D pairing, designed so you can plug in multiple data sources.

### 1. Ingestion layer

**Data sources**

- **Static 3D datasets**
    - ScanNet / ScanNet++: ingest `.sens` RGB‑D streams, meshes, camera poses, and 2D projection zips.[^26]
    - Matterport3D / HM3D: ingest panoramas, depth maps, and alignment/pose files from the official downloads.[^3][^28]
    - Replica, Hypersim, 3RScan, ARKitScenes, HSSD: ingest scene meshes, textures, camera trajectories, and annotations from their repositories or Habitat-compatible variants.[^30][^32][^7][^8][^5][^33][^6]
    - SceneSplat‑7K: ingest precomputed Gaussian splat scenes and associated source frame metadata from its Hugging Face distribution.[^35][^36][^1]
- **Your own captures**
    - **Video or image sequences** from any smartphone or DSLR (no depth required).
    - Optionally, **Luma/Polycam exports** (meshes/point clouds/NeRF assets) for your lab spaces and physical teaching props.[^12][^16][^10][^11]

**Ingestion services**

Implement a modular Python-based ingestion service that:

- Normalizes paths and metadata into a common catalog table: `(scene_id, source_dataset, raw_path, modality, license_tag, scale_info)`.
- Stores camera intrinsics/extrinsics, if available, in a standard format (e.g., `K` 3×3, `R` 3×3, `t` 3×1, following OpenCV/COLMAP conventions).


### 2. Processing and reconstruction

**2.1 Frame extraction and pose handling**

- For datasets that already provide camera poses (ScanNet, Matterport3D, Hypersim, Replica, ARKitScenes, HSSD/HM3D), you simply **sample frames** from their RGB streams or panoramas and read the corresponding camera extrinsics/intrinsics from their metadata.[^7][^3][^4][^5][^33][^6][^26]
- For your own videos or unposed image sets:
    - Run **COLMAP** SfM to estimate sparse 3D points and camera poses, then MVS to densify the reconstruction.[^37][^15]
    - Export the poses and reconstructed sparse/dense point clouds in COLMAP’s standard formats, then convert to Nerfstudio’s data format using their utilities.[^16]

During this step you also:

- Decide on a consistent **world coordinate system** (e.g., z-up, meters) and rescale reconstructions when necessary (either via known scene dimensions or using dataset scale metadata).

**2.2 3D representation training (optional when not already provided)**

For scenes that do not come with high-quality meshes or Gaussian splats:

- Use **Nerfstudio** to train either:
    - A NeRF-based model (e.g., `nerfacto`, `instant-ngp`), or
    - A 3D Gaussian Splatting model (e.g., `splatfacto`) leveraging **gsplat** for fast training and rendering.[^38][^14][^16][^9]
- Nerfstudio natively supports images/videos + camera poses, as well as inputs from mobile capture apps and photogrammetry tools like RealityCapture and Polycam, making it easy to integrate external sources.[^14][^16]

Outputs you should keep per scene:

- A persistent **3D representation**:
    - 3DGS checkpoint (Gaussians with position, scale, opacity, color) **or**
    - Mesh/point cloud (from COLMAP, HSSD/Replica meshes, or photogrammetry exports).[^5][^6][^7][^14]
- **Per-frame camera parameters** aligned to the above representation.
- Optional **depth and normal maps** rendered from the representation for each frame using Nerfstudio’s rendering tools.[^16][^14]


### 3. 2D–3D pairing and coordinate linkage

This is the crucial step for Uni‑RAG: for every 2D view, you need a precise mapping to 3D scene coordinates.

**3.1 Minimal pairing (for retrieval)**

For each frame $i$ from a scene $s$:

- Store:
    - `image_id`, `scene_id`
    - Camera intrinsics $K_i$ and extrinsics $(R_i, t_i)$ in the chosen world coordinate frame.
    - A reference to the 3D representation (mesh file, GS checkpoint ID, or scene URI).
- This lets you later project any 3D point $X_{world}$ into image coordinates via $x_{image} \sim K_i[R_i \,|\, t_i] X_{world}$, giving you a direct mathematical link between 2D and 3D for any pixel or object.

**3.2 Object-level pairing**

Where object annotations are available (ScanNet labels, Matterport3D semantics, Replica/Hypersim instance IDs, HSSD object configs, AI2‑THOR configs), you can go further:[^39][^28][^3][^30][^6][^7][^5][^26]

- For each object instance $o$ with a 3D bounding box or mesh subset:
    - Compute its 3D **centroid** $c_o$ and approximate **extent** $e_o$ in world coordinates.
    - For each image, project the 8 corners of its 3D box into 2D using the camera matrix to derive tight 2D bounding boxes and visibility masks.
- Store, per (image, object) pair:
    - `bbox_3d` (center, size, orientation),
    - `bbox_2d` (x, y, w, h),
    - `visible` flag and approximate **distance to camera** $d_{cam} = \| R_i c_o + t_i \|$.

For reconstructed scenes without ground-truth semantics, you can plug in a **3D instance segmentation model** (e.g., from Open3D-ML or existing indoor segmentation networks) and treat detected clusters as pseudo-objects; details are flexible and not tied to any dataset.

**3.3 Spatial relation extraction**

Before calling a VLM, you want *structured spatial facts*:

- For all visible objects in an image, compute pairwise relations like:
    - `in_front_of` / `behind` based on depth ordering along the camera z‑axis.
    - `left_of` / `right_of` based on x image coordinates or world x after transforming into camera frame.
    - `above` / `below` based on y image coordinates or world y/z coordinates.
    - Metric distances between object centroids in world space.
- These relations will be provided to the VLM as **machine-derived metadata** and then verbalized and enriched into natural language captions.

***

## Tool stack summary

### Major components

| Layer | Function | Recommended tools |
| :-- | :-- | :-- |
| Data hosting | Store raw datasets (TB-scale) | Lab NFS / object storage (S3 / GCS) |
| 3D indoor data | Static 3D + 2D–3D pairs | ScanNet, Matterport3D/HM3D, Replica, Hypersim, 3RScan, ARKitScenes, HSSD, AI2‑THOR/Habitat, SceneSplat‑7K[^2][^3][^4][^5][^32][^33][^6][^7][^8][^39][^35][^36] |
| Reconstruction | SfM/MVS from RGB | COLMAP[^15][^37] |
| Neural 3D | NeRF / 3DGS training \& rendering | Nerfstudio (`nerfacto`, `splatfacto`, etc.) + gsplat[^14][^16][^9][^38] |
| Photogrammetry (optional) | Fast scanning of custom scenes | Luma AI, Polycam (+ polyform)[^12][^10][^11][^13] |
| Geometry ops | 3D I/O, ray casting, segmentation | PyTorch3D / Open3D / trimesh, Open3D‑ML |
| VLM API | Spatial captioning | Gemini 1.5 Pro / 2.x; GPT‑4o / GPT‑4o‑mini[^17][^23][^25][^18][^20][^22][^24][^19][^21] |
| Orchestration | Pipelines, batching | Python + Airflow or Prefect |
| Storage for annotations | Training data | Parquet/JSONL in object store + Postgres/SQLite index |


***

## Annotation strategy (Agent 3)

### VLM capabilities and choice

- **Gemini 1.5 Pro**: multimodal model with very long context (up to ~10M tokens) and native support for images, video, and interleaved text, optimized for long-context multimodal reasoning; suitable if you want to feed sequences of frames plus extensive metadata.[^23][^25][^17]
- **Gemini 2/3 Pro**: further emphasize spatial understanding in recent releases, with tools in AI Studio to query positions of objects in images.[^19][^21]
- **GPT‑4o**: strong spatial awareness for images (can describe layouts, compare images) and supports structured JSON outputs via `response_format: { type: "json_object" }`, as showcased in OpenAI’s cookbook examples.[^18][^20][^22][^24]

For a lab setting, selecting **one primary VLM** (e.g., GPT‑4o or Gemini 1.5 Pro) is recommended for consistency; the design below works with both.

### Two-stage annotation process

1. **Geometry-driven pre-annotation**
    - From the 3D data (meshes/3DGS + object annotations), compute a compact, machine-readable summary per image:
        - List of detected objects with categories, 2D/3D boxes, centroids, sizes, and distances.
        - Pairwise spatial relations (left/right, in front of/behind, near/far, above/below).
        - Global room-type and high-level scene tags (e.g., classroom, lab, anatomy exhibit), derived from dataset metadata or VLM classification on the raw image.
    - This step is deterministic and fully under your control, ensuring metric correctness.
2. **VLM-based natural language + structured captioning**
    - For each (image, metadata) pair, call the VLM with:
        - The **image** itself.
        - A **structured JSON blob** describing objects and relations.
        - A **prompt** that instructs the VLM to:
            - Produce a detailed prose caption with explicit metric phrasing (e.g., “A wooden desk is approximately 2 meters in front of the whiteboard.”).
            - Produce a normalized **JSON object** following your schema.

This approach ensures your spatial descriptions are grounded in real 3D geometry but expressed in natural, teacher-friendly language that a Uni‑RAG system can index and retrieve.

### Prompt framework (for GPT‑4o / Gemini 1.5 Pro)

**System message template**

> You are an expert educational spatial scene annotator.
> You are given a single RGB image from a 3D indoor educational environment and structured metadata about detected objects and their 3D layout.
> Your task is to (1) verify and refine the spatial information using the image, and (2) output a JSON object that accurately describes the scene for a 3D-grounded retrieval system.
> Always ground distances and directions in the camera’s viewpoint unless otherwise specified.
> Use metric units (meters) when specifying distances.
> Do not invent objects that are not visible.
> If metadata and image disagree, prefer the image and note low confidence.

**User message template (conceptual)**

Content (simplified):

1. Text part:

> Here is the image and the structured metadata for one camera view of a 3D scene.
> 1. Read the `metadata_json` carefully.
> 2. Use the image to correct obvious errors and enrich descriptions.
> 3. Output **only** a JSON object following the given schema.
> Do not include any additional text.

2. `image_url` part: the frame image.
3. `metadata_json` part (as text) – for example:
```json
{
  "scene_id": "scannet_0001_00",
  "image_id": "scannet_0001_00_0123",
  "camera": {
    "position_world": [1.2, 1.6, 0.8],
    "forward_world": [0.0, 0.0, 1.0]
  },
  "objects": [
    {
      "object_id": "desk_1",
      "category": "desk",
      "center_world": [1.5, 0.8, 2.0],
      "size_world": [1.2, 0.75, 0.6],
      "distance_to_camera": 2.1,
      "bbox_2d": [320, 280, 180, 120],
      "relations": [
        { "relation": "in_front_of", "other_id": "whiteboard_1" }
      ]
    },
    ...
  ]
}
```

For **GPT‑4o**, specify `response_format: { "type": "json_object" }` as in the cookbook examples to guarantee JSON output; Gemini has analogous structured-output modes via its SDKs.[^20][^22][^25][^17][^23]

***

## JSON schema for annotations

Below is a concrete JSON schema for each annotated image. This is your **training unit** for Uni‑RAG: one record per `(scene_id, image_id)`.

```json
{
  "scene_id": "string",             // unique scene identifier (from dataset or reconstruction)
  "image_id": "string",             // unique image identifier
  "dataset": "string",              // e.g., "scannet", "matterport3d", "custom_recon"

  "camera": {
    "position_world": [0.0, 0.0, 0.0],
    "rotation_world": [[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]],
    "intrinsics": {
      "fx": 0.0, "fy": 0.0,
      "cx": 0.0, "cy": 0.0,
      "width": 0, "height": 0
    },
    "coordinate_frame": "right-handed_z_up"
  },

  "global_caption": "string",       // detailed natural language description of the scene

  "scene_tags": [                   // high-level tags for retrieval
    "classroom", "lab", "anatomy", "virtual_whiteboard"
  ],

  "objects": [
    {
      "object_id": "string",        // stable within scene (e.g., "desk_1")
      "category": "string",         // coarse category (e.g., "desk", "chair", "skeleton_model")
      "subtype": "string",          // optional fine-grained type (e.g., "standing_skeleton")
      "attributes": [               // color/material/state attributes
        "wooden", "brown", "rectangular"
      ],

      "bbox_2d": [0, 0, 0, 0],      // [x, y, width, height] in pixels
      "bbox_3d": {
        "center_world": [0.0, 0.0, 0.0],
        "size_world": [0.0, 0.0, 0.0],   // [dx, dy, dz] in meters
        "orientation_world": [0.0, 0.0, 0.0, 1.0] // quaternion (x, y, z, w)
      },

      "distance_to_camera": 0.0,    // meters (Euclidean from camera position to center)
      "visibility": {
        "visible": true,
        "occlusion_ratio": 0.0      // 0.0–1.0 fraction of object area occluded (approximate)
      },

      "natural_caption": "string",  // short per-object caption (e.g., 'A wooden lab bench with glassware.')

      "canonical_queries": [        // phrases useful for retrieval
        "teacher's desk", "front lab bench"
      ]
    }
  ],

  "spatial_relations": [
    {
      "subject_id": "string",       // object_id of subject
      "relation": "string",         // e.g., "in_front_of", "behind", "left_of",
                                    // "right_of", "above", "below", "near", "on_top_of"
      "object_id": "string",        // object_id of object
      "distance_m": 0.0,            // optional metric distance between centroids
      "frame_of_reference": "camera", // "camera" or "world"
      "confidence": 0.0             // 0.0–1.0 confidence from VLM or geometry
    }
  ],

  "education_specific": {
    "pedagogical_role": "string",   // e.g., "teacher_area", "student_workspace", "demo_zone"
    "target_concepts": [            // AIED concepts taught in this scene
      "human_skeleton_structure", "classroom_layout"
    ]
  },

  "generation_metadata": {
    "vlm_model": "string",          // e.g., "gpt-4o-2024-05-xx" or "gemini-1.5-pro"
    "vlm_prompt_version": "v1.0",
    "geometry_source": "string",    // e.g., "scannet_mesh", "nerfstudio_splatfacto"
    "timestamp": "ISO-8601 string"
  }
}
```

For Uni‑RAG, you can treat `global_caption`, `objects[*].natural_caption`, `scene_tags`, `canonical_queries`, and `spatial_relations` as the main textual fields, and the camera/object geometry as the grounding information.

***

## Pipeline data flow (end-to-end)

Putting it all together, your **2D→3D→Caption** pipeline looks like:

1. **Ingest** scene from dataset or reconstruction: `(scene_id, raw data, camera poses, mesh/3DGS)`.
2. **Sample** one or more frames per scene (or use all frames) and associate each with camera intrinsics/extrinsics.
3. **Compute** per-frame object projections and 3D–2D mappings; derive spatial relations and distances.
4. **Construct** a `metadata_json` blob per frame with object lists, geometric attributes, and relations.
5. **Call** VLM (Gemini or GPT‑4o) with image + `metadata_json` + schema-aware prompt to generate `global_caption`, per-object captions, and relation corrections in a JSON object adhering to your schema.
6. **Store** that JSON (one per (scene, image)) in a central annotation store (e.g., JSONL/Parquet), along with the original 2D image and a pointer to the 3D representation.
7. **Index** these records into your Uni‑RAG retrieval layer (e.g., vector indices over captions and canonical queries, plus symbolic indices over categories and spatial relations).

***

## Immediate next steps (PoC plan)

Here is what you can concretely do this week to get a working proof of concept.

### Step 1: Acquire a manageable dataset slice

1. Request and download **ScanNet** for non‑commercial research (start with a small subset of scans to keep it light).[^2][^26]
2. Optionally, download a few scenes from **HSSD** (Habitat Synthetic Scenes Dataset) or **Replica** for clean synthetic geometry.[^29][^8][^6][^7][^5]
3. If disk allows, also grab a handful of **SceneSplat‑7K** scenes from Hugging Face to avoid doing 3DGS training yourself.[^36][^1][^35]

### Step 2: Set up 3D tooling

1. Install **COLMAP** on your lab machine or workstation (for your own captures and for experiments).[^15][^37]
2. Install **Nerfstudio** and verify you can run a demo training on a sample dataset (e.g., using `nerfacto` or `splatfacto` with gsplat).[^9][^38][^14][^16]
3. Write a Python module that:
    - Given a ScanNet scan, extracts a handful of RGB frames and loads their camera intrinsics/extrinsics from ScanNet’s metadata.[^26]
    - Optionally renders depth/normal maps from NeRF/3DGS for your own reconstructed scenes.

### Step 3: Implement 2D–3D linkage for one dataset

1. For a single ScanNet or Replica scene, build a simple script that:
    - Loads the mesh and semantic annotations.
    - Samples, say, 20 frames and for each:
        - Computes per-object centroids and extents in world coordinates.
        - Projects object boxes into 2D to get `bbox_2d`.
        - Computes distances and simple relations (in_front_of, left_of, etc.).
2. Emit a `metadata_json` per frame matching the `metadata_json` described in the prompt framework.

### Step 4: Wire up the VLM for captioning

1. Choose an API to start with:
    - **GPT‑4o** – follow the OpenAI cookbook for image captioning with JSON outputs (`response_format: "json_object"`).[^22][^24][^18][^20]
    - **Gemini 1.5 Pro** – follow Google’s API tutorial for multimodal requests (images + JSON text), leveraging its long-context and spatial understanding abilities.[^21][^25][^17][^19][^23]
2. Implement a small Python script that:
    - Takes a single image + `metadata_json`.
    - Sends them to the VLM with the system/user prompts described above.
    - Receives and validates the JSON response against your schema.

### Step 5: Store and inspect annotated samples

1. Save the VLM outputs into a JSONL file, one record per image, following the schema.
2. Write a simple visualization script:
    - Overlay `bbox_2d` and object labels on the image.
    - Show the `global_caption` and a subset of `spatial_relations`.
3. Manually inspect 20–50 samples to:
    - Check that distances and qualitative relations match your intuition.
    - Adjust the geometric pre-processing and prompt wording as needed (e.g., controlling verbosity, confidence thresholds, or handling occlusions).

### Step 6: Extend to educational content

Once the PoC works on generic indoor scenes:

1. Capture a **short handheld video** of a real classroom or lab using a standard smartphone.
2. Run **COLMAP → Nerfstudio (splatfacto)** to reconstruct a 3DGS scene.[^37][^38][^14][^15][^16][^9]
3. Reuse the same 2D–3D linkage + VLM annotation pipeline to produce spatially-grounded captions for this real educational environment.

This gives you a compelling end-to-end demo for your mentor: starting from a basic smartphone video, you generate a 3D representation, align 2D views, and automatically produce rich, spatially grounded captions ready for Uni‑RAG training.

***

If you’d like, I can next help you with: (a) a concrete Python skeleton for the metadata extraction and VLM call, or (b) a minimal schema for your retrieval index (e.g., how Uni‑RAG should index `spatial_relations` and `canonical_queries`).
<span style="display:none">[^40][^41][^42][^43]</span>

<div align="center">⁂</div>

[^1]: https://huggingface.co/datasets/GaussianWorld/scene_splat_7k

[^2]: http://www.scan-net.org

[^3]: https://niessner.github.io/Matterport/

[^4]: https://machinelearning.apple.com/research/hypersim

[^5]: https://arxiv.org/abs/1906.05797

[^6]: https://huggingface.co/datasets/yali30/hssd-hab/blob/416f08c9c2c198f9839a2fc993089ea3a9b9e8cf/README.md

[^7]: https://huggingface.co/datasets/hssd/hssd-hab

[^8]: https://arxiv.org/abs/2306.11290

[^9]: https://github.com/nerfstudio-project/gsplat

[^10]: https://as-proceeding.com/index.php/ijanser/article/view/1883

[^11]: https://github.com/orgs/PolyCam/repositories

[^12]: https://as-proceeding.com/index.php/ijanser/article/download/1883/1823/3534

[^13]: https://poly.cam/legal/terms-of-service

[^14]: https://www.thefuture3d.com/software/nerfstudio/

[^15]: https://demuc.de/colmap/

[^16]: https://dl.acm.org/doi/fullHtml/10.1145/3588432.3591516

[^17]: https://www.datacamp.com/tutorial/gemini-pro-api-tutorial

[^18]: https://getstream.io/blog/gpt-4o-vision-guide/

[^19]: https://www.youtube.com/watch?v=-XmoDzDMqj4

[^20]: https://developers.openai.com/cookbook/examples/tag_caption_images_with_gpt4v.md

[^21]: https://blog.google/innovation-and-ai/technology/developers-tools/gemini-3-pro-vision/

[^22]: https://cookbook.openai.com/examples/tag_caption_images_with_gpt4v

[^23]: https://www.emergentmind.com/topics/gemini-1-5-pro

[^24]: https://mirascope.com/docs/v1/guides/more-advanced/generating-captions

[^25]: https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf

[^26]: http://www.scan-net.org/ScanNet/

[^27]: https://matterport.com/partners/meta

[^28]: https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md

[^29]: https://www.academia.edu/103220907/The_Replica_Dataset_A_Digital_Replica_of_Indoor_Spaces

[^30]: https://arxiv.org/abs/2011.02523

[^31]: https://www.emergentmind.com/topics/tum-rgb-d-and-3rscan-datasets

[^32]: https://arxiv.org/abs/1908.06109

[^33]: https://github.com/apple/ARKitScenes

[^34]: https://openreview.net/pdf?id=tjZjv_qh_CE

[^35]: https://unique1i.github.io/SceneSplat_webpage/

[^36]: https://arxiv.org/abs/2503.18052

[^37]: https://colmap.github.io

[^38]: https://github.com/nerfstudio-project/gsplat?search=1

[^39]: https://huggingface.co/datasets/hssd/ai2thor-hab

[^40]: https://arxiv.org/abs/2510.12095

[^41]: https://github.com/NVlabs/instant-ngp/discussions/818

[^42]: https://www.juheapi.com/datasets/hssd-hssd-hab

[^43]: https://openaccess.thecvf.com/content/ICCV2025/papers/Li_SceneSplat_Gaussian_Splatting-based_Scene_Understanding_with_Vision-Language_Pretraining_ICCV_2025_paper.pdf

