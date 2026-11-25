# Random Dot Stereo Club

*For the love of random dots and stereo vision.*

> **Work in progress.** The full **Train** and **Eval** downloads are coming to Zenodo. The links will appear here once the uploads are ready. In the meantime, grab a [sample](#play-with-the-samples) and have fun with it!

Point a random-dot projector at a plain tabletop and suddenly your stereo pair has a pattern to match. That’s what I love about a RealSense setup like the D435 or D555. **Two IR cameras, a color camera, and a random-dot projector!** The projector gives the IR pair texture to work with, and the color camera adds another view. If we already have all three views, why not see what learning-based methods can measure with them?

I wanted a realistic synthetic dataset that would give more people a chance to try those ideas. So I made one. Welcome to the club.

## Dataset at a glance

[![Three tabletop scenes with large RGB renders, left and right IR images, ground-truth disparity, and RGB instance masks](samples/dataset-overview.jpg)](samples/dataset-overview.jpg)

These are renders from the dataset. RGB is shown at a larger size to make the materials, reflections, and transparent objects easier to inspect. Click the image to zoom in.


## Play with the samples

[Download the sample ZIP](samples/sample-e00000.zip) for the same tabletop scene captured with all three camera configurations. It includes the images, ground-truth disparity, masks, calibration, and pose and OBB JSONs.

To use the shared meshes, download or clone this repository, then extract the sample ZIP beside `meshes/`. It creates a `sample/` folder. The Train and Eval archives will create sibling folders.

| Camera | Visual preview |
| --- | --- |
| D435 | [View preview](samples/D435.png) |
| D555 | [View preview](samples/D555.png) |
| OAK-D with a simulated projector | [View preview](samples/OAK-D-active.png) |


The previews show the stereo pair, ground-truth disparity, a LAS2-H prediction, absolute error, and the stereo visibility mask.

![OAK-D stereo pair, ground truth, prediction, and error](samples/OAK-D-active.png)

## Camera calibration

[D435](calibration/D435.json) · [D555](calibration/D555.json) · [OAK-D with simulated projector](calibration/OAK-D-active.json)

The JSON files contain intrinsics for 1280 × 720 images, the stereo baseline, transforms from the left IR camera to the other views, and the projector position and field of view. The optical axes point right (+X), down (+Y), and forward (+Z). Intrinsics are in pixels and translations are in meters. The simulation uses ideal pinhole cameras with zero lens distortion and nominal camera geometry. The OAK-D variant adds a simulated dot projector.

For positive left IR disparity `d`, depth is `Z = fx_left × baseline / d`. Back-project the pixel with `X_left = Z × inv(K_left) × [u, v, 1]ᵀ`. To project it into RGB, apply `X_rgb = R × X_left + t_m`, then project with `K_rgb`.

## Convert disparity to depth with PyTorch

The [example script](examples/disparity_to_depth.py) loads the calibration JSON and runs a placeholder `predict_disparity(left, right)` function. The placeholder returns 32 pixels everywhere, so the output is a fronto-parallel plane.

```bash
python examples/disparity_to_depth.py --calibration calibration/D435.json --output depth_m.npy
```

To use your own network, replace the example input tensors with a preprocessed, rectified stereo pair and implement `predict_disparity` with your model. Return left-reference disparity in pixels as a `[B, H, W]` tensor on the input device. Remove model padding and, if you resized the images, restore both the calibrated image dimensions and the disparity scale.

The example uses CUDA automatically when available. Use `--device cuda` or `--device cpu` to choose explicitly. The conversion runs on the input device and produces a float32 depth image in meters. Zero, negative, and non-finite disparities become NaN. Positive disparities below 1 px are converted as well.

## Dataset

Scenes are rendered in Blender at 1280 × 720. Each camera configuration includes paired IR views, RGB, ground-truth disparity on the left IR and RGB pixel grids, RGB-aligned instance masks, and masks for geometric validity, stereo visibility, and transparent materials.

Both disparity maps store left-minus-right IR disparity in pixels. In `disparity_rgb.pfm`, those values are sampled on the RGB pixel grid. Ground truth comes from the first surface hit, including transparent surfaces and surfaces occluded in the other IR view. Pixels with no surface hit contain NaN. Camera views share the scene geometry, and the two IR views share a random-dot pattern generated from the scene’s recorded seed.

The **Eval** set uses held-out object shapes, indoor HDRIs, and textures. Paired scene IDs let you compare the same layout across all three camera configurations.

| Download | Status |
| --- | --- |
| Train | Zenodo link coming soon |
| Eval | Zenodo link coming soon |

## For 3D object recognition

The 3D annotations pair each object instance with a local STL mesh and its 6DoF pose at capture time. Instance IDs match the RGB segmentation masks. Mesh coordinates and pose translations are in meters. The JSON gives both `T_left_ir_from_mesh` and `T_rgb_from_mesh` as row-major 4 × 4 matrices acting on homogeneous column vectors. Use the corresponding camera intrinsics to project the mesh into an image. `obb_left_ir` and `obb_rgb` give the oriented box center, full XYZ side lengths, and box-to-camera rotation in each view. The box axes follow the mesh frame. Meshes are shared across camera views of the same scene.

![Outlines of three STL objects projected onto the D435 RGB image using their ground-truth poses](samples/stl-pose-overlay.jpg)

[View the same scene with the oriented 3D bounding boxes projected onto RGB.](samples/obb-pose-overlay.jpg)

This example uses the outlines reconstructed from STL geometry and pose JSON. After projection into the RGB view with the scene's occluders, the instance labels agreed with the dataset mask at all 921,600 pixels.

The [sample ZIP](samples/sample-e00000.zip) includes poses and oriented boxes in both the left IR and RGB frames. The Train and held-out Eval archives include 2,460 per-camera pose and oriented-box JSON files. The [shared mesh catalog](meshes/) contains 24 STL files, one per object shape represented in those splits. The bottle meshes include the rigid container, cap, and label. The simulated liquid remains part of the images.


### Pair images with the shared meshes

Extract the Zenodo Train or Eval archive into your data directory. Copy [the `meshes/` directory](meshes/) from this repository into the same directory. Train downloads are organized by camera. The held-out Eval archive contains all three cameras, with matching scene IDs across camera views.

```text
random-dot-stereo-data/
├── meshes/
│   ├── catalog.json
│   └── *.stl
├── sample/
│   ├── D435/
│   │   ├── calibration.json
│   │   └── e00000/
│   ├── D555/
│   └── OAK-D-active/
├── train/
│   ├── D435/
│   │   ├── calibration.json
│   │   └── 00100/
│   │       ├── rgb.png
│   │       ├── ir_left.png
│   │       ├── ir_right.png
│   │       ├── disparity_left.pfm
│   │       ├── disparity_rgb.pfm
│   │       ├── segmentation_rgb.png
│   │       ├── objects_3d.json
│   │       └── cables_3d.json
│   ├── D555/
│   └── OAK-D-active/
└── eval/
    ├── D435/
    ├── D555/
    └── OAK-D-active/
```

Each camera folder keeps its shared `calibration.json` at the top level. The scene folders sit directly alongside `calibration.json` and are named by scene ID. This applies to `sample/`, `train/`, and `eval/`. The disparity validity, stereo visibility, and transparent material masks are also in each scene folder. Scenes with cables have a `cables_3d.json` containing cable instance IDs and OBBs in the left IR and RGB frames. Cable labels remain in the RGB mask.

Paths in `objects_3d.json` are relative to that scene directory. Use `mesh_id` to identify the shared geometry. Given a vertex `p_stl` in meters, compute `p_rgb = R @ (scale_xyz * p_stl) + t`, with `R` and `t` taken from `T_rgb_from_mesh`. The OBBs are already expressed in their respective camera frames.

```python
import json
from pathlib import Path

import numpy as np

scene = Path("sample/D435/e00000")
annotation = json.loads((scene / "objects_3d.json").read_text(encoding="utf-8"))
instance = annotation["instances"][0]

rgb_path = scene / annotation["rgb_image"]
mask_path = scene / annotation["rgb_instance_mask"]
stl_path = (scene / instance["mesh_stl"]).resolve()
instance_id = instance["instance_id"]
scale = np.asarray(instance["scale_xyz"])
pose = np.asarray(instance["T_rgb_from_mesh"])

# vertices_stl is an [N, 3] array read from stl_path, in meters.
def to_rgb_camera(vertices_stl):
    return (vertices_stl * scale) @ pose[:3, :3].T + pose[:3, 3]
```


## 🍺

If this dataset was useful to you and we ever meet, feel free to buy me a beer.
