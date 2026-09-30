# scroll_surface

A deterministic 3D surface-extraction experiment for the [Vesuvius Challenge – Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/).

The task is to identify thin papyrus surfaces inside reconstructed X-ray CT volumes of carbonized scrolls. Those surfaces are a prerequisite for virtual unwrapping: before text can be flattened and read, the geometry of the sheet has to be found inside a dense 3D voxel volume.

This repository explores that problem without a trained segmentation model. Instead, it treats surface detection as an explicit signal-processing and voxel-geometry pipeline whose intermediate states can be inspected and modified.

## Problem

Input is a 3D TIFF volume:

```
volume[z, y, x] -> voxel intensity
```

The useful structure is not simply "all bright voxels". CT volumes contain background variation, noise, weak material responses and connected structures that are not necessarily part of the target surface.

The experiment therefore separates the problem into four stages:

```
3D TIFF volume
    |
    v
[ scanner ]
candidate foreground voxels
    |
    v
[ cleaner ]
locally supported voxels
    |
    v
[ surface extraction ]
boundary voxels of the remaining structure
    |
    v
[ label output ]
3D TIFF segmentation mask
```

PLY is used between stages as an inspectable geometric representation rather than hiding every transformation inside one segmentation pass.

## Approach

### 1. Scanner — intensity -> candidate geometry

`modules/scanner.py` loads the source volume and turns likely foreground voxels into a 3D point cloud.

The scanner supports several deliberately independent mechanisms:

- **Local background removal** — each 2D slice can be divided into tiles, a local background estimate is derived from the tile intensity distribution, the resulting field is smoothed and bilinearly interpolated, then subtracted from the source.
- **Fixed thresholding** — useful when intensity characteristics are known and stable.
- **Quantile thresholding** — derives the foreground threshold from the sampled intensity distribution.
- **MAD thresholding** — uses median absolute deviation as a robust estimate of signal spread.
- **Hysteresis thresholding** — strong voxels seed the mask and connected weaker voxels can be retained using configurable 4- or 8-neighbour connectivity.
- **6-neighbour support gating** — isolated candidate voxels can be rejected based on local 3D support.
- **Subsampling** — `STEP` controls how densely the candidate volume is exported.

The result is written as ASCII PLY containing XYZ coordinates, RGBA channels and the source intensity.

Optional debug PLYs and scanner statistics make the thresholding/background stages observable instead of treating them as a black box.

### 2. Cleaner — reject weak local structure

`modules/cleaner.py` operates on the candidate point cloud.

For each voxel it checks the six direct 3D neighbours:

```
+x  -x
+y  -y
+z  -z
```

A point is retained only when its direct-neighbour count is greater than `VCOUNT`.

This is a simple structural filter: the decision is based on local 3D connectivity rather than another image-space blur or a learned classifier.

Output:

```
ply/<id>.ply
    ->
ply_clean/<id>.ply
```

### 3. Surface extraction — volume -> boundary

`modules/find_outlines.py` converts the cleaned voxel structure into a surface representation.

A voxel is considered a surface voxel when **at least one of its six direct neighbours is missing**.

In other words, the stage does not try to recognize a surface visually. It derives the boundary directly from occupancy topology:

```
all 6 neighbours present  -> interior voxel
one or more missing       -> surface voxel
```

Output is written to:

```
surf/<id>.ply
```

### 4. Label output — geometry -> segmentation volume

`modules/label_output.py` rasterizes the extracted surface coordinates back into the original `(Z, Y, X)` volume shape and writes the final TIFF mask.

An optional `BAND` parameter can expand the extracted surface before export.

Output:

```
out_entry/<id>.tif
```

## Why this experiment is interesting

Most high-performing 3D segmentation systems approach this kind of problem with trained neural networks and substantial post-processing.

This repository explores a different question:

> How far can the useful structure be recovered by making the assumptions explicit?

The implementation exposes those assumptions as individual operations: background estimation, threshold selection, connectivity, neighbourhood support, boundary extraction and rasterization.

That makes failure modes inspectable. If an output is wrong, the pipeline provides concrete intermediate representations for determining *where* the structure disappeared or where noise entered it.

It also keeps signal interpretation and geometry processing separate:

```
intensity
   -> classification of candidate voxels
   -> structural filtering
   -> topology / boundary extraction
   -> output mask
```

There is no model-training loop and no model weights in this repository.

## Repository structure

```
scroll_surface/
├── main.py
├── config.py
├── clean.py
├── modules/
│   ├── scanner.py
│   ├── cleaner.py
│   ├── find_outlines.py
│   └── label_output.py
├── ply/
├── ply_clean/
├── surf/
└── out_entry/
```

`main.py` runs the full pipeline in order:

```
scanner -> cleaner -> find_outlines -> label_output
```

Each stage can also be run independently.

## Requirements

The code uses:

- Python 3
- NumPy
- tifffile

Install the Python dependencies with your preferred environment manager, for example:

```bash
pip install numpy tifffile
```

## Dataset layout

Set `DATA_DIR` and `SPLIT` in `config.py`.

The expected source layout is:

```
<DATA_DIR>/
├── train_images/
│   ├── <id>.tif
│   └── ...
└── test_images/
    ├── <id>.tif
    └── ...
```

The volume is read as:

```python
(Z, Y, X)
```

## Running

Process one volume:

```bash
python main.py --id <image_id>
```

Process every TIFF in the configured split:

```bash
python main.py --all
```

Individual stages can also be called directly:

```bash
python modules/scanner.py --id <image_id>
python modules/cleaner.py --id <image_id>
python modules/find_outlines.py --id <image_id>
python modules/label_output.py --id <image_id>
```

Generated outputs can be cleared with:

```bash
python clean.py
```

or inspected first:

```bash
python clean.py --dry-run
```

## Configuration

The most relevant controls are in `config.py`:

| Setting | Purpose |
| --- | --- |
| `DATA_DIR` | Dataset root |
| `SPLIT` | `train` or `test` |
| `STEP` | Scanner sampling density |
| `BG_ENABLE` | Enable local background subtraction |
| `BG_TILE` | Background estimation tile size |
| `MASK_MODE` | Single threshold or hysteresis |
| `THRESH_MODE` | Fixed, quantile or MAD threshold |
| `HYST_LOW_RATIO` | Low/high relation for hysteresis |
| `FG_SUPPORT_N` | Required direct-neighbour support |
| `VCOUNT` | Cleaner neighbour-count threshold |
| `BAND` | Optional output surface expansion |

## Scope

This repository is published as a code artifact from an exploratory engineering solution, not as a claim of state-of-the-art competition performance or a general-purpose segmentation library.

Its purpose is to preserve the actual problem-solving approach: decompose a difficult 3D perception problem into observable stages, keep the geometry explicit, and make each assumption independently testable.
