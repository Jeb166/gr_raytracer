# Schwarzschild GR Ray Tracer

An interactive, real-time black hole visualizer for Windows. The renderer traces Schwarzschild null geodesics through GPU lookup tables, so the lensing of the fixed star field and accretion disk follows the ray paths rather than a screen-space distortion.

The project is an experimental physics visualizer. Its current implementation is Schwarzschild only (`G = c = M = 1`) and uses a CUDA-capable NVIDIA GPU through CuPy.

## Features

- Mouse and keyboard orbit camera, with mouse-wheel field of view control
- World-fixed spherical star field and optional coordinate grid
- Lensed accretion disk with primary, secondary, and higher-order images
- Schwarzschild circular-orbit frequency shift, Doppler brightness asymmetry, and reversible disk rotation
- Warm default color mode, grayscale mode, experimental spectral mode, image-order colors, and a `g`-factor map
- Default 1920 × 1080 rendering, lower-resolution presets, CPU physics checks, and an off-screen GPU benchmark

## Requirements and launch

- Windows 11 and Python 3.13
- NVIDIA GPU with a working CUDA driver
- NumPy, pygame-ce, and CuPy for CUDA 12.x

From this repository's root directory:

```powershell
py -3.13 -m pip install numpy pygame-ce "cupy-cuda12x[ctk]>=14.2"
py -3.13 gpu_disk_viewer_v062.py
```

CUDA packages and drivers must be compatible. If Windows blocks a CuPy or NVIDIA DLL, inspect the Windows security notification and the affected installed package before retrying.

## Controls

| Input | Action |
| --- | --- |
| Left mouse drag, `W`/`S`, `Q`/`E`, arrow keys | Orbit camera |
| Mouse wheel | Change field of view |
| `F1`–`F4` | Set 640 × 360, 960 × 540, 1280 × 720, or 1920 × 1080 render resolution |
| `D` / `G` | Toggle disk / sky grid |
| `H`, `Z`/`X` | Toggle finite disk thickness; decrease/increase its opening angle |
| `J` / `V` | Toggle relativistic transfer / reverse disk rotation |
| `C` | Cycle grayscale, warm classic, and spectral color modes |
| `K` / `O` | Show frequency-shift map / image-order colors |
| `P` | Limit disk images to primary, primary + secondary, or all |
| `R` / `Esc` | Reset camera / exit |

## Physics model and limits

The radial null path is integrated as `u''(φ) = 3u² − u`, with `u = 1/r`. The event horizon is at `r = 2M`, the photon sphere at `r = 3M`, and the critical impact parameter is `3√3 M`. GPU path tables are interpolated for each pixel. The default camera radius is `30M`, the background sphere is `100M`, and the disk spans `6M` to `20M`.

The frequency-shift factor for circular equatorial disk matter is

```text
g = sqrt(1 - 3/r) / [sqrt(1 - 2/r_camera) (1 - Ω λ)]
Ω = ± r^(-3/2)
```

Here `λ` is the photon's conserved axial angular momentum divided by its energy. The warm default mode uses this factor mainly for brightness, with restrained color changes. The spectral mode is a three-wavelength visualization, not a calibrated color pipeline.

The finite-thickness disk is an **opaque equatorial wedge with a continuous first-surface intersection**. It does not integrate emission or absorption through a volume. The local emissivity is the normalized proxy `F(r) ∝ r⁻³ [1 − √(r_in/r)]`, not a full Novikov–Thorne disk. An exactly coplanar ray and a razor-thin disk remain geometrically degenerate. Kerr spin, frame dragging, and true volumetric radiative transfer are outside the current model.

## Validation and performance

The CPU checks run without opening a window or initializing CUDA:

```powershell
py -3.13 gpu_disk_viewer_v062.py --validate
py -3.13 -m unittest -v test_physics.py
```

Measure GPU rendering and host transfer on your machine:

```powershell
py -3.13 gpu_disk_viewer_v062.py --benchmark-frames 60
```

Generate a still frame for an edge-on comparison:

```powershell
py -3.13 gpu_disk_viewer_v062.py --render-width 960 --render-height 540 --start-elevation 0 --snapshot edge-on.png
```
