# Schwarzschild GR Ray Tracer

An interactive, real-time black hole visualizer. The renderer traces Schwarzschild null geodesics through GPU lookup tables, so the lensing of the star field and of the accretion disk follows the actual ray paths rather than a screen-space distortion.

The project is an experimental physics visualizer. It is Schwarzschild only (`G = c = M = 1`) and needs a CUDA-capable NVIDIA GPU through CuPy.

## Features

- Mouse and keyboard orbit camera, with mouse-wheel field of view control
- Star field at infinity, lensed by the hole, with an optional 15° coordinate grid
- Lensed accretion disk with primary, secondary, and higher-order images
- Novikov–Thorne disk flux, blackbody colours through CIE 1931 → sRGB, exact Doppler and gravitational frequency shift, limb darkening, and reversible disk rotation
- Blackbody (default), grayscale and hand-painted "classic" colour modes, plus image-order and `g`-factor debug maps
- Linear-light shading with optional supersampling (SSAA)
- Headless snapshots, a GPU benchmark, CPU physics checks, and GPU-versus-CPU parity tests

## Requirements and launch

- Python 3.10 or newer (developed and tested on Windows 11 with Python 3.13)
- NVIDIA GPU with a working CUDA 12 driver
- NumPy, pygame-ce, and CuPy for CUDA 12.x

From this repository's root directory:

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 gr_raytracer.py
```

CUDA packages and drivers must be compatible. If Windows blocks a CuPy or NVIDIA DLL, inspect the Windows security notification and the affected installed package before retrying. The first launch compiles the CUDA kernels and can take noticeably longer than later ones.

## Controls

| Input | Action |
| --- | --- |
| Left mouse drag, `W`/`S`, `Q`/`E`, arrow keys | Orbit camera |
| Mouse wheel | Change field of view |
| `C` | Cycle blackbody, classic, and grayscale colour modes |
| `-` / `=` | Decrease / increase exposure |
| `,` / `.` | Decrease / increase the peak disk temperature (blackbody mode) |
| `J` / `V` / `L` | Toggle frequency shifts / reverse disk rotation / toggle limb darkening |
| `H`, `Z`/`X` | Toggle finite disk thickness; decrease/increase its opening angle |
| `D` / `G` | Toggle disk / sky grid |
| `K` / `O` | Show frequency-shift map / image-order colours |
| `P` | Limit disk images to primary, primary + secondary, or all |
| `A` | Cycle supersampling 1× → 2× → 3× |
| `F1`–`F4` | Set 640 × 360, 960 × 540, 1280 × 720, or 1920 × 1080 render resolution |
| `F12` | Save the current frame as `screenshot_<date>_<time>.png` |
| `Tab` | Show / hide the overlay |
| `R` / `Esc` | Reset view / exit |

A still scene is not re-traced: the GPU only works while something changes.

## Physics model

**Geodesics.** The radial null path is integrated as `u''(φ) = 3u² − u`, with `u = 1/r`, using RK4 in double precision. The event horizon is at `r = 2M`, the photon sphere at `r = 3M`, and the critical impact parameter is `3√3 M`. The camera is a static observer at `30M` by default; a ray leaving it at local angle `θ` from the radial direction has impact parameter `b = r sin θ / √(1 − 2/r)`. Two tables are built once and interpolated per pixel: the escape angle of every sky ray, and `u(φ)` along every ray for disk intersections.

**Sky.** Stars are at infinity: a ray is followed until `u = 0` and its asymptotic direction is looked up in a procedural star texture (uniform over the sphere, power-law brightness distribution, blackbody star colours, a tilted Milky Way band). `--background-r` places the star sphere at a finite radius instead.

**Disk.** The disk spans `6M` (the ISCO) to `20M` and is geometrically an opaque equatorial wedge with a continuous first-surface intersection, including its outer rim. The local flux is the Novikov–Thorne (Page & Thorne 1974) profile for a Schwarzschild hole with zero torque at the inner edge,

```text
F(r) ∝ [√r − √r_in + (√3/2) ln((√r + √3)(√r_in − √3) / ((√r − √3)(√r_in + √3)))] / (r^(5/2) (r − 3))
```

which vanishes at the ISCO, peaks at `9.55M`, and tends to the Newtonian `r⁻³` law far away. The gas moves on circular geodesics, so the frequency-shift factor seen by the static camera is

```text
g = ν_obs / ν_emit = √(1 − 3/r) / [√(1 − 2/r_camera) (1 − Ω λ)],    Ω = ± r^(−3/2)
```

Here `λ` is the photon's conserved axial angular momentum divided by its energy.

**Colour.** The surface radiates as a blackbody with `T(r) = T_max [F(r)/F_max]^(1/4)`. Because `I_ν/ν³` is invariant, the observed spectrum is exactly a blackbody at `g·T`; that spectrum is integrated against the CIE 1931 colour matching functions and converted to linear sRGB, which carries both the colour shift and the Doppler/gravitational brightening without a separate `g⁴` factor. The grayscale and classic modes show the bolometric intensity `F g⁴` instead. All modes apply electron-scattering limb darkening `I(μ) ∝ 1 + 2.06 μ`, using the emission angle in the gas rest frame, then an exponential exposure curve and the sRGB transfer function.

`T_max` defaults to 4500 K so that the shifts stay inside the visible range. Real disks are far hotter (10⁵–10⁷ K) and would look uniformly blue-white to the eye; raise it with `--tmax` or `.` to see that.

**Limits.** Kerr spin and frame dragging, emission from gas plunging inside the ISCO, light returning to the disk, and volumetric radiative transfer through the disk are outside the model. The disk interior is not resolved: thickness only changes the shape of the opaque surface. A razor-thin disk viewed exactly edge-on is geometrically degenerate and invisible. The camera must stay outside the disk (`--camera-r` > `--disk-rout`).

## Validation and performance

The CPU checks run without opening a window or initializing CUDA:

```powershell
py -3.13 gr_raytracer.py --validate
py -3.13 -m unittest -v test_physics.py
```

They compare the integrator with an independent quadrature of the orbit integral, the analytic periapsis and weak-field deflection, the Novikov–Thorne closed form with a numerical Page–Thorne integral, and blackbody chromaticities with the Planckian locus. The GPU tests render real frames and compare pixels with values predicted by the CPU reference (table contents, shadow size, disk brightness and colour, mirror symmetry under spin reversal, rim continuity at edge-on); they are skipped when no CUDA device is available:

```powershell
py -3.13 -m unittest -v test_gpu.py
```

Measure GPU rendering and host transfer on your machine:

```powershell
py -3.13 gr_raytracer.py --benchmark-frames 60
```

Render a still frame (snapshots default to 3 × 3 supersampling; every view option of the interactive viewer has a command-line flag, see `--help`):

```powershell
py -3.13 gr_raytracer.py --render-width 1280 --render-height 720 --start-elevation 4 --fov 40 --no-grid --snapshot edge-on.png
```

## Files

| File | Contents |
| --- | --- |
| `gr_raytracer.py` | CUDA kernels, renderer, command line, interactive viewer |
| `gr_physics.py` | NumPy reference physics, lookup-table builders, sky generator |
| `test_physics.py` | CPU-only tests |
| `test_gpu.py` | GPU-versus-CPU parity tests |
