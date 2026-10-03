# Schwarzschild GR Ray Tracer v0.6.2 — Stabilized Hybrid Thick Disk

This version deliberately backs away from the failed **screen-space center fallback** used in v0.6.1.

That fallback caused the obvious problem you noticed:
- a fixed black rectangle / bar across the center
- wrong zoom behavior
- visibly artificial geometry

So v0.6.2 does the sensible thing instead:

## Main change
It keeps the **v0.6 relativistic color / order / g-map features**, but returns to a **finite-thickness visual disk intersection** similar to the older stable behavior.

In plain English:
- no more screen-space black bar hack
- finite-thickness intersections are solved continuously in geodesic angle instead of choosing one of three layers
- exactly edge-on rays that lie inside the finite wedge hit the cached outer radial face in world space
- zoom behavior should stay physically tied to the image, not the screen

## What the thickness model is

The finite-thickness mode is an **optically thick equatorial wedge with first-surface hits**.
It is continuous geometry, so it no longer renders three sampled copies of the thin disk.
It is still not volumetric radiative transfer: there is no density integration, absorption,
or emission accumulated along a ray. Razor-thin rays exactly coplanar with the disk remain
mathematically degenerate and are deliberately not covered by a screen-space fallback.

## Color modes (`C`)
- `GRAY`
- `CLASSIC` (default; warm and toned down)
- `SPECTRAL`

## Thickness controls
- `H` → thickness on/off
- `Z` / `X` → decrease / increase disk thickness
- also works with `[` `]`, `,` `.` and `-` `=`

The displayed `H/R` value is based on `disk_half_opening_deg`.

## Physics and approximation boundary

Implemented from the Schwarzschild model:

- null paths satisfy `u'' = 3u^2 - u` in `G = c = M = 1` units
- the shadow boundary uses `b_crit = 3 sqrt(3) M`
- the disk frequency shift uses the circular-geodesic `Omega`, `u^t`, and conserved axial `lambda`
- image order is the successive equatorial crossing index, not screen position

Numerical or visual approximations:

- angular paths are RK4-integrated into GPU lookup tables and interpolated per pixel
- finite thickness is an opaque wedge first-hit surface, not radiative transfer
- emissivity is the normalized thin-disk-like proxy `r^-3 [1 - sqrt(r_in/r)]`
- CLASSIC deliberately uses a restrained `g^2.6` brightness boost and subtle tint
- SPECTRAL uses three sampled wavelengths and `g^4`; it is a diagnostic visualization, not a full color pipeline

## Other controls
- `J` → relativity on/off
- `V` → reverse rotation
- `K` → g-factor debug map
- `O` → image-order color debug
- `P` → PRIMARY / PRIMARY+SECONDARY / ALL
- `D` → disk on/off
- `G` → grid on/off
- `R` → reset camera

## Run
```powershell
py -3.13 gpu_disk_viewer_v062.py
```

## Validate without CUDA or pygame

```powershell
py -3.13 gpu_disk_viewer_v062.py --validate
py -3.13 -m unittest -v test_physics.py
```

The checks cover the critical capture/escape split, the null-orbit first integral,
step-size convergence, the analytical emissivity peak, camera basis, and Doppler
symmetry under rotation reversal.

## Repeatable GPU benchmark

This renders off-screen and does not initialize a pygame window:

```powershell
py -3.13 gpu_disk_viewer_v062.py --benchmark-frames 60
```

For a reproducible still (useful for edge-on regression comparisons):

```powershell
py -3.13 gpu_disk_viewer_v062.py --render-width 960 --render-height 540 --start-elevation 0 --snapshot edge-on.png
```

## Honest note
This is not a full volumetric disk solution.
It is a pragmatic rollback from a bad hack.

If the center still looks imperfect, that means the real remaining issue is the razor-thin / near-edge-on geometry itself, not something that should be "painted over" with a screen-space patch.
