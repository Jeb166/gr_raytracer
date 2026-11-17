"""CPU reference physics for the Schwarzschild ray tracer (G = c = M = 1).

Everything in this module is plain NumPy so it can be imported and tested
without CUDA or pygame.  The GPU renderer does not re-derive these formulas:
the disk flux profile, the blackbody colour table and the sRGB decode table
are computed here and uploaded as lookup tables, so the unit tests exercise
the numbers the kernel actually uses.
"""
from __future__ import annotations

import functools
import math

import numpy as np

HORIZON_R = 2.0
PHOTON_SPHERE_R = 3.0
ISCO_R = 6.0
B_CRIT = 3.0 * math.sqrt(3.0)

# Chandrasekhar's electron-scattering atmosphere, I(mu) ~ 1 + 2.06 mu.
LIMB_DARKENING_COEFF = 2.06


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------

def camera_basis(yaw: float, pitch: float):
    """Orbit-camera frame: outward radial unit vector, forward, right, up."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    er = np.array([sy * cp, sp, cy * cp], float)
    forward = -er
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    if np.linalg.norm(right) < 1e-10:
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return er, forward, right, up


# --------------------------------------------------------------------------
# Null geodesics
# --------------------------------------------------------------------------

def impact_parameter(camera_radius: float, mu: float) -> float:
    """Impact parameter of a ray seen by a static observer.

    ``mu`` is the cosine between the ray direction and the outward radial
    direction in the observer's local frame.
    """
    sin_theta = math.sqrt(max(0.0, 1.0 - mu * mu))
    return camera_radius * sin_theta / math.sqrt(1.0 - 2.0 / camera_radius)


def periapsis_radius(impact: float) -> float:
    """Closest approach of a scattering null geodesic (largest root of
    r^3 - b^2 r + 2 b^2 = 0)."""
    if impact <= B_CRIT:
        raise ValueError("rays with b <= 3*sqrt(3) are captured and have no periapsis")
    return 2.0 * impact / math.sqrt(3.0) * math.cos(math.acos(-B_CRIT / impact) / 3.0)


def angle_to_periapsis(impact: float, radius: float, nodes: int = 96) -> float:
    """Orbital angle between ``radius`` (may be inf) and periapsis: the
    integral of du / sqrt(1/b^2 - u^2 + 2u^3) from u = 1/radius to u_p.

    The inverse-square-root endpoint singularity is removed with the
    substitution u = u_p - (u_p - u_from) t^2 before Gauss-Legendre quadrature.
    """
    u_from = 0.0 if math.isinf(radius) else 1.0 / radius
    u_p = 1.0 / periapsis_radius(impact)
    # 2u^3 - u^2 + 1/b^2 = 2 (u - u_n)(u_p - u)(u_3 - u) with u_n < 0 < u_p < u_3.
    root_sum = 0.5 - u_p
    root_prod = -1.0 / (2.0 * impact * impact * u_p)
    disc = math.sqrt(root_sum * root_sum - 4.0 * root_prod)
    u_n, u_3 = 0.5 * (root_sum - disc), 0.5 * (root_sum + disc)
    x, w = np.polynomial.legendre.leggauss(nodes)
    t, w = 0.5 * (x + 1.0), 0.5 * w
    span = u_p - u_from
    u = u_p - span * t * t
    return float(np.sum(w * 2.0 * math.sqrt(span) / np.sqrt(2.0 * (u - u_n) * (u_3 - u))))


def exit_angle_quadrature(
    impact: float,
    camera_radius: float = 30.0,
    background_radius: float = math.inf,
) -> float:
    """Orbital angle swept from the camera, through periapsis, out to the
    background sphere (or to infinity).  Independent of the RK4 integrator."""
    return angle_to_periapsis(impact, camera_radius) + angle_to_periapsis(impact, background_radius)


def deflection_angle(impact: float) -> float:
    """Total light-bending angle for a ray coming from and escaping to infinity."""
    return 2.0 * angle_to_periapsis(impact, math.inf) - math.pi


def integrate_null_geodesic(
    impact_parameter: float,
    camera_radius: float = 30.0,
    background_radius: float = math.inf,
    step: float = 0.001,
    max_phi: float = 40.0 * math.pi,
) -> dict[str, float | bool]:
    """Reference RK4 integration of u'' = 3u^2 - u (u = 1/r) for an inward ray.

    Mirrors the GPU table builder: the escape and capture crossings are
    interpolated inside the final step instead of being rounded to a step
    count.  ``background_radius=inf`` returns the asymptotic direction.
    """
    if impact_parameter <= 0.0 or camera_radius <= HORIZON_R or background_radius <= camera_radius:
        raise ValueError("invalid Schwarzschild geodesic parameters")
    u = 1.0 / camera_radius
    invariant = 1.0 / impact_parameter**2
    radial_term = invariant - u * u + 2.0 * u**3
    if radial_term < 0.0:
        raise ValueError("impact parameter does not define an inward local ray at the camera")
    v = math.sqrt(radial_term)
    u_background = 0.0 if math.isinf(background_radius) else 1.0 / background_radius
    u_horizon = 1.0 / HORIZON_R
    max_u = u
    max_invariant_error = 0.0
    steps = int(max_phi / step)

    def result(escaped: bool, phi: float) -> dict[str, float | bool]:
        return {"escaped": escaped, "phi": phi, "r_min": 1.0 / max_u,
                "invariant_error": max_invariant_error}

    for i in range(steps):
        k1u, k1v = v, 3.0 * u * u - u
        u2, v2 = u + 0.5 * step * k1u, v + 0.5 * step * k1v
        k2u, k2v = v2, 3.0 * u2 * u2 - u2
        u3, v3 = u + 0.5 * step * k2u, v + 0.5 * step * k2v
        k3u, k3v = v3, 3.0 * u3 * u3 - u3
        u4, v4 = u + step * k3u, v + step * k3v
        k4u, k4v = v4, 3.0 * u4 * u4 - u4
        new_u = u + (step / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        new_v = v + (step / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

        if v > 0.0 >= new_v:
            # Turning point inside this step: u rises by v^2 / (2|u''|).
            max_u = max(max_u, u + v * v / (2.0 * (u - 3.0 * u * u)))
        max_u = max(max_u, min(new_u, u_horizon))
        current = new_v * new_v + new_u * new_u - 2.0 * new_u**3
        max_invariant_error = max(max_invariant_error, abs(current - invariant))

        if new_u >= u_horizon:
            fraction = (u_horizon - u) / (new_u - u)
            return result(False, (i + fraction) * step)
        if new_u <= u_background and new_v < 0.0:
            fraction = (u - u_background) / (u - new_u)
            return result(True, (i + fraction) * step)
        u, v = new_u, new_v

    return result(False, steps * step)


def radius_at_angle(
    impact_parameter: float,
    phi: float,
    camera_radius: float = 30.0,
    step: float = 0.001,
) -> float | None:
    """Radius of an inward ray after sweeping orbital angle ``phi`` from the
    camera, or ``None`` if it has already been captured or has escaped."""
    u = 1.0 / camera_radius
    v = math.sqrt(max(0.0, 1.0 / impact_parameter**2 - u * u + 2.0 * u**3))
    whole = int(phi / step)
    for i in range(whole + 1):
        h = step if i < whole else phi - whole * step
        k1u, k1v = v, 3.0 * u * u - u
        u2, v2 = u + 0.5 * h * k1u, v + 0.5 * h * k1v
        k2u, k2v = v2, 3.0 * u2 * u2 - u2
        u3, v3 = u + 0.5 * h * k2u, v + 0.5 * h * k2v
        k3u, k3v = v3, 3.0 * u3 * u3 - u3
        u4, v4 = u + h * k3u, v + h * k3v
        k4u, k4v = v4, 3.0 * u4 * u4 - u4
        u += (h / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        v += (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        if u >= 1.0 / HORIZON_R or u <= 0.0:
            return None
    return 1.0 / u


# --------------------------------------------------------------------------
# Thin-disk emission
# --------------------------------------------------------------------------

def novikov_thorne_flux(radius, inner_radius: float = ISCO_R):
    """Page-Thorne (1974) time-averaged flux of a Schwarzschild thin disk.

    Returned per unit accretion rate (G = c = M = 1); zero torque is imposed
    at ``inner_radius`` (>= ISCO).  Tends to the Newtonian 3/(8 pi r^3) at
    large radius.
    """
    if inner_radius < ISCO_R:
        raise ValueError("circular geodesic orbits are unstable inside the ISCO (6M)")
    r = np.asarray(radius, dtype=float)
    x = np.sqrt(np.maximum(r, inner_radius))
    x0 = math.sqrt(inner_radius)
    s3 = math.sqrt(3.0)
    bracket = x - x0 + 0.5 * s3 * np.log((x + s3) * (x0 - s3) / ((x - s3) * (x0 + s3)))
    flux = 3.0 / (8.0 * math.pi) * bracket / (x**5 * (x * x - 3.0))
    flux = np.where(r > inner_radius, flux, 0.0)
    return float(flux) if flux.ndim == 0 else flux


@functools.lru_cache(maxsize=16)
def flux_peak(inner_radius: float = ISCO_R) -> tuple[float, float]:
    """Radius and value of the Novikov-Thorne flux maximum."""
    lo, hi = inner_radius, 4.0 * inner_radius
    grid = np.linspace(lo, hi, 4097)
    i = int(np.argmax(novikov_thorne_flux(grid, inner_radius)))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(80):
        a, b = hi - inv_phi * (hi - lo), lo + inv_phi * (hi - lo)
        if novikov_thorne_flux(a, inner_radius) > novikov_thorne_flux(b, inner_radius):
            hi = b
        else:
            lo = a
    r_peak = 0.5 * (lo + hi)
    return r_peak, float(novikov_thorne_flux(r_peak, inner_radius))


def disk_flux_profile(radius, inner_radius: float = ISCO_R):
    """Novikov-Thorne flux normalised to 1 at its maximum."""
    return novikov_thorne_flux(radius, inner_radius) / flux_peak(inner_radius)[1]


def build_flux_lut(inner_radius: float, outer_radius: float, samples: int = 2048) -> np.ndarray:
    """Normalised flux on a uniform radial grid for the GPU kernel."""
    radii = np.linspace(inner_radius, outer_radius, samples)
    return np.asarray(disk_flux_profile(radii, inner_radius), dtype=np.float32)


def schwarzschild_g_factor(
    radius: float,
    lambda_axis: float,
    camera_radius: float = 30.0,
    rotation_sign: float = 1.0,
) -> float:
    """nu_observed / nu_emitted for a circular equatorial emitter and a
    static observer.  ``lambda_axis`` is the photon's axial angular momentum
    per unit energy."""
    omega = math.copysign(radius**-1.5, rotation_sign)
    numerator = math.sqrt(1.0 - 3.0 / radius)
    denominator = math.sqrt(1.0 - 2.0 / camera_radius) * (1.0 - omega * lambda_axis)
    return numerator / denominator


def emission_cosine(
    radius: float,
    impact: float,
    lambda_axis: float,
    rotation_sign: float = 1.0,
) -> float:
    """Cosine between the photon and the disk normal in the gas rest frame.

    cos = |p_theta| / (r nu_emit) with p_theta^2 = b^2 - lambda^2 at the
    equator and nu_emit = u^t (1 - Omega lambda) for unit energy at infinity.
    """
    omega = math.copysign(radius**-1.5, rotation_sign)
    nu_emit = (1.0 - omega * lambda_axis) / math.sqrt(1.0 - 3.0 / radius)
    polar = math.sqrt(max(0.0, impact * impact - lambda_axis * lambda_axis))
    return min(1.0, polar / (radius * nu_emit))


def rim_emission_cosine(
    radius: float,
    impact: float,
    lambda_axis: float,
    rotation_sign: float = 1.0,
) -> float:
    """Same as :func:`emission_cosine` for a surface whose normal is radial
    (the outer rim of the disk wedge): the photon's local radial momentum is
    sqrt(1 - f b^2/r^2)/sqrt(f) with f = 1 - 2/r."""
    omega = math.copysign(radius**-1.5, rotation_sign)
    nu_emit = (1.0 - omega * lambda_axis) / math.sqrt(1.0 - 3.0 / radius)
    f = 1.0 - 2.0 / radius
    radial = math.sqrt(max(0.0, 1.0 - f * impact * impact / (radius * radius)) / f)
    return min(1.0, radial / nu_emit)


def limb_darkening(cosine: float) -> float:
    """Flux-preserving electron-scattering limb darkening factor."""
    k = LIMB_DARKENING_COEFF
    return (1.0 + k * cosine) / (1.0 + 2.0 * k / 3.0)


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------

PLANCK_C2_NM_K = 1.438776877e7  # h c / k_B in nm K

# CIE XYZ (D65) -> linear sRGB / Rec.709 primaries.
XYZ_TO_LINEAR_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def cie_1931_cmf(wavelength_nm):
    """CIE 1931 2-degree colour matching functions, using the multi-lobe
    Gaussian fit of Wyman, Sloan & Shirley (JCGT 2013)."""
    wl = np.asarray(wavelength_nm, dtype=float)

    def lobe(mu, sigma_lo, sigma_hi):
        sigma = np.where(wl < mu, sigma_lo, sigma_hi)
        return np.exp(-0.5 * ((wl - mu) / sigma) ** 2)

    x = 1.056 * lobe(599.8, 37.9, 31.0) + 0.362 * lobe(442.0, 16.0, 26.7) - 0.065 * lobe(501.1, 20.4, 26.2)
    y = 0.821 * lobe(568.8, 46.9, 40.5) + 0.286 * lobe(530.9, 16.3, 31.1)
    z = 1.217 * lobe(437.0, 11.8, 36.0) + 0.681 * lobe(459.0, 26.0, 13.8)
    return x, y, z


def blackbody_xyz(temperature) -> np.ndarray:
    """CIE XYZ of blackbody radiance B_lambda(T), in fixed arbitrary units."""
    temps = np.atleast_1d(np.asarray(temperature, dtype=float))
    wl = np.arange(360.0, 830.5, 1.0)
    x, y, z = cie_1931_cmf(wl)
    exponent = PLANCK_C2_NM_K / (wl[None, :] * temps[:, None])
    radiance = (wl[None, :] / 555.0) ** -5.0 / np.expm1(exponent)
    return np.stack([radiance @ x, radiance @ y, radiance @ z], axis=-1)


def blackbody_chromaticity(temperature: float) -> tuple[float, float]:
    xyz = blackbody_xyz(temperature)[0]
    total = float(xyz.sum())
    return float(xyz[0]) / total, float(xyz[1]) / total


def blackbody_linear_srgb(temperature) -> np.ndarray:
    """Linear sRGB radiance of a blackbody with out-of-gamut components
    clipped to zero.  Same arbitrary units for every temperature."""
    rgb = blackbody_xyz(temperature) @ XYZ_TO_LINEAR_SRGB.T
    return np.maximum(rgb, 0.0)


BLACKBODY_LUT_T_MIN = 300.0
BLACKBODY_LUT_T_MAX = 3.0e5


def build_blackbody_lut(samples: int = 2048):
    """Table over log-temperature of (r/Y, g/Y, b/Y, ln Y) in linear sRGB.

    Luminance spans dozens of decades across the table, so it is stored as a
    logarithm and interpolated in log space.  Returns ``(table, ln_t_min,
    d_ln_t)``.
    """
    ln_t = np.linspace(math.log(BLACKBODY_LUT_T_MIN), math.log(BLACKBODY_LUT_T_MAX), samples)
    rgb = blackbody_linear_srgb(np.exp(ln_t))
    luminance = rgb @ np.array([0.2126729, 0.7151522, 0.0721750])
    table = np.empty((samples, 4), dtype=np.float32)
    table[:, :3] = rgb / luminance[:, None]
    table[:, 3] = np.log(luminance)
    return table, float(ln_t[0]), float(ln_t[1] - ln_t[0])


def blackbody_log_luminance(temperature: float) -> float:
    rgb = blackbody_linear_srgb(temperature)[0]
    return math.log(float(rgb @ np.array([0.2126729, 0.7151522, 0.0721750])))


def srgb_encode(linear):
    """Linear light -> sRGB transfer function (IEC 61966-2-1)."""
    c = np.clip(np.asarray(linear, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def srgb_decode(encoded):
    c = np.clip(np.asarray(encoded, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


def srgb_decode_lut() -> np.ndarray:
    """256-entry table turning 8-bit sRGB texels into linear light."""
    return np.asarray(srgb_decode(np.arange(256) / 255.0), dtype=np.float32)


# --------------------------------------------------------------------------
# Sky
# --------------------------------------------------------------------------

def _galactic_frame() -> np.ndarray:
    """Rows are the galactic x/y/z axes in world coordinates; the galactic
    plane is tilted away from the disk plane so the two do not hide each
    other."""
    tilt, twist = math.radians(62.0), math.radians(35.0)
    ct, st, cw, sw = math.cos(tilt), math.sin(tilt), math.cos(twist), math.sin(twist)
    rot_x = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]], float)
    rot_y = np.array([[cw, 0, sw], [0, 1, 0], [-sw, 0, cw]], float)
    return (rot_y @ rot_x).T


def _direction_to_lon_lat(d: np.ndarray):
    """World direction -> sky texture angles.  The texture pole is the disk
    axis (+y); longitude is measured from +z towards +x."""
    return np.arctan2(d[..., 0], d[..., 2]), np.arcsin(np.clip(d[..., 1], -1.0, 1.0))


def make_sky_texture(
    seed: int = 7,
    width: int = 4096,
    height: int = 2048,
    stars: int = 26000,
) -> np.ndarray:
    """Procedural equirectangular star field, returned as 8-bit sRGB.

    Stars are distributed uniformly over the sphere (plus a population
    concentrated towards a tilted galactic plane), follow a power-law
    brightness distribution, are tinted with blackbody colours, and are drawn
    with a point-spread function that compensates for the equirectangular
    stretch so they stay round on the sky.
    """
    rng = np.random.default_rng(seed)
    galaxy = _galactic_frame()
    image = np.zeros((height, width, 3), dtype=np.float32)

    # --- diffuse Milky Way glow, built at half resolution -------------------
    gh, gw = height // 2, width // 2
    lat = (0.5 - (np.arange(gh) + 0.5) / gh) * math.pi
    lon = ((np.arange(gw) + 0.5) / gw - 0.5) * 2.0 * math.pi
    cos_lat = np.cos(lat)[:, None]
    direction = np.stack([
        cos_lat * np.sin(lon)[None, :],
        np.broadcast_to(np.sin(lat)[:, None], (gh, gw)),
        cos_lat * np.cos(lon)[None, :],
    ], axis=-1).astype(np.float32)
    gal = direction @ galaxy.T.astype(np.float32)
    gal_lat = np.arcsin(np.clip(gal[..., 2], -1.0, 1.0))
    gal_lon = np.arctan2(gal[..., 1], gal[..., 0])

    def cloud_noise(octaves: int, base_frequency: float) -> np.ndarray:
        # Sum of plane waves on the unit sphere: seamless, no pole artefacts.
        total = np.zeros((gh, gw), dtype=np.float32)
        norm = 0.0
        for octave in range(octaves):
            frequency = base_frequency * 1.9**octave
            amplitude = 0.62**octave
            for _ in range(4):
                k = rng.normal(size=3)
                k *= frequency / np.linalg.norm(k)
                total += amplitude * np.cos(direction @ k.astype(np.float32) + rng.uniform(0, 2 * math.pi))
                norm += amplitude
        return 0.5 + 0.5 * total / (0.45 * norm)

    clouds = np.clip(cloud_noise(5, 3.0), 0.0, 1.5)
    dust = np.clip(cloud_noise(5, 5.0), 0.0, 1.5)
    band = np.exp(-(gal_lat / math.radians(11.0)) ** 2)
    bulge = 1.0 + 1.6 * np.exp(-(gal_lon / math.radians(38.0)) ** 2 - (gal_lat / math.radians(14.0)) ** 2)
    lane = 1.0 - 0.75 * np.exp(-(gal_lat / math.radians(2.6)) ** 2) * np.clip(dust * 1.2 - 0.25, 0.0, 1.0)
    glow = 0.030 * band * bulge * (0.35 + 0.65 * clouds) * lane + 0.0012
    glow = np.repeat(np.repeat(glow, 2, axis=0), 2, axis=1)[:height, :width]
    for axis, wrap in ((1, True), (0, False)):
        before = np.roll(glow, 1, axis=axis)
        after = np.roll(glow, -1, axis=axis)
        if not wrap:
            before[0], after[-1] = glow[0], glow[-1]
        glow = 0.25 * before + 0.5 * glow + 0.25 * after
    image += glow[..., None] * np.array([1.0, 0.93, 0.84], dtype=np.float32)

    # --- stars ----------------------------------------------------------------
    n_field = int(stars * 0.68)
    n_disk = stars - n_field
    z = rng.uniform(-1.0, 1.0, n_field)
    az = rng.uniform(-math.pi, math.pi, n_field)
    rho = np.sqrt(1.0 - z * z)
    field = np.stack([rho * np.cos(az), z, rho * np.sin(az)], axis=-1)
    g_lat = np.clip(rng.normal(0.0, math.radians(9.0), n_disk), -1.4, 1.4)
    g_lon = rng.uniform(-math.pi, math.pi, n_disk)
    in_plane = np.stack([np.cos(g_lat) * np.cos(g_lon), np.cos(g_lat) * np.sin(g_lon), np.sin(g_lat)], axis=-1)
    directions = np.concatenate([field, in_plane @ galaxy])
    star_lon, star_lat = _direction_to_lon_lat(directions)

    # log N(<m) ~ 0.45 m: many faint stars, a few hundred bright ones.
    magnitude_above_limit = -np.log10(rng.uniform(1.0 / stars, 1.0, stars)) / 0.45
    amplitude = 0.022 * 10.0 ** (0.4 * magnitude_above_limit)
    temperature = np.clip(np.exp(rng.normal(math.log(5800.0), 0.38, stars)), 2800.0, 26000.0)
    colour = blackbody_linear_srgb(temperature)
    colour /= colour.max(axis=1, keepdims=True)
    colour = 0.4 + 0.6 * colour

    # Stars brighter than full scale get a wider PSF instead of clipping.
    sigma = 0.62 * np.sqrt(np.clip(amplitude, 1.0, 5.0))
    peak = np.minimum(amplitude, 1.0)
    xc = (star_lon / (2.0 * math.pi) + 0.5) * width - 0.5
    yc = (0.5 - star_lat / math.pi) * height - 0.5
    xi, yi = np.rint(xc).astype(np.int64), np.rint(yc).astype(np.int64)
    cos_star = np.maximum(np.cos(star_lat), 0.1)
    # Footprint in texels: 3 sigma vertically, widened by 1/cos(lat)
    # horizontally.  Stars are grouped by footprint (powers of two) so the
    # many small ones do not pay for the few wide ones near the poles.
    reach_y = np.ceil(3.0 * sigma)
    reach_x = np.ceil(3.0 * sigma / cos_star)
    bucket_y = (2 ** np.ceil(np.log2(reach_y))).astype(np.int64)
    bucket_x = (2 ** np.ceil(np.log2(reach_x))).astype(np.int64)
    for by, bx in sorted({(int(a), int(b)) for a, b in zip(bucket_y, bucket_x)}):
        sel = np.flatnonzero((bucket_y == by) & (bucket_x == bx))
        s_xc, s_yc, s_xi, s_yi = xc[sel], yc[sel], xi[sel], yi[sel]
        s_cos, s_peak, s_var, s_colour = cos_star[sel], peak[sel], sigma[sel] ** 2, colour[sel]
        for dy in range(-by, by + 1):
            row = s_yi + dy
            row_ok = (row >= 0) & (row < height)
            ddy = row - s_yc
            for dx in range(-bx, bx + 1):
                ddx = (s_xi + dx - s_xc) * s_cos
                weight = s_peak * np.exp(-0.5 * (ddx * ddx + ddy * ddy) / s_var)
                hit = row_ok & (weight > 2.0e-3)
                if not hit.any():
                    continue
                np.add.at(image, (row[hit], (s_xi[hit] + dx) % width),
                          (weight[hit, None] * s_colour[hit]).astype(np.float32))

    return np.rint(srgb_encode(image) * 255.0).astype(np.uint8)
