"""CPU-only checks of the reference physics.  No CUDA or pygame required."""
import io
import math
import os
import struct
import tempfile
import unittest
import zlib
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

import gr_physics as phys
import gr_raytracer as tracer


class GeodesicTests(unittest.TestCase):
    def test_constants_match_schwarzschild_analytics(self):
        self.assertEqual(phys.HORIZON_R, 2.0)
        self.assertEqual(phys.PHOTON_SPHERE_R, 3.0)
        self.assertEqual(phys.ISCO_R, 6.0)
        self.assertAlmostEqual(phys.B_CRIT, 3.0 * math.sqrt(3.0), places=14)

    def test_camera_basis_is_orthonormal_and_right_handed(self):
        er, forward, right, up = phys.camera_basis(0.7, -0.3)
        self.assertTrue(np.allclose(forward, -er))
        frame = np.array([forward, right, up])
        self.assertTrue(np.allclose(frame @ frame.T, np.eye(3), atol=1e-12))
        self.assertTrue(np.allclose(np.cross(right, up), -forward, atol=1e-12))

    def test_critical_impact_parameter_separates_capture_and_escape(self):
        self.assertFalse(phys.integrate_null_geodesic(phys.B_CRIT * 0.999)["escaped"])
        self.assertTrue(phys.integrate_null_geodesic(phys.B_CRIT * 1.001)["escaped"])

    def test_rk4_preserves_null_orbit_first_integral(self):
        result = phys.integrate_null_geodesic(phys.B_CRIT * 1.001, step=0.002)
        self.assertTrue(result["escaped"])
        self.assertLess(result["invariant_error"], 1.0e-10)

    def test_periapsis_matches_cubic_root(self):
        for b in (5.3, 6.0, 10.0, 20.0):
            r_p = phys.periapsis_radius(b)
            self.assertAlmostEqual(r_p**3 - b * b * r_p + 2.0 * b * b, 0.0, delta=1e-9)
            ray = phys.integrate_null_geodesic(b, step=0.002)
            self.assertAlmostEqual(ray["r_min"], r_p, delta=1e-8)
        self.assertAlmostEqual(phys.periapsis_radius(phys.B_CRIT * (1 + 1e-12)), 3.0, delta=1e-4)
        with self.assertRaises(ValueError):
            phys.periapsis_radius(phys.B_CRIT)

    def test_exit_angle_matches_independent_quadrature(self):
        for b in (5.3, 6.0, 10.0, 20.0):
            for background in (math.inf, 100.0):
                ray = phys.integrate_null_geodesic(b, background_radius=background, step=0.003)
                reference = phys.exit_angle_quadrature(b, background_radius=background)
                # Limited by the linear interpolation of the crossing inside the
                # last step; still four orders of magnitude below the step size.
                self.assertAlmostEqual(ray["phi"], reference, delta=5e-7)

    def test_integrator_converges_at_fourth_order(self):
        # Just above the critical orbit the ray winds around the photon
        # sphere, so the step error is far above round-off.
        b = phys.B_CRIT * 1.0001
        reference = phys.exit_angle_quadrature(b)
        coarse = abs(phys.integrate_null_geodesic(b, step=0.02)["phi"] - reference)
        fine = abs(phys.integrate_null_geodesic(b, step=0.01)["phi"] - reference)
        self.assertLess(fine, 1.0e-5)
        self.assertGreater(coarse / fine, 8.0)

    def test_weak_field_deflection(self):
        b = 1000.0
        self.assertAlmostEqual(phys.deflection_angle(b), 4.0 / b + 15.0 * math.pi / (4.0 * b * b), delta=1e-7)

    def test_static_observer_shadow_edge(self):
        # sin(theta_shadow) = b_c sqrt(1-2/r) / r  (Synge 1966).
        r = 30.0
        mu = -math.sqrt(1.0 - (phys.B_CRIT * math.sqrt(1.0 - 2.0 / r) / r) ** 2)
        self.assertAlmostEqual(phys.impact_parameter(r, mu), phys.B_CRIT, places=12)

    def test_radius_at_angle_follows_the_orbit(self):
        b = 8.0
        to_periapsis = phys.angle_to_periapsis(b, 30.0)
        self.assertAlmostEqual(phys.radius_at_angle(b, to_periapsis), phys.periapsis_radius(b), delta=1e-8)
        # The orbit is symmetric about periapsis.
        self.assertAlmostEqual(phys.radius_at_angle(b, 2.0 * to_periapsis), 30.0, delta=1e-6)
        self.assertIsNone(phys.radius_at_angle(5.0, 20.0))


class DiskTests(unittest.TestCase):
    def test_flux_matches_page_thorne_integral(self):
        # f = -Omega,r (E - Omega L)^-2 Int_{r_ms}^{r} (E - Omega L) L,r dr,  F = f / (4 pi r).
        def numeric_flux(radius: float) -> float:
            r = np.linspace(phys.ISCO_R, radius, 200001)
            energy = (1.0 - 2.0 / r) / np.sqrt(1.0 - 3.0 / r)
            ang_mom = np.sqrt(r) / np.sqrt(1.0 - 3.0 / r)
            omega = r**-1.5
            integrand = (energy - omega * ang_mom) * np.gradient(ang_mom, r)
            integral = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r)))
            shear = 1.5 * radius**-2.5
            return shear / (energy[-1] - omega[-1] * ang_mom[-1]) ** 2 * integral / (4.0 * math.pi * radius)

        for radius in (6.5, 8.0, 12.0, 20.0, 60.0):
            self.assertAlmostEqual(phys.novikov_thorne_flux(radius) / numeric_flux(radius), 1.0, delta=1e-6)

    def test_flux_limits(self):
        self.assertEqual(phys.novikov_thorne_flux(phys.ISCO_R), 0.0)
        self.assertEqual(phys.novikov_thorne_flux(4.0), 0.0)
        # Newtonian Shakura-Sunyaev limit 3/(8 pi r^3) far from the hole.
        self.assertAlmostEqual(phys.novikov_thorne_flux(1.0e6) * 8.0 * math.pi * 1.0e18 / 3.0, 1.0, delta=5e-3)
        with self.assertRaises(ValueError):
            phys.novikov_thorne_flux(10.0, inner_radius=4.0)

    def test_flux_peak(self):
        r_peak, _ = phys.flux_peak()
        self.assertAlmostEqual(r_peak, 9.5509, delta=1e-3)
        self.assertAlmostEqual(phys.disk_flux_profile(r_peak), 1.0, places=9)
        self.assertLess(phys.disk_flux_profile(20.0), 1.0)
        # A truncated disk still vanishes at its own inner edge.
        self.assertEqual(phys.disk_flux_profile(8.0, inner_radius=8.0), 0.0)
        self.assertGreater(phys.flux_peak(8.0)[0], r_peak)

    def test_flux_lut_samples_the_profile(self):
        lut = phys.build_flux_lut(6.0, 20.0, samples=257)
        self.assertEqual(lut.dtype, np.float32)
        self.assertEqual(lut[0], 0.0)
        self.assertAlmostEqual(float(lut[-1]), phys.disk_flux_profile(20.0), places=6)
        self.assertLessEqual(float(lut.max()), 1.0)

    def test_g_factor_limits(self):
        # On the rotation axis (lambda = 0) only time dilation remains.
        r, r_cam = 10.0, 30.0
        expected = math.sqrt(1.0 - 3.0 / r) / math.sqrt(1.0 - 2.0 / r_cam)
        self.assertAlmostEqual(phys.schwarzschild_g_factor(r, 0.0, r_cam), expected, places=14)

    def test_rotation_reversal_mirrors_doppler_factor(self):
        prograde = phys.schwarzschild_g_factor(10.0, 3.0, rotation_sign=1.0)
        reversed_rotation = phys.schwarzschild_g_factor(10.0, 3.0, rotation_sign=-1.0)
        mirrored_ray = phys.schwarzschild_g_factor(10.0, -3.0, rotation_sign=1.0)
        self.assertGreater(prograde, reversed_rotation)
        self.assertAlmostEqual(reversed_rotation, mirrored_ray, places=14)

    def test_emission_cosine(self):
        # Flat-space limit: a face-on ray leaves along the normal, an
        # in-plane ray grazes the surface.
        r = 1.0e6
        self.assertAlmostEqual(phys.emission_cosine(r, r, 0.0), 1.0, places=5)
        self.assertAlmostEqual(phys.emission_cosine(r, 0.5 * r, 0.5 * r), 0.0, places=5)
        self.assertAlmostEqual(phys.rim_emission_cosine(r, 0.0, 0.0), 1.0, places=5)
        for lam in (-6.0, 0.0, 6.0):
            self.assertTrue(0.0 <= phys.emission_cosine(10.0, 9.0, lam) <= 1.0)
            self.assertTrue(0.0 <= phys.rim_emission_cosine(20.0, 9.0, lam) <= 1.0)

    def test_limb_darkening_preserves_flux(self):
        mu = np.linspace(0.0, 1.0, 100001)
        weights = 2.0 * mu * np.array([phys.limb_darkening(m) for m in mu])
        flux = float(np.sum(0.5 * (weights[1:] + weights[:-1]) * np.diff(mu)))
        self.assertAlmostEqual(flux, 1.0, places=8)


class ColourTests(unittest.TestCase):
    def test_blackbody_follows_the_planckian_locus(self):
        # Reference chromaticities of the Planckian locus (CIE 1931 2 deg).
        for temperature, (x_ref, y_ref) in {
            2856.0: (0.4476, 0.4074),
            4000.0: (0.3805, 0.3768),
            6500.0: (0.3135, 0.3237),
            10000.0: (0.2807, 0.2884),
        }.items():
            x, y = phys.blackbody_chromaticity(temperature)
            self.assertAlmostEqual(x, x_ref, delta=2.5e-3)
            self.assertAlmostEqual(y, y_ref, delta=2.5e-3)

    def test_blackbody_colour_and_brightness_ordering(self):
        cool, white, hot = (phys.blackbody_linear_srgb(t)[0] for t in (3000.0, 6500.0, 20000.0))
        self.assertGreater(cool[0], cool[2])
        self.assertGreater(hot[2], hot[0])
        self.assertLess(white.max() / white.min(), 1.1)
        self.assertLess(cool.sum(), white.sum())
        self.assertLess(white.sum(), hot.sum())

    def test_blackbody_lut_interpolates_log_luminance(self):
        table, ln_t0, dln_t = phys.build_blackbody_lut()
        self.assertEqual(table.dtype, np.float32)
        self.assertTrue(np.isfinite(table).all())
        for temperature in (800.0, 4500.0, 31234.0):
            x = (math.log(temperature) - ln_t0) / dln_t
            i = int(x)
            ln_y = table[i, 3] * (1.0 - (x - i)) + table[i + 1, 3] * (x - i)
            self.assertAlmostEqual(ln_y, phys.blackbody_log_luminance(temperature), delta=1e-3)
        luminance_weights = np.array([0.2126729, 0.7151522, 0.0721750])
        self.assertTrue(np.allclose(table[:, :3] @ luminance_weights, 1.0, atol=1e-5))

    def test_srgb_transfer_roundtrip(self):
        values = np.linspace(0.0, 1.0, 1001)
        self.assertTrue(np.allclose(phys.srgb_decode(phys.srgb_encode(values)), values, atol=1e-12))
        self.assertAlmostEqual(float(phys.srgb_encode(0.5)), 0.7353569831, places=9)
        lut = phys.srgb_decode_lut()
        self.assertEqual(lut.shape, (256,))
        self.assertEqual(lut[0], 0.0)
        self.assertEqual(lut[255], 1.0)


class SkyTests(unittest.TestCase):
    def test_sky_texture_is_deterministic_and_dark(self):
        first = phys.make_sky_texture(seed=3, width=512, height=256, stars=800)
        again = phys.make_sky_texture(seed=3, width=512, height=256, stars=800)
        other = phys.make_sky_texture(seed=4, width=512, height=256, stars=800)
        self.assertEqual(first.shape, (256, 512, 3))
        self.assertEqual(first.dtype, np.uint8)
        self.assertTrue(np.array_equal(first, again))
        self.assertFalse(np.array_equal(first, other))
        self.assertLess(first.mean(), 40.0)
        self.assertGreater(first.max(), 200)

    def test_stars_are_uniform_over_the_sphere(self):
        # Light per unit solid angle in the polar caps (|lat| > 60 deg) versus
        # the rest.  Sampling uniformly in texture coordinates would crowd the
        # caps to about 3x the density; equal-area sampling gives < 1 here
        # because the Milky Way population mostly lies outside the caps.
        sky = phys.make_sky_texture(seed=11, width=1024, height=512, stars=6000).max(axis=2).astype(float)
        lat = (0.5 - (np.arange(512) + 0.5) / 512) * math.pi
        solid_angle = np.cos(lat)[:, None] * np.ones((1, 1024))
        light = phys.srgb_decode(sky / 255.0) * solid_angle
        caps = np.abs(lat) > math.radians(60.0)
        cap_density = light[caps].sum() / solid_angle[caps].sum()
        band_density = light[~caps].sum() / solid_angle[~caps].sum()
        self.assertLess(cap_density / band_density, 1.5)
        self.assertGreater(cap_density / band_density, 0.2)


class CommandLineTests(unittest.TestCase):
    def parse(self, *argv):
        parser = tracer.build_parser()
        args = parser.parse_args(list(argv))
        tracer.check_args(parser, args)
        return args

    def test_defaults_are_valid(self):
        args = self.parse()
        view = tracer.view_from_args(args)
        self.assertTrue(math.isinf(args.background_r))
        self.assertEqual(tracer.COLOR_MODES[view.color_mode], "blackbody")
        self.assertEqual(view.ssaa, 1)

    def test_unphysical_geometry_is_rejected(self):
        for argv in (("--disk-rin", "4"), ("--camera-r", "15"), ("--background-r", "20"),
                     ("--disk-rout", "5"), ("--disk-half-opening-deg", "9")):
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                self.parse(*argv)

    def test_view_flags(self):
        view = tracer.view_from_args(self.parse("--retrograde", "--thin-disk", "--color-mode", "gray",
                                                "--start-elevation", "5", "--ssaa", "2"))
        self.assertEqual(view.rotation_sign, -1.0)
        self.assertFalse(view.thick)
        self.assertEqual(view.color_mode, 0)
        self.assertAlmostEqual(view.pitch, math.radians(5.0))
        self.assertEqual(view.ssaa, 2)

    def test_validation_suite_passes(self):
        with redirect_stdout(io.StringIO()) as output:
            tracer.run_validations()
        self.assertIn("validation checks passed", output.getvalue())
        self.assertNotIn("FAIL", output.getvalue())

    def test_png_writer_roundtrip(self):
        image = np.random.default_rng(1).integers(0, 256, (5, 7, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "frame.png")
            tracer.write_png(path, image)
            with open(path, "rb") as f:
                data = f.read()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        width, height, depth, colour_type = struct.unpack(">IIBB", data[16:26])
        self.assertEqual((width, height, depth, colour_type), (7, 5, 8, 2))
        idat_length = struct.unpack(">I", data[33:37])[0]
        self.assertEqual(data[37:41], b"IDAT")
        rows = np.frombuffer(zlib.decompress(data[41:41 + idat_length]), np.uint8).reshape(5, 22)
        self.assertTrue((rows[:, 0] == 0).all())
        self.assertTrue(np.array_equal(rows[:, 1:].reshape(5, 7, 3), image))


if __name__ == "__main__":
    unittest.main()
