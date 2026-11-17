"""GPU-versus-CPU parity checks.  Skipped when CuPy or a CUDA device is missing.

These tests render small frames with the real kernels and compare them with
values predicted by the NumPy reference in ``gr_physics``.
"""
import dataclasses
import math
import unittest
import warnings

import numpy as np

import gr_physics as phys
import gr_raytracer as tracer

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cupy
        HAVE_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:  # missing package, driver or DLL
    HAVE_CUDA = False

WIDTH, HEIGHT = 640, 360
ORDER_COLOURS = np.array([[45, 255, 80], [255, 55, 220], [45, 190, 255], [255, 185, 35], [235, 235, 235]])


def pixel_ray(view: tracer.View, x: int, y: int, camera_r: float, width: int = WIDTH, height: int = HEIGHT):
    """Reference camera model: local direction cosine, orbital-plane tangent,
    impact parameter and axial angular momentum of a pixel-centre ray."""
    er, forward, right, up = phys.camera_basis(view.yaw, view.pitch)
    tan_y = math.tan(math.radians(view.fov) / 2.0)
    tan_x = tan_y * width / height
    px = ((x + 0.5) / width * 2.0 - 1.0) * tan_x
    py = (1.0 - (y + 0.5) / height * 2.0) * tan_y
    d = forward + px * right + py * up
    d /= np.linalg.norm(d)
    mu = float(d @ er)
    tangent = d - mu * er
    tangent /= np.linalg.norm(tangent)
    scale = camera_r / math.sqrt(1.0 - 2.0 / camera_r)
    return er, tangent, scale * math.sqrt(1.0 - mu * mu), -scale * (er[2] * d[0] - er[0] * d[2])


def primary_hit_radius(er, tangent, impact: float, camera_r: float):
    """Radius at which the ray first crosses the equatorial plane."""
    crossing = math.atan2(-er[1], tangent[1])
    if crossing <= 0.0:
        crossing += math.pi
    return phys.radius_at_angle(impact, crossing, camera_r)


def disk_mask(order_image: np.ndarray) -> np.ndarray:
    distance = np.abs(order_image[:, :, None, :].astype(int) - ORDER_COLOURS[None, None]).sum(axis=3)
    return distance.min(axis=2) < 12


@unittest.skipUnless(HAVE_CUDA, "CuPy with a CUDA device is required")
class GpuParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = tracer.build_parser()
        cls.args = parser.parse_args(["--render-width", str(WIDTH), "--render-height", str(HEIGHT)])
        tracer.check_args(parser, cls.args)
        cls.renderer = tracer.Renderer(cls.args, log=lambda *_: None)
        cls.view = dataclasses.replace(tracer.view_from_args(cls.args), grid=False)

    def render(self, **changes) -> np.ndarray:
        return self.renderer.render(dataclasses.replace(self.view, **changes)).copy()

    def table_impact(self, index: int, count: int) -> float:
        return phys.impact_parameter(self.args.camera_r, -1.0 + index / (count - 1))

    # --- geodesic tables -----------------------------------------------------

    def test_transfer_table_matches_quadrature(self):
        escaped = self.renderer.esc.get()
        exit_angle = self.renderer.pexit.get()
        count = escaped.size
        checked = 0
        for index in range(0, count, 257):
            b = self.table_impact(index, count)
            if abs(b / phys.B_CRIT - 1.0) < 2.0e-3:
                continue
            self.assertEqual(bool(escaped[index]), b > phys.B_CRIT, f"ray {index}, b = {b}")
            if escaped[index] and index < count - 1:
                reference = phys.exit_angle_quadrature(b, self.args.camera_r, self.args.background_r)
                self.assertAlmostEqual(float(exit_angle[index]), reference, delta=1.0e-5)
                checked += 1
        self.assertGreater(checked, 100)
        # The exit angle is continuous in mu: no step-sized stairs.
        smooth = exit_angle[-2000:]
        self.assertLess(float(np.abs(np.diff(smooth, 2)).max()), 1.0e-5)

    def test_path_table_matches_reference_orbits(self):
        a, r = self.args, self.renderer
        path = r.path.get()
        rout_hit, phi_rout = r.rout_hit.get(), r.phi_rout.get()
        for index in (500, 3000, 9000, 15000):
            b = self.table_impact(index, a.path_rays)
            for column in (40, 150, 260):
                reference = phys.radius_at_angle(b, column * r.dphi, a.camera_r)
                stored = float(path[index, column])
                if reference is None:
                    self.assertEqual(stored, 0.0)
                elif stored > 0.0:  # rows stop once the ray recedes past 2 r_out
                    self.assertAlmostEqual(1.0 / stored, reference, delta=2.0e-5 * reference)
            reaches_disk = b <= phys.B_CRIT or phys.periapsis_radius(b) < a.disk_rout
            self.assertEqual(bool(rout_hit[index]), reaches_disk)
            if b > phys.B_CRIT and reaches_disk:
                reference = phys.angle_to_periapsis(b, a.camera_r) - phys.angle_to_periapsis(b, a.disk_rout)
                self.assertAlmostEqual(float(phi_rout[index]), reference, delta=1.0e-5)

    # --- images --------------------------------------------------------------

    def test_frame_format(self):
        frame = self.render()
        self.assertEqual(frame.shape, (HEIGHT, WIDTH, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(int(frame.max()), 200)

    def test_shadow_has_the_analytic_angular_size(self):
        frame = self.render(disk=False, pitch=0.0)
        black = (frame[HEIGHT // 2] == 0).all(axis=1)
        r = self.args.camera_r
        theta = math.asin(phys.B_CRIT * math.sqrt(1.0 - 2.0 / r) / r)
        tan_x = math.tan(math.radians(self.view.fov) / 2.0) * WIDTH / HEIGHT
        expected = math.tan(theta) / tan_x * WIDTH
        self.assertAlmostEqual(int(black.sum()), expected, delta=1.5)
        columns = np.flatnonzero(black)
        self.assertAlmostEqual(0.5 * (columns[0] + columns[-1]), 0.5 * (WIDTH - 1), delta=0.6)

    def predicted_disk_pixels(self, view: tracer.View, shade):
        """(x, y, expected sRGB triple) for pixels whose primary image lies
        well inside the disk, using only the CPU reference."""
        a = self.args
        samples = []
        for y in range(8, HEIGHT, 23):
            for x in range(8, WIDTH, 37):
                er, tangent, b, lam = pixel_ray(view, x, y, a.camera_r)
                radius = primary_hit_radius(er, tangent, b, a.camera_r)
                if radius is None or not a.disk_rin + 0.4 < radius < a.disk_rout - 0.4:
                    continue
                linear = np.asarray(shade(radius, b, lam), dtype=float)
                samples.append((x, y, np.rint(255.0 * phys.srgb_encode(linear))))
        self.assertGreater(len(samples), 40)
        return samples

    def assert_pixels(self, frame, samples, tolerance=2):
        worst = max(float(np.abs(frame[y, x].astype(float) - expected).max()) for x, y, expected in samples)
        self.assertLessEqual(worst, tolerance)

    def test_gray_disk_matches_cpu_flux_shift_and_limb_darkening(self):
        view = dataclasses.replace(self.view, color_mode=0, thick=False, pitch=math.radians(25.0), yaw=0.4)
        a = self.args

        def shade(radius, b, lam):
            g = phys.schwarzschild_g_factor(radius, lam, a.camera_r, view.rotation_sign)
            limb = phys.limb_darkening(phys.emission_cosine(radius, b, lam, view.rotation_sign))
            intensity = phys.disk_flux_profile(radius, a.disk_rin) * g**4 * limb
            return np.full(3, 1.0 - math.exp(-view.exposure * intensity))

        self.assert_pixels(self.renderer.render(view), self.predicted_disk_pixels(view, shade))

    def test_blackbody_disk_matches_cpu_colour(self):
        view = dataclasses.replace(self.view, color_mode=1, thick=False, pitch=math.radians(40.0),
                                   rotation_sign=-1.0, limb_darkening=False)
        a = self.args
        reference_luminance = math.exp(phys.blackbody_log_luminance(view.tmax))

        def shade(radius, b, lam):
            g = phys.schwarzschild_g_factor(radius, lam, a.camera_r, view.rotation_sign)
            temperature = view.tmax * phys.disk_flux_profile(radius, a.disk_rin) ** 0.25 * g
            rgb = phys.blackbody_linear_srgb(temperature)[0] / reference_luminance
            return 1.0 - np.exp(-view.exposure * rgb)

        self.assert_pixels(self.renderer.render(view), self.predicted_disk_pixels(view, shade))

    def test_edge_on_rim_matches_cpu_shift_and_limb_darkening(self):
        # Exactly edge-on, every centre-row ray that reaches r_out hits the
        # wedge's outer rim first, so the pixel is fixed by r = r_out alone.
        view = dataclasses.replace(self.view, color_mode=0, pitch=0.0)
        a = self.args
        flux = phys.disk_flux_profile(a.disk_rout, a.disk_rin)
        frame = self.renderer.render(view)
        worst = 0.0
        for x in range(100, WIDTH - 99, 20):
            _, _, b, lam = pixel_ray(view, x, HEIGHT // 2, a.camera_r)
            g = phys.schwarzschild_g_factor(a.disk_rout, lam, a.camera_r)
            limb = phys.limb_darkening(phys.rim_emission_cosine(a.disk_rout, b, lam))
            expected = round(255.0 * float(phys.srgb_encode(1.0 - math.exp(-view.exposure * flux * g**4 * limb))))
            worst = max(worst, float(np.abs(frame[HEIGHT // 2, x].astype(float) - expected).max()))
        self.assertLessEqual(worst, 2)

    def test_unshifted_disk_depends_on_radius_only(self):
        view = dataclasses.replace(self.view, color_mode=0, thick=False, relativity=False,
                                   limb_darkening=False, pitch=math.radians(30.0))
        a = self.args

        def shade(radius, b, lam):
            return np.full(3, 1.0 - math.exp(-view.exposure * phys.disk_flux_profile(radius, a.disk_rin)))

        frame = self.renderer.render(view).copy()
        self.assert_pixels(frame, self.predicted_disk_pixels(view, shade))
        mask = disk_mask(self.render(order_colors=True, thick=False, pitch=view.pitch))
        mask &= mask[:, ::-1]
        difference = np.abs(frame.astype(int) - frame[:, ::-1].astype(int))[mask]
        self.assertLessEqual(int(difference.max()), 2)

    def test_approaching_side_is_brighter_and_reverses_with_spin(self):
        settings = dict(color_mode=0, pitch=math.radians(15.0))
        mask = disk_mask(self.render(order_colors=True, pitch=settings["pitch"]))
        mask &= mask[:, ::-1]
        prograde = self.render(**settings).astype(int)
        retrograde = self.render(rotation_sign=-1.0, **settings).astype(int)
        left = mask.copy()
        left[:, WIDTH // 2:] = False
        # Spin about +y: the left side of the image moves towards the camera.
        self.assertGreater(prograde[left].mean(), 1.3 * prograde[left[:, ::-1]].mean())
        # Reversing the spin is the mirror image.
        self.assertLessEqual(int(np.abs(prograde - retrograde[:, ::-1])[mask].max()), 2)

    def test_image_orders(self):
        orders = self.render(order_colors=True, thick=False, pitch=math.radians(20.0))
        primary = (np.abs(orders.astype(int) - ORDER_COLOURS[0]).sum(axis=2) < 12)
        secondary = (np.abs(orders.astype(int) - ORDER_COLOURS[1]).sum(axis=2) < 12)
        self.assertGreater(int(primary.sum()), 20 * int(secondary.sum()))
        self.assertGreater(int(secondary.sum()), 50)
        limited = self.render(order_colors=True, thick=False, pitch=math.radians(20.0), max_order=0)
        self.assertEqual(int((np.abs(limited.astype(int) - ORDER_COLOURS[1]).sum(axis=2) < 12).sum()), 0)

    def test_thick_and_thin_disks_agree_away_from_edge_on(self):
        pitch = math.radians(35.0)
        thick = disk_mask(self.render(order_colors=True, pitch=pitch))
        thin = disk_mask(self.render(order_colors=True, pitch=pitch, thick=False))
        self.assertLess(int((thick ^ thin).sum()), 0.02 * int(thin.sum()))

    def test_outer_rim_is_continuous_through_edge_on(self):
        # Seen edge-on, the wedge's outer rim at this column is the near side
        # of the r_out circle; its apparent height follows from flat geometry
        # because the ray never comes closer than ~12 M.
        a = self.args
        width, height, column = 1920, 1080, 1400
        self.renderer.set_resolution(width, height)
        try:
            tan_y = math.tan(math.radians(self.view.fov) / 2.0)
            tan_alpha = ((column + 0.5) / width * 2.0 - 1.0) * tan_y * width / height
            alpha = math.atan(tan_alpha)
            distance = a.camera_r * math.cos(alpha) - math.sqrt(
                a.disk_rout**2 - (a.camera_r * math.sin(alpha)) ** 2)
            half_height = a.disk_rout * math.tan(math.radians(self.view.half_opening_deg))
            expected = 2.0 * half_height / distance / (2.0 * tan_y / height * math.cos(alpha))
            counts = []
            for elevation in (0.0, 0.05, 0.1, 0.15, 0.2, 0.24, 0.26, 0.3):
                frame = self.render(order_colors=True, pitch=math.radians(elevation))
                counts.append(int(disk_mask(frame[:, column:column + 1]).sum()))
        finally:
            self.renderer.set_resolution(WIDTH, HEIGHT)
        self.assertAlmostEqual(counts[0], expected, delta=0.15 * expected)
        for before, after in zip(counts, counts[1:]):
            self.assertLessEqual(abs(after - before), 2, counts)

    def test_supersampling_only_changes_edges(self):
        aliased = self.render(pitch=math.radians(20.0)).astype(int)
        smooth = self.render(pitch=math.radians(20.0), ssaa=3).astype(int)
        difference = np.abs(aliased - smooth).max(axis=2)
        self.assertGreater(int(difference.max()), 20)
        self.assertLess(float((difference > 12).mean()), 0.08)


if __name__ == "__main__":
    unittest.main()
