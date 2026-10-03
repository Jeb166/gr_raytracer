import math
import unittest

import numpy as np

import gpu_disk_viewer_v062 as tracer


class SchwarzschildPhysicsTests(unittest.TestCase):
    def test_camera_basis_is_orthonormal(self):
        er, forward, right, up = tracer.basis(0.7, -0.3)
        self.assertTrue(np.allclose(forward, -er))
        vectors = (forward, right, up)
        for vector in vectors:
            self.assertAlmostEqual(float(np.dot(vector, vector)), 1.0, places=12)
        for i, left in enumerate(vectors):
            for right_vector in vectors[i + 1:]:
                self.assertAlmostEqual(float(np.dot(left, right_vector)), 0.0, places=12)

    def test_emissivity_has_analytical_peak(self):
        peak_radius = tracer.ISCO_R * 49.0 / 36.0
        self.assertEqual(tracer.disk_flux_proxy(tracer.ISCO_R), 0.0)
        self.assertAlmostEqual(tracer.disk_flux_proxy(peak_radius), 1.0, places=12)
        self.assertLess(tracer.disk_flux_proxy(20.0), 1.0)

    def test_critical_impact_parameter_separates_capture_and_escape(self):
        captured = tracer.integrate_null_geodesic_cpu(tracer.B_CRIT * 0.995, step=0.002)
        escaped = tracer.integrate_null_geodesic_cpu(tracer.B_CRIT * 1.005, step=0.002)
        self.assertFalse(captured["escaped"])
        self.assertTrue(escaped["escaped"])

    def test_rk4_preserves_null_orbit_first_integral(self):
        result = tracer.integrate_null_geodesic_cpu(6.0, step=0.001)
        self.assertTrue(result["escaped"])
        self.assertLess(result["invariant_error"], 2.0e-8)

    def test_path_converges_when_step_is_halved(self):
        coarse = tracer.integrate_null_geodesic_cpu(6.0, step=0.002)
        fine = tracer.integrate_null_geodesic_cpu(6.0, step=0.001)
        self.assertLess(abs(coarse["phi"] - fine["phi"]), 0.004)
        self.assertLess(abs(coarse["r_min"] - fine["r_min"]), 0.004)

    def test_rotation_reversal_mirrors_doppler_factor(self):
        prograde = tracer.schwarzschild_g_factor(10.0, 3.0, rotation_sign=1.0)
        reversed_rotation = tracer.schwarzschild_g_factor(10.0, 3.0, rotation_sign=-1.0)
        mirrored_ray = tracer.schwarzschild_g_factor(10.0, -3.0, rotation_sign=1.0)
        self.assertGreater(prograde, reversed_rotation)
        self.assertAlmostEqual(reversed_rotation, mirrored_ray, places=14)

    def test_constants_match_schwarzschild_analytics(self):
        self.assertEqual(tracer.HORIZON_R, 2.0)
        self.assertEqual(tracer.PHOTON_SPHERE_R, 3.0)
        self.assertAlmostEqual(tracer.B_CRIT, 3.0 * math.sqrt(3.0), places=14)


if __name__ == "__main__":
    unittest.main()
