# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.RayTracer import (
    Camera,
    KerrSchild,
    OrthographicProjection,
    PerspectiveProjection,
    RayStatus,
    trace_frame,
)


class TestRayTracer(unittest.TestCase):
    def make_camera(self, projection):
        return Camera(
            position=[50.0, 0.0, 0.0],
            focus=[0.0, 0.0, 0.0],
            up=[0.0, 0.0, 1.0],
            projection=projection,
            resolution=[3, 3],
            center_rays=True,
            start_time=0.0,
            interval=0.0,
            num_frames=1,
            integration_time=500.0,
        )

    def test_schwarzschild_shadow(self):
        background = KerrSchild(mass=1.0, spin=[0.0, 0.0, 0.0])
        # A camera at r = 50M sees the shadow with an angular radius of about
        # 5.8 degrees. With 3x3 pixels across a 30 degree opening angle only
        # the center pixel lies inside the shadow.
        camera = self.make_camera(PerspectiveProjection(opening_angle=[30, 30]))
        npt.assert_allclose(camera.position, [50.0, 0.0, 0.0])
        self.assertEqual(camera.resolution, [3, 3])
        npt.assert_allclose(camera.projection.opening_angle, [30.0, 30.0])
        ray_results, ray_statuses = trace_frame(
            background, camera, escape_radius=100.0
        )
        self.assertEqual(ray_results.shape, (9, 7))
        self.assertEqual(ray_statuses.shape, (9,))
        expected_statuses = np.full(9, int(RayStatus.Escaped))
        expected_statuses[4] = int(RayStatus.Captured)
        npt.assert_array_equal(ray_statuses, expected_statuses)
        # Escaped rays stop beyond the escape radius
        radii = np.linalg.norm(ray_results[:, :3], axis=1)
        self.assertTrue(np.all(radii[ray_statuses == RayStatus.Escaped] > 100))
        # The captured ray stops near the horizon
        self.assertLess(radii[4], 2.5)

    def test_orthographic_projection(self):
        background = KerrSchild(mass=1.0, spin=[0.0, 0.0, 0.0])
        # Parallel rays with impact parameters up to 1M all fall into the hole
        camera = self.make_camera(OrthographicProjection(extent=[2.0, 2.0]))
        npt.assert_allclose(camera.projection.extent, [2.0, 2.0])
        _, ray_statuses = trace_frame(background, camera)
        npt.assert_array_equal(ray_statuses, int(RayStatus.Captured))


if __name__ == "__main__":
    unittest.main(verbosity=2)
