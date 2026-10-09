import unittest
from unittest.mock import patch

import numpy as np

from app.face_privacy import _detect_faces, _expanded_region, anonymize_faces


class FacePrivacyTests(unittest.TestCase):
    def test_expanded_region_is_clamped_to_frame(self) -> None:
        self.assertEqual(_expanded_region((0, 0, 20, 20), 100, 80, margin_ratio=0.22), (0, 0, 24, 24))
        self.assertEqual(_expanded_region((90, 70, 20, 20), 100, 80), (84, 62, 100, 80))

    def test_default_region_covers_forehead_more_than_chin(self) -> None:
        x0, y0, x1, y1 = _expanded_region((100, 100, 100, 100), 1000, 1000)
        self.assertEqual((x0, x1), (70, 230))
        self.assertLess(y0, 70)
        self.assertEqual(y1, 230)

    def test_anonymize_faces_changes_only_detected_region(self) -> None:
        row = np.arange(50, dtype=np.uint8)
        grid_x, grid_y = np.meshgrid(row, row)
        frame = np.dstack((grid_y, grid_x, np.full((50, 50), 127, dtype=np.uint8)))

        with patch("app.face_privacy._detect_faces", return_value=[(10, 10, 20, 20)]):
            anonymized = anonymize_faces(frame)

        self.assertFalse(np.array_equal(anonymized[6:34, 6:34], frame[6:34, 6:34]))
        self.assertTrue(np.array_equal(anonymized[:5, :5], frame[:5, :5]))
        self.assertTrue(np.array_equal(frame[:, :, 2], np.full((50, 50), 127, dtype=np.uint8)))

    def test_yunet_model_loads_and_returns_face_boxes(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        faces = _detect_faces(frame)

        self.assertEqual(faces, [])


if __name__ == "__main__":
    unittest.main()