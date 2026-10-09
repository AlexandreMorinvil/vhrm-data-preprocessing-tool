import unittest
from unittest.mock import call, patch

from app.audio_sync import compute_all_offsets


class AudioSyncOffsetTests(unittest.TestCase):
    def test_single_reference_mode_adjusts_for_camera_start_offsets(self) -> None:
        segments = ["cam1.mp4", "cam2.mp4", "cam3.mp4"]
        start_offsets = [1.0, 3.5, 0.0]

        with patch("app.audio_sync.compute_sync_offset", side_effect=[8.0, -2.0]) as mocked_sync:
            offsets = compute_all_offsets(
                segments,
                audio_start_offsets_sec=start_offsets,
                pairwise_refinement=False,
            )

        self.assertEqual(offsets, [0.0, 5.5, -1.0])
        self.assertEqual(
            mocked_sync.call_args_list,
            [
                call(
                    "cam1.mp4",
                    "cam2.mp4",
                    ffmpeg="",
                    max_offset_sec=60.0,
                    audio_duration_sec=60.0,
                    reference_start_sec=1.0,
                    target_start_sec=3.5,
                ),
                call(
                    "cam1.mp4",
                    "cam3.mp4",
                    ffmpeg="",
                    max_offset_sec=60.0,
                    audio_duration_sec=60.0,
                    reference_start_sec=1.0,
                    target_start_sec=0.0,
                ),
            ],
        )

    def test_pairwise_mode_recovers_original_relations(self) -> None:
        segments = ["cam1.mp4", "cam2.mp4", "cam3.mp4"]
        start_offsets = [1.0, 4.0, 0.5]

        with patch("app.audio_sync.compute_sync_offset", side_effect=[5.0, -1.5, -6.5]):
            offsets = compute_all_offsets(
                segments,
                audio_start_offsets_sec=start_offsets,
                pairwise_refinement=True,
            )

        self.assertEqual(offsets, [0.0, 2.0, -1.0])

    def test_raises_when_offset_count_does_not_match_camera_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "one value per camera"):
            compute_all_offsets(
                ["cam1.mp4", "cam2.mp4"],
                audio_start_offsets_sec=[0.0],
            )


if __name__ == "__main__":
    unittest.main()
