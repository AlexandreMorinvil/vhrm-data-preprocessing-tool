import tempfile
import unittest
from pathlib import Path

from app.file_cleanup import ARCHIVE_DIRECTORY_NAME, cleanup_obsolete_paths


class FileCleanupTests(unittest.TestCase):
    def test_archives_file_with_project_relative_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "labelled_segments" / "0000_Baseline" / "signal.csv"
            source.parent.mkdir(parents=True)
            source.write_text("legacy", encoding="utf-8")

            cleaned = cleanup_obsolete_paths([source], root, archive=True)

            archived = root / ARCHIVE_DIRECTORY_NAME / source.relative_to(root)
            self.assertEqual(cleaned, 1)
            self.assertFalse(source.exists())
            self.assertEqual(archived.read_text(encoding="utf-8"), "legacy")

    def test_archive_keeps_existing_copy_on_name_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "signal_synced.csv"
            source.write_text("first", encoding="utf-8")
            cleanup_obsolete_paths([source], root, archive=True)
            source.write_text("second", encoding="utf-8")

            cleanup_obsolete_paths([source], root, archive=True)

            archived_files = list((root / ARCHIVE_DIRECTORY_NAME).glob("signal_synced*.csv"))
            self.assertEqual(len(archived_files), 2)
            self.assertEqual(
                {path.read_text(encoding="utf-8") for path in archived_files},
                {"first", "second"},
            )

    def test_deletes_permanently_when_archive_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "hr_synced.csv"
            source.write_text("legacy", encoding="utf-8")

            cleaned = cleanup_obsolete_paths([source], root, archive=False)

            self.assertEqual(cleaned, 1)
            self.assertFalse(source.exists())
            self.assertFalse((root / ARCHIVE_DIRECTORY_NAME).exists())

    def test_does_not_remove_file_outside_project(self) -> None:
        with tempfile.TemporaryDirectory() as project_directory:
            with tempfile.TemporaryDirectory() as external_directory:
                source = Path(external_directory) / "signal_synced.csv"
                source.write_text("keep", encoding="utf-8")

                cleaned = cleanup_obsolete_paths(
                    [source], project_directory, archive=False
                )

                self.assertEqual(cleaned, 0)
                self.assertTrue(source.exists())


if __name__ == "__main__":
    unittest.main()