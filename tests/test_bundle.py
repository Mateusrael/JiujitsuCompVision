import tempfile
import unittest
import zipfile
from pathlib import Path

from src.Scripts.bundle_project import build_bundle


class BundleTests(unittest.TestCase):
    def test_excludes_data_local_config_caches_and_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("src/Scripts/train.py", "README.md", "Data/annotations.json",
                         "trainings/run/model.pt", "launchers/local_config.sh", ".env",
                         "src/__pycache__/module.pyc", "launchers/START_TRAINING.sh"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"example\r\n")
            output = root / "dist/source.zip"
            build_bundle(root, output)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(set(archive.namelist()), {
                    "src/Scripts/train.py", "README.md", "launchers/START_TRAINING.sh",
                    "source_manifest.json"})
                self.assertNotIn(b"\r", archive.read("launchers/START_TRAINING.sh"))
            with self.assertRaises(FileExistsError):
                build_bundle(root, output)


if __name__ == "__main__":
    unittest.main()
