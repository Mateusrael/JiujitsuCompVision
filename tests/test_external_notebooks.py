"""Offline checks for external-image identity and the annotation notebooks."""

from collections import Counter
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def notebook(name):
    return json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))


def execute_cell(document, index, namespace):
    exec(compile("".join(document["cells"][index]["source"]), f"cell-{index}", "exec"), namespace)


class ExternalDatasetIdentityTests(unittest.TestCase):
    def test_unique_verified_images_and_preserved_ids(self):
        directory = ROOT / "Data" / "jiujitsu_images_out_of_sample"
        rows = json.loads((ROOT / "Data" / "jiujitsu_images_out_of_sample_annotations.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 99)
        self.assertEqual(len({row["image"] for row in rows}), 99)
        self.assertEqual(len({row["frame"] for row in rows}), 99)
        self.assertNotIn("half_guard/image_0005.jpg", {row["image"] for row in rows})
        self.assertFalse((directory / "half_guard/image_0005.jpg").exists())
        counts = Counter(row["image"].split("/")[0] for row in rows)
        self.assertEqual(counts["half_guard"], 9)
        self.assertTrue(all(count == 10 for name, count in counts.items() if name != "half_guard"))
        digests = []
        for row in rows:
            digest = hashlib.sha256((directory / row["image"]).read_bytes()).hexdigest()
            self.assertEqual(digest, row["image_sha256"], row["image"])
            digests.append(digest)
        self.assertEqual(len(set(digests)), 99)

    def test_notebook_code_compiles_without_shell_mutations(self):
        for name in ("annotate_positions", "extract_pose"):
            for index, cell in enumerate(notebook(name)["cells"]):
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"])
                    compile(source, f"{name}:cell-{index}", "exec")
                    self.assertNotIn("rename_images.sh", source)
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])


@unittest.skipUnless(os.environ.get("JIUJITSU_RUN_NOTEBOOK_TESTS") == "1",
                     "Set JIUJITSU_RUN_NOTEBOOK_TESTS=1 with optional notebook dependencies")
class AnnotationNotebookTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.images = self.directory / "images"
        self.image = self.images / "mount" / "image_0007.jpg"
        self.image.parent.mkdir(parents=True)
        Image.new("RGB", (128, 128), "white").save(self.image)
        self.annotation = self.directory / "images_annotations.json"
        self.pose = [[30 + i * 3, 50 + i * 2, .9] for i in range(17)]
        self.record = {"image": "mount/image_0007.jpg", "frame": 77, "position": "mount1",
                       "pose1": self.pose, "pose2": self.pose,
                       "image_sha256": hashlib.sha256(self.image.read_bytes()).hexdigest()}
        self.annotation.write_text(json.dumps([self.record]), encoding="utf-8")
        self.document = notebook("annotate_positions")
        self.namespace = {"hashlib": hashlib, "io": io, "json": json, "Path": Path,
                          "PASTA": self.images, "SAIDA": self.annotation,
                          "CACHE_POSES": self.directory / "cache.json", "imagens": [self.image],
                          "POSITIONS": ["mount1", "mount2"]}
        with contextlib.redirect_stdout(io.StringIO()):
            execute_cell(self.document, 3, self.namespace)

    def test_saved_poses_render_and_edit_without_detector_or_renumbering(self):
        with patch("src.Eval.pose_detection.VitPoseDetector", side_effect=AssertionError("Unexpected detector")) as detector:
            self.assertEqual(self.namespace["obter_poses"](self.image), (self.pose, self.pose))
            execute_cell(self.document, 7, self.namespace)
            with patch("IPython.display.display"):
                execute_cell(self.document, 9, self.namespace)
            self.assertTrue(self.namespace["tela"].value)
            self.namespace["menu"].value = "mount2"
            self.namespace["ao_salvar"](None)
            detector.assert_not_called()
        saved = json.loads(self.annotation.read_text(encoding="utf-8"))[0]
        self.assertEqual(saved["frame"], 77)
        self.assertEqual(saved["image"], self.record["image"])
        self.assertEqual(saved["image_sha256"], self.record["image_sha256"])
        self.assertEqual(saved["pose1"], self.pose)
        self.assertEqual(saved["position"], "mount2")
        self.assertFalse(self.namespace["CACHE_POSES"].exists())
        self.assertFalse((self.directory / "images_metadata.json").exists())

    def test_new_annotation_updates_existing_metadata_counts_preserving_provenance(self):
        from PIL import Image
        second = self.images / "standing" / "new.jpg"
        second.parent.mkdir()
        Image.new("RGB", (128, 128), "blue").save(second)
        self.namespace["imagens"].append(second)
        self.namespace["POSITIONS"].append("standing")
        digest = hashlib.sha256(second.read_bytes()).hexdigest()
        self.namespace["_cache"][digest] = [self.pose, self.pose]
        metadata_path = self.directory / "images_metadata.json"
        provenance = {"poses": {"origin": "detector_generated", "pose_model": "original-model"},
                      "source_provenance": {"independence_from_training_dataset_verified": False}}
        metadata_path.write_text(json.dumps({**provenance, "samples": 99,
                                            "class_counts": {"obsolete": 99},
                                            "annotations_file": "previous.json"}), encoding="utf-8")
        with patch("src.Eval.pose_detection.VitPoseDetector", side_effect=AssertionError("Unexpected detector")) as detector:
            execute_cell(self.document, 7, self.namespace)
            with patch("IPython.display.display"):
                execute_cell(self.document, 9, self.namespace)
            self.namespace["menu"].value = "standing"
            self.namespace["ao_salvar"](None)
            detector.assert_not_called()
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(saved, {**provenance, "samples": 2,
                                "class_counts": {"mount": 1, "standing": 1},
                                "annotations_file": self.annotation.name})
        records = json.loads(self.annotation.read_text(encoding="utf-8"))
        self.assertEqual([row["frame"] for row in records], [77, 78])

    def test_saved_missing_athlete_is_not_recomputed(self):
        del self.namespace["anotacoes"][self.record["image"]]["pose2"]
        with patch("src.Eval.pose_detection.VitPoseDetector", side_effect=AssertionError("Unexpected detector")):
            self.assertEqual(self.namespace["obter_poses"](self.image), (self.pose, None))

    def test_replaced_image_and_unverified_annotation_are_rejected(self):
        self.image.write_bytes(b"replacement bytes under the old annotation path")
        with self.assertRaisesRegex(ValueError, "conteúdo mudou"):
            self.namespace["obter_poses"](self.image)
        del self.namespace["anotacoes"][self.record["image"]]["image_sha256"]
        with self.assertRaisesRegex(ValueError, "sem image_sha256"):
            self.namespace["obter_poses"](self.image)

    def test_notebook_schema_and_included_default_image(self):
        import nbformat
        for name in ("annotate_positions", "extract_pose"):
            nbformat.validate(nbformat.from_dict(notebook(name)))
        import ast
        cell = "".join(notebook("extract_pose")["cells"][9]["source"])
        assignment = ast.Module(body=[ast.parse(cell).body[0]], type_ignores=[])
        namespace = {"ROOT": ROOT}
        exec(compile(assignment, "default-image", "exec"), namespace)
        self.assertTrue(namespace["IMAGEM"].is_file())


class RenameIdentityGuardTests(unittest.TestCase):
    def setUp(self):
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        self.bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
        if not self.bash:
            self.skipTest("Bash unavailable")
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.images = self.directory / "sample" / "half_guard"
        self.images.mkdir(parents=True)
        for name, data in (("added.jpg", b"new"), ("image_0001.jpg", b"old1"), ("image_0002.jpg", b"old2")):
            (self.images / name).write_bytes(data)

    def run_rename(self, *arguments):
        return subprocess.run([self.bash, (ROOT / "notebooks/rename_images.sh").as_posix(),
                               self.images.as_posix(), *arguments], capture_output=True, text=True,
                              encoding="utf-8", timeout=20)

    def snapshot(self):
        return {path.name: path.read_bytes() for path in self.images.iterdir()}

    def test_existing_dataset_annotations_block_rename_but_allow_dry_run(self):
        (self.directory / "sample_annotations.json").write_text("[]", encoding="utf-8")
        original = self.snapshot()
        dry_run = self.run_rename("--dry-run")
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertEqual(self.snapshot(), original)
        rename = self.run_rename()
        self.assertNotEqual(rename.returncode, 0)
        self.assertIn("anotações existentes", rename.stderr)
        self.assertEqual(self.snapshot(), original)

    def test_unannotated_folder_can_be_organized_before_annotation(self):
        result = self.run_rename()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.snapshot(), {"image_0001.jpg": b"new", "image_0002.jpg": b"old1", "image_0003.jpg": b"old2"})


if __name__ == "__main__":
    unittest.main()
