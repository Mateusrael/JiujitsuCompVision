"""Moving reviewed section boundaries must move the holdout and its buffers together."""

import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10, both_poses_present, sha256_file
from src.Dataset.multiview import build_multiview_manifest
from src.Loader.splits import read_manifest
from test_multiview import fixture


class BoundaryAdjustmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.plan = fixture()
        cls.original = build_multiview_manifest(cls.records, "a" * 64, section_plan=cls.plan)
        cls.adjusted_plan = copy.deepcopy(cls.plan)
        cls.adjusted_plan["groups"][0]["boundary_adjustments"] = [
            {"before_section": "section-01", "new_boundary": 77,
             "reason": "Move validation earlier after reviewing the label transition."},
            {"before_section": "section-02", "new_boundary": 137,
             "reason": "Keep the validation window width and buffer the uncertain ending."},
        ]
        cls.adjusted = build_multiview_manifest(
            cls.records, "a" * 64, section_plan=cls.adjusted_plan)
        clocks = cls.plan["groups"][0]["alignment"]
        cls.times = {
            row["image"]: (row["frame"] - clocks[row["image"][:2]]["offset"]) /
            clocks[row["image"][:2]]["scale"] for row in cls.records
        }

    def test_plans_without_adjustments_preserve_the_original_manifest(self):
        serialized = json.dumps(self.original, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(serialized).hexdigest(),
                         "032b18145e0e276f7ce425645f9632fc5b0e4feaf68a1ee24309c174922904c8")

    def test_validation_moves_earlier_and_both_neighboring_buffers_move_with_it(self):
        original_sections = {section["id"]: section for section in self.original["sections"]}
        self.assertEqual(original_sections["section-01"]["evaluation_type"], "unseen_moment")
        self.assertEqual(original_sections["section-01"]["partition"], "val")
        changes = {72: ("train", "excluded"), 74: ("train", "excluded"),
                   83: ("excluded", "val"), 85: ("excluded", "val"),
                   132: ("val", "excluded"), 134: ("val", "excluded"),
                   143: ("excluded", "train"), 145: ("excluded", "train")}
        by_id = {row["image"]: row for row in self.records}
        for source, mapping in self.plan["groups"][0]["alignment"].items():
            for moment, (before, after) in changes.items():
                image = f"{source}{moment * mapping['scale'] + mapping['offset']:05d}"
                # Neighboring sections hold out one camera: that camera keeps its
                # validation role while the other two supply training examples.
                if moment < 80 and source == original_sections["section-00"]["held_out_source"]:
                    before = original_sections["section-00"]["partition"]
                if moment > 140 and source == original_sections["section-02"]["held_out_source"]:
                    after = original_sections["section-02"]["partition"]
                with self.subTest(source=source, moment=moment):
                    self.assertEqual(self.original["assignments"][image], before)
                    self.assertEqual(self.adjusted["assignments"][image], after)
                    if after == "excluded":
                        self.assertEqual(self.adjusted["pose_assignments"][image], "excluded")
                        self.assertEqual(self.adjusted["exclusion_reasons"][image], "temporal_buffer")
                        self.assertEqual(self.adjusted["pose_exclusion_reasons"][image], "temporal_buffer")
                    elif both_poses_present(by_id[image]):
                        self.assertEqual(self.adjusted["pose_assignments"][image], after)
                    else:
                        self.assertEqual(self.adjusted["pose_assignments"][image], "excluded")
                        self.assertEqual(self.adjusted["pose_exclusion_reasons"][image], "missing_pose")
                    if after == "val" and moment < 140:
                        self.assertEqual(self.adjusted["evaluation_types"][image], "unseen_moment")

    def test_section_roles_and_unaffected_assignments_are_preserved(self):
        for before, after in zip(self.original["sections"], self.adjusted["sections"]):
            for key in ("id", "group", "partition", "evaluation_type", "held_out_source",
                        "buffer_left", "buffer_right"):
                self.assertEqual(after[key], before[key], (before["id"], key))
        # Only the two displaced boundaries and their old/new buffer bands can change.
        affected = {image for image, time in self.times.items()
                    if 72 <= time <= 85 or 132 <= time <= 145}
        for key in ("assignments", "pose_assignments", "evaluation_types",
                    "exclusion_reasons", "pose_exclusion_reasons", "section_ids"):
            before = {image: value for image, value in self.original[key].items() if image not in affected}
            after = {image: value for image, value in self.adjusted[key].items() if image not in affected}
            self.assertEqual(after, before, key)
        for key, value in self.original["allocation"].items():
            self.assertEqual(self.adjusted["allocation"][key], value, key)
        self.assertIn("objective_scope", self.adjusted["allocation"])

    def test_effective_sections_and_audit_keep_the_nominal_plan_reproducible(self):
        sections = {section["id"]: section for section in self.adjusted["sections"]}
        self.assertEqual((sections["section-00"]["start"], sections["section-00"]["end"]), (20, 77))
        self.assertEqual((sections["section-01"]["start"], sections["section-01"]["end"]), (77, 137))
        self.assertEqual((sections["section-02"]["start"], sections["section-02"]["end"]), (137, 200))
        self.assertEqual(self.adjusted["section_plan"], self.adjusted_plan)
        self.assertEqual(self.adjusted_plan["groups"][0]["sections"], self.plan["groups"][0]["sections"])
        self.assertEqual(self.adjusted["boundary_adjustment_audit"], [
            {"group": "fixture-recording", "before_section": edit["before_section"],
             "old_boundary": old, "new_boundary": edit["new_boundary"], "reason": edit["reason"]}
            for edit, old in zip(self.adjusted_plan["groups"][0]["boundary_adjustments"], (80, 140))
        ])
        for population in ("image", "pose"):
            audit = self.adjusted["separation_audit"]["populations"][population]
            for key in ("minimum_unseen_moment_training_gap_reference_frames",
                        "minimum_validation_test_gap_reference_frames",
                        "minimum_same_camera_cross_partition_gap_reference_frames"):
                self.assertGreater(audit[key], self.plan["buffer_frames"])

    def test_counts_balance_and_buffer_accounting_follow_the_new_membership(self):
        by_image = {row["image"]: row for row in self.records}
        for population, prefix in (("image", ""), ("pose", "pose_")):
            assignments = self.adjusted[f"{prefix}assignments"]
            self.assertEqual(set(assignments), set(by_image))
            self.assertEqual(self.adjusted[f"{prefix}counts"], dict(Counter(assignments.values())))
            for part in ("train", "val", "test"):
                counts = Counter(LABEL_MAP_10[by_image[image]["position"]]
                                 for image, assigned in assignments.items() if assigned == part)
                self.assertEqual(self.adjusted[f"{prefix}class_counts"][part],
                                 {label: counts[label] for label in CLASS_NAMES_10})
            for section in self.adjusted["sections"]:
                expected = Counter()
                for image, assigned in assignments.items():
                    if assigned == "excluded" or self.adjusted["section_ids"][image] != section["id"]:
                        continue
                    category = assigned
                    if assigned in ("val", "test"):
                        suffix = "view" if self.adjusted["evaluation_types"][image] == "held_out_view" else "moment"
                        category = f"{assigned}_{suffix}"
                    expected[category] += 1
                self.assertEqual(section["counts_by_population_category"][population],
                                 {category: expected[category] for category in
                                  ("train", "val_view", "val_moment", "test_view", "test_moment")})
            audit = self.adjusted["balance_audit"]["populations"][population]
            retained = sum(self.adjusted[f"{prefix}counts"][part] for part in ("train", "val", "test"))
            for part in ("train", "val", "test"):
                self.assertAlmostEqual(audit["overall_fractions"][part],
                                       self.adjusted[f"{prefix}counts"][part] / retained)
        buffered = [by_image[image] for image, reason in self.adjusted["exclusion_reasons"].items()
                    if reason == "temporal_buffer"]
        complete = sum(both_poses_present(row) for row in buffered)
        self.assertEqual(self.adjusted["pose_buffer_audit"], {
            "annotated_images_in_buffers": len(buffered),
            "already_ineligible_missing_athlete": len(buffered) - complete,
            "otherwise_eligible_pose_samples": complete,
        })

    def test_loader_rebuilds_adjusted_manifest_and_rejects_audit_or_boundary_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps(self.records), encoding="utf-8")
            manifest = copy.deepcopy(self.adjusted)
            manifest["annotation_sha256"] = sha256_file(annotations)
            path = root / "split.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(read_manifest(path, annotations, self.records), manifest)
            for field in ("boundary_adjustment_audit", "sections"):
                modified = copy.deepcopy(manifest)
                if field == "boundary_adjustment_audit":
                    modified[field][0]["new_boundary"] += 1
                else:
                    modified[field][1]["start"] += 1
                path.write_text(json.dumps(modified), encoding="utf-8")
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                    read_manifest(path, annotations, self.records)

    def test_supported_thirds_are_preserved_and_recounted_from_adjusted_membership(self):
        records, plan = fixture(width=120)
        original = build_multiview_manifest(records, "a" * 64, section_plan=plan)
        plan["groups"][0]["boundary_adjustments"] = [
            {"before_section": "section-01", "new_boundary": 137, "reason": "Review start."},
            {"before_section": "section-02", "new_boundary": 257, "reason": "Review end."},
        ]
        adjusted = build_multiview_manifest(records, "a" * 64, section_plan=plan)
        clocks = plan["groups"][0]["alignment"]
        labeled_times = []
        for row in records:
            clock = clocks[row["image"][:2]]
            labeled_times.append((row["image"], LABEL_MAP_10[row["position"]],
                                  (row["frame"] - clock["offset"]) / clock["scale"]))
        ranges = {label: (min(time for _, actual, time in labeled_times if actual == label),
                          max(time for _, actual, time in labeled_times if actual == label))
                  for label in CLASS_NAMES_10}
        for field, prefix in (("coverage", ""), ("pose_coverage", "pose_")):
            before = original["distribution_audit"][field]
            after = adjusted["distribution_audit"][field]
            self.assertTrue(before)
            support = lambda rows: {(r["group_index"], r["class"], r["third"]) for r in rows}
            self.assertEqual(support(after), support(before))
            counts = Counter()
            for image, label, time in labeled_times:
                if adjusted[f"{prefix}assignments"][image] not in ("val", "test"):
                    continue
                first, last = ranges[label]
                third = ("early", "middle", "late")[min(2, int(3 * (time - first) / (last - first)))]
                counts[label, third] += 1
            for row in after:
                self.assertEqual(row["evaluation_samples"], counts[row["class"], row["third"]])

    def test_large_moves_cannot_silently_remove_required_class_coverage(self):
        plan = copy.deepcopy(self.plan)
        sections = self.original["sections"]
        plan["groups"][0]["boundary_adjustments"] = [
            {"before_section": sections[index + 1]["id"], "new_boundary": section["start"] + 1,
             "reason": "Synthetic coverage failure: shrink every unseen validation section."}
            for index, section in enumerate(sections[:-1])
            if section["partition"] == "val" and section["evaluation_type"] == "unseen_moment"
        ]
        self.assertTrue(plan["groups"][0]["boundary_adjustments"])
        with self.assertRaisesRegex(ValueError, "insufficient .* class coverage"):
            build_multiview_manifest(self.records, "a" * 64, section_plan=plan)

    def test_invalid_boundary_edits_are_rejected(self):
        valid = {"before_section": "section-01", "new_boundary": 77, "reason": "Visual label review."}
        invalid = [
            None, {}, [None],
            [{**valid, "new_boundary": float("nan")}],
            [{**valid, "new_boundary": float("inf")}],
            [{**valid, "new_boundary": True}],
            [{**valid, "new_boundary": "77"}],
            [{**valid, "new_boundary": 20}],
            [{**valid, "new_boundary": 140}],
            [{**valid, "before_section": "missing"}],
            [{**valid, "before_section": "section-00"}],
            [{**valid, "before_section": 1}],
            [{key: value for key, value in valid.items() if key != "reason"}],
            [{**valid, "reason": "   "}],
            [{**valid, "reason": 5}],
            [{**valid, "unexpected": True}],
            [valid, {**valid, "new_boundary": 76}],
            [{**valid, "new_boundary": 130},
             {**valid, "before_section": "section-02", "new_boundary": 100}],
        ]
        for edits in invalid:
            plan = copy.deepcopy(self.plan)
            plan["groups"][0]["boundary_adjustments"] = edits
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                build_multiview_manifest(self.records, "a" * 64, section_plan=plan)


if __name__ == "__main__":
    unittest.main()
