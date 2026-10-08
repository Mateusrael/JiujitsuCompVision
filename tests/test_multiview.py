"""Check whole-section isolation and paired-view eligibility on offset cameras."""

import copy
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import tempfile
import unittest

from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10, both_poses_present, sha256_file
from src.Dataset.multiview import build_multiview_manifest
from src.Loader.splits import filter_pose_records, read_manifest, split_records


def fixture(*, width=60):
    """Three cameras share moments despite different local clocks and missing data."""
    raw_labels = {label: next(raw for raw, normalized in LABEL_MAP_10.items() if normalized == label)
                  for label in CLASS_NAMES_10}
    mappings = {"00": (1, 0), "01": (2, 7), "02": (1, 37)}
    start, blocks = 20, 30
    pose = [[float(j), float(j + 1), .8] for j in range(17)]
    zero_pose = [[float(j), float(j + 1), 0.] for j in range(17)]
    records = []
    for source, (scale, offset) in mappings.items():
        for moment in range(start, start + width * blocks):
            if source == "02" and moment >= start + width * (blocks - 3):
                continue  # This recording ends earlier; no invented paired view.
            if moment % 31 == 0 and source != "02":
                continue  # The remaining camera has no released counterpart here.
            frame = moment * scale + offset
            pose2 = pose
            if moment % 7 == int(source) or (moment % 17 == 0 and source != "02"):
                pose2 = None
            elif moment % 29 == int(source):
                pose2 = zero_pose
            label_index = (moment - start) % 10
            if moment % 19 == 0 and source != "00":
                label_index = (label_index + 1) % 10  # Matching time alone is insufficient.
            records.append({"image": f"{source}{frame:05d}", "frame": frame,
                            "position": raw_labels[CLASS_NAMES_10[label_index]],
                            "pose1": pose, "pose2": pose2})
    plan = {
        "schema_version": 1, "target_fractions": [.8, .1, .1], "seed": 42,
        "buffer_frames": 10, "paired_view_tolerance_frames": 0, "min_samples_per_class": 1,
        "groups": [{"id": "fixture-recording", "sources": list(mappings),
                    "alignment": {source: {"scale": scale, "offset": offset,
                                           "evidence": "Exact synthetic frame-clock transformation."}
                                  for source, (scale, offset) in mappings.items()},
                    "sections": [{"id": f"section-{index:02d}",
                                  "start": start + width * index,
                                  "end": start + width * (index + 1)}
                                 for index in range(blocks)]}],
    }
    return records, plan


def imbalanced_fixture():
    """Global fractions can look right while rare classes and pose cohorts drift."""
    raw_labels = {label: next(raw for raw, normalized in LABEL_MAP_10.items() if normalized == label)
                  for label in CLASS_NAMES_10}
    pose = [[float(j), float(j + 1), .8] for j in range(17)]
    clocks = {"00": 0, "01": 7, "02": 37}
    records, sections = [], []
    for block in range(90):
        featured_class = 1 + block % 9
        # One background observation of every minority survives the boundary buffers.
        # Most examples belong to class 0; rare-class frequencies vary by section.
        labels = ([0] * 5 + [value for label in range(1, 10) for value in (label, 0, 0, 0)]
                  + [featured_class] * 12 + [0] * 7)
        start = 20 + 60 * block
        sections.append({"id": f"imbalanced-{block:02d}", "start": start, "end": start + 60})
        for source, offset in clocks.items():
            for index, label in enumerate(labels):
                frame = start + index + offset
                missing = index >= 41 and label == featured_class and (block // 9 + label) % 3 == 0
                records.append({"image": f"{source}{frame:05d}", "frame": frame,
                                "position": raw_labels[CLASS_NAMES_10[label]],
                                "pose1": pose, "pose2": None if missing else pose})
    plan = {
        "schema_version": 1, "target_fractions": [.8, .1, .1], "seed": 42,
        "buffer_frames": 4, "paired_view_tolerance_frames": 0, "min_samples_per_class": 1,
        "groups": [{"id": "imbalanced-recording", "sources": list(clocks),
                    "alignment": {source: {"scale": 1, "offset": offset,
                                           "evidence": "Exact synthetic frame-clock offset."}
                                  for source, offset in clocks.items()}, "sections": sections}],
    }
    return records, plan


class MultiviewSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.plan = fixture()
        cls.manifest = build_multiview_manifest(cls.records, "a" * 64, section_plan=cls.plan)

    def test_both_populations_have_conserved_membership_coverage_and_target_shares(self):
        manifest = self.manifest
        for prefix in ("", "pose_"):
            assignments = manifest[f"{prefix}assignments"]
            self.assertEqual(set(assignments), {row["image"] for row in self.records})
            self.assertEqual(dict(Counter(assignments.values())), manifest[f"{prefix}counts"])
            counts = manifest[f"{prefix}counts"]
            retained = sum(counts[part] for part in ("train", "val", "test"))
            for part, target in zip(("train", "val", "test"), (.8, .1, .1)):
                self.assertLessEqual(abs(counts[part] / retained - target), .025)
                self.assertTrue(all(n >= 1 for n in manifest[f"{prefix}class_counts"][part].values()))
            for part in ("val", "test"):
                self.assertEqual(sum(manifest[f"{prefix}evaluation_counts"][part].values()), counts[part])
                for kind in ("held_out_view", "unseen_moment"):
                    self.assertTrue(all(n >= 1 for n in
                                        manifest[f"{prefix}evaluation_class_counts"][part][kind].values()))
        for row in self.records:
            image = row["image"]
            pose_part = manifest["pose_assignments"][image]
            if pose_part != "excluded":
                self.assertTrue(both_poses_present(row))
                self.assertEqual(pose_part, manifest["assignments"][image])
            elif not both_poses_present(row):
                self.assertIn(image, manifest["pose_exclusion_reasons"])
        self.assertLess(sum(manifest["pose_counts"][part] for part in ("train", "val", "test")),
                        sum(manifest["counts"][part] for part in ("train", "val", "test")))

    def test_each_retained_held_out_view_has_a_same_class_training_witness(self):
        manifest = self.manifest
        clocks = self.plan["groups"][0]["alignment"]
        times = {row["image"]: (row["frame"] - clocks[row["image"][:2]]["offset"]) /
                 clocks[row["image"][:2]]["scale"] for row in self.records}
        by_id = {row["image"]: row for row in self.records}
        for prefix in ("", "pose_"):
            assignments = manifest[f"{prefix}assignments"]
            witnesses = defaultdict(list)
            for image, part in assignments.items():
                if part == "train":
                    key = (manifest["section_ids"][image], LABEL_MAP_10[by_id[image]["position"]])
                    witnesses[key].append((times[image], image[:2]))
            held_sources = set()
            for image, part in assignments.items():
                if part not in ("val", "test") or manifest["evaluation_types"][image] != "held_out_view":
                    continue
                held_sources.add(image[:2])
                key = (manifest["section_ids"][image], LABEL_MAP_10[by_id[image]["position"]])
                self.assertTrue(any(source != image[:2] and
                                    abs(time - times[image]) <= self.plan["paired_view_tolerance_frames"]
                                    for time, source in witnesses[key]), image)
            self.assertEqual(held_sources, {"00", "01", "02"})
        self.assertIn("no_paired_training_view", manifest["exclusion_reasons"].values())
        self.assertIn("no_paired_training_view", manifest["pose_exclusion_reasons"].values())

    def test_unseen_sections_and_buffers_withhold_all_cameras_in_shared_time(self):
        manifest = self.manifest
        sections = {section["id"]: section for section in manifest["sections"]}
        clocks = self.plan["groups"][0]["alignment"]
        radius = self.plan["buffer_frames"] / 2
        buffered_sources = set()
        buffered_count = complete_in_buffer = 0
        for row in self.records:
            image = row["image"]
            source = image[:2]
            mapping = clocks[source]
            time = (row["frame"] - mapping["offset"]) / mapping["scale"]
            section = sections[manifest["section_ids"][image]]
            buffered = ((section["buffer_left"] and time <= section["start"] + radius) or
                        (section["buffer_right"] and time >= section["end"] - radius))
            if buffered:
                buffered_sources.add(source)
                buffered_count += 1
                complete_in_buffer += both_poses_present(row)
                self.assertEqual(manifest["assignments"][image], "excluded")
                self.assertEqual(manifest["pose_assignments"][image], "excluded")
                self.assertEqual(manifest["exclusion_reasons"][image], "temporal_buffer")
            elif section["evaluation_type"] == "unseen_moment":
                self.assertEqual(manifest["assignments"][image], section["partition"])
                self.assertNotEqual(manifest["pose_assignments"][image], "train")
            elif section["evaluation_type"] == "held_out_view" and source != section["held_out_source"]:
                self.assertEqual(manifest["assignments"][image], "train")
        self.assertEqual(buffered_sources, {"00", "01", "02"})
        self.assertGreater(complete_in_buffer, 0)
        self.assertLess(complete_in_buffer, buffered_count)
        self.assertEqual(manifest["pose_buffer_audit"], {
            "annotated_images_in_buffers": buffered_count,
            "already_ineligible_missing_athlete": buffered_count - complete_in_buffer,
            "otherwise_eligible_pose_samples": complete_in_buffer,
        })

    def test_separation_audit_reports_retained_gaps_and_actual_camera_rotation(self):
        manifest = self.manifest
        audit = manifest["separation_audit"]
        self.assertEqual(audit["required_gap_reference_frames"], self.plan["buffer_frames"])
        self.assertEqual(audit["paired_view_tolerance_reference_frames"],
                         self.plan["paired_view_tolerance_frames"])
        for population, prefix in (("image", ""), ("pose", "pose_")):
            reported = audit["populations"][population]
            held_counts = Counter(image[:2] for image, part in manifest[f"{prefix}assignments"].items()
                                  if part in ("val", "test") and
                                  manifest["evaluation_types"][image] == "held_out_view")
            self.assertEqual(reported["held_out_view_samples_by_source"], dict(held_counts))
            self.assertEqual(set(held_counts), {"00", "01", "02"})
            self.assertEqual(reported["maximum_nearest_paired_training_distance_reference_frames"], 0)
            for field in ("minimum_unseen_moment_training_gap_reference_frames",
                          "minimum_validation_test_gap_reference_frames",
                          "minimum_same_camera_cross_partition_gap_reference_frames"):
                self.assertGreater(reported[field], self.plan["buffer_frames"])

    def test_evaluation_spreads_across_real_thirds_without_inventing_labels_in_gaps(self):
        records, plan = fixture(width=120)
        clocks = plan["groups"][0]["alignment"]

        def moment(row):
            mapping = clocks[row["image"][:2]]
            return (row["frame"] - mapping["offset"]) / mapping["scale"]

        # Leave >600 examples in many blocks but remove one label's entire middle third.
        # Its absence must not make a valid plan impossible, or fabricate coverage there.
        gapped_label = CLASS_NAMES_10[0]
        times = [moment(row) for row in records if LABEL_MAP_10[row["position"]] == gapped_label]
        first, last = min(times), max(times)
        replacement = next(raw for raw, label in LABEL_MAP_10.items() if label == CLASS_NAMES_10[1])
        for row in records:
            third = min(2, int(3 * (moment(row) - first) / (last - first)))
            if LABEL_MAP_10[row["position"]] == gapped_label and third == 1:
                row["position"] = replacement
        label_rows = defaultdict(list)
        for row in records:
            label_rows[LABEL_MAP_10[row["position"]]].append(row)
        self.assertGreaterEqual(len(label_rows[gapped_label]), 600)
        # One class has no supplied second athlete in its middle third. Another
        # lacks poses in its early third, so redefining thirds on pose-only time
        # ranges would give a wrong answer even though the image labels continue.
        pose_gaps = {CLASS_NAMES_10[2]: 1, CLASS_NAMES_10[3]: 0}
        for label, missing_third in pose_gaps.items():
            rows = label_rows[label]
            first, last = min(moment(row) for row in rows), max(moment(row) for row in rows)
            for row in rows:
                third = min(2, int(3 * (moment(row) - first) / (last - first)))
                if third == missing_third:
                    row["pose2"] = None
        manifest = build_multiview_manifest(records, "a" * 64, section_plan=plan)
        audit = manifest["distribution_audit"]
        self.assertEqual(audit["minimum_per_third"], 20)
        expected, expected_pose = {}, {}
        for label, rows in label_rows.items():
            first, last = min(moment(row) for row in rows), max(moment(row) for row in rows)
            raw_counts, evaluation_counts = Counter(), Counter()
            raw_pose_counts, pose_evaluation_counts = Counter(), Counter()
            for row in rows:
                third = min(2, int(3 * (moment(row) - first) / (last - first)))
                raw_counts[third] += 1
                raw_pose_counts[third] += both_poses_present(row)
                if manifest["assignments"][row["image"]] in ("val", "test"):
                    evaluation_counts[third] += 1
                if manifest["pose_assignments"][row["image"]] in ("val", "test"):
                    pose_evaluation_counts[third] += 1
            if label == gapped_label:
                self.assertEqual(raw_counts[1], 0)
                continue
            for third, name in enumerate(("early", "middle", "late")):
                self.assertGreaterEqual(raw_counts[third], 20)
                self.assertGreaterEqual(evaluation_counts[third], 20)
                expected[label, name] = evaluation_counts[third]
                if pose_gaps.get(label) == third:
                    self.assertEqual(raw_pose_counts[third], 0)
                    self.assertEqual(pose_evaluation_counts[third], 0)
                else:
                    self.assertGreaterEqual(raw_pose_counts[third], 20)
                    self.assertGreaterEqual(pose_evaluation_counts[third], 20)
                    expected_pose[label, name] = pose_evaluation_counts[third]
        reported = {(row["class"], row["third"]): row["evaluation_samples"]
                    for row in audit["coverage"]}
        self.assertEqual(reported, expected)
        self.assertEqual(len(reported), 3 * (len(CLASS_NAMES_10) - 1))
        self.assertTrue(all(row["group_index"] == 0 for row in audit["coverage"]))
        reported_pose = {(row["class"], row["third"]): row["evaluation_samples"]
                         for row in audit["pose_coverage"]}
        self.assertEqual(reported_pose, expected_pose)
        self.assertEqual(len(reported_pose), len(reported) - len(pose_gaps))
        self.assertTrue(all(row["group_index"] == 0 for row in audit["pose_coverage"]))

    def test_versioned_section_plan_is_in_the_source_bundle(self):
        from src.Scripts.bundle_project import source_files

        root = Path(__file__).resolve().parents[1]
        included = {path.relative_to(root).as_posix() for path in source_files(root)}
        self.assertIn("docs/multiview_sections.json", included)

    def test_per_class_balance_is_not_hidden_by_a_dominant_class_or_missing_poses(self):
        records, plan = imbalanced_fixture()
        raw = Counter(LABEL_MAP_10[row["position"]] for row in records)
        paired = Counter(LABEL_MAP_10[row["position"]] for row in records if both_poses_present(row))
        self.assertGreater(max(raw.values()), 10 * min(raw.values()))
        self.assertGreater(len(set(paired.values())), 2)
        manifest = build_multiview_manifest(records, "a" * 64, section_plan=plan)
        audit = manifest["balance_audit"]
        targets = dict(zip(("train", "val", "test"), plan["target_fractions"]))
        self.assertEqual(audit["target_fractions"], targets)
        self.assertEqual(audit["maximum_allowed_class_deviation_percentage_points"], 4)
        for population, prefix in (("image", ""), ("pose", "pose_")):
            reported = audit["populations"][population]
            counts = manifest[f"{prefix}counts"]
            retained = sum(counts[part] for part in ("train", "val", "test"))
            for part, target in zip(("train", "val", "test"), (.8, .1, .1)):
                self.assertLessEqual(abs(counts[part] / retained - target), .025)
                self.assertAlmostEqual(reported["overall_fractions"][part], counts[part] / retained)
            classes = manifest[f"{prefix}class_counts"]
            errors = []
            for label in CLASS_NAMES_10:
                total = sum(classes[part][label] for part in ("train", "val", "test"))
                for part, target in zip(("train", "val", "test"), (.8, .1, .1)):
                    with self.subTest(population=prefix or "image", label=label, partition=part):
                        fraction = classes[part][label] / total
                        self.assertLessEqual(abs(fraction - target), .04)
                        self.assertAlmostEqual(reported["class_fractions"][label][part], fraction)
                        errors.append(100 * (fraction - target))
            self.assertAlmostEqual(reported["maximum_class_deviation_percentage_points"],
                                   max(abs(error) for error in errors))
            self.assertAlmostEqual(reported["rms_class_deviation_percentage_points"],
                                   (sum(error * error for error in errors) / len(errors)) ** .5)

    def test_class_balance_tolerance_rejects_invalid_values(self):
        for tolerance in (0, -0.01, 1, float("nan"), float("inf"), True, ".04", None):
            with self.subTest(tolerance=tolerance), self.assertRaisesRegex(ValueError, "max_class_deviation"):
                build_multiview_manifest(self.records, "a" * 64,
                                         section_plan={**self.plan, "max_class_deviation": tolerance})

    def test_unattainable_class_balance_fails_instead_of_publishing_a_bad_split(self):
        strict_plan = {**self.plan, "max_class_deviation": 1e-6}
        with self.assertRaisesRegex(ValueError, "exceeds max_class_deviation"):
            build_multiview_manifest(self.records, "a" * 64, section_plan=strict_plan)

    def test_shuffling_records_cannot_change_assignments(self):
        shuffled = list(self.records)
        random.Random(118).shuffle(shuffled)
        result = build_multiview_manifest(shuffled, "a" * 64, section_plan=self.plan)
        self.assertEqual(result, self.manifest)

    def test_invalid_plan_coordinates_prefixes_and_digest_fail(self):
        overlap = copy.deepcopy(self.plan)
        overlap["groups"][0]["sections"][1]["start"] -= 1
        missing_source = copy.deepcopy(self.plan)
        missing_source["groups"][0]["sources"].remove("02")
        del missing_source["groups"][0]["alignment"]["02"]
        wrong_hash = {**self.plan, "annotation_sha256": "b" * 64}
        for plan, message in ((overlap, "contiguous and nonoverlapping"),
                              (missing_source, "every observed prefix"), (wrong_hash, "SHA-256")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                build_multiview_manifest(self.records, "a" * 64, section_plan=plan)

    def test_loader_verifies_derived_fields_and_filters_each_evaluation_type(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps(self.records), encoding="utf-8")
            manifest = copy.deepcopy(self.manifest)
            manifest["annotation_sha256"] = sha256_file(annotations)
            path = root / "split.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            checked = read_manifest(path, annotations, self.records)
            for part in ("val", "test"):
                for kind in ("held_out_view", "unseen_moment"):
                    selected = split_records(self.records, checked, part, evaluation_type=kind)
                    self.assertEqual(len(selected), checked["evaluation_counts"][part][kind])
                    self.assertEqual(len(filter_pose_records(selected, manifest=checked)),
                                     checked["pose_evaluation_counts"][part][kind])
            # Rebuilding must detect independently edited population accounting.
            modified = copy.deepcopy(manifest)
            modified["pose_counts"]["train"] += 1
            path.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pose_counts"):
                read_manifest(path, annotations, self.records)


if __name__ == "__main__":
    unittest.main()
