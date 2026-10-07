import copy
import random
import unittest

from src.Dataset.annotations import CLASS_NAMES_10, LABEL_MAP_10
from src.Dataset.temporal import DEFAULT_CLASS_SOURCES, build_temporal_manifest


def fixture(sources=None, count=240, spacing=1):
    sources = DEFAULT_CLASS_SOURCES if sources is None else sources
    records = []
    raw = {label: [key for key, value in LABEL_MAP_10.items() if value == label]
           for label in CLASS_NAMES_10}
    for index, label in enumerate(CLASS_NAMES_10):
        for number in range(count):
            frame = index * 5000 + number * spacing + 1
            records.append({"image": f"{sources[label]}{frame:05d}", "frame": frame,
                            "position": raw[label][number % len(raw[label])]})
    return records


class TemporalSplitTests(unittest.TestCase):
    def build(self, records=None, **kwargs):
        options = {"gap_frames": 10, "min_samples_per_class": 20, **kwargs}
        return build_temporal_manifest(fixture() if records is None else records,
                                       "a" * 64, **options)

    def test_conservation_role_mapping_missing_poses_and_excluded_views(self):
        records = fixture()
        alternate = [{**row, "image": "99" + row["image"][2:]} for row in records]
        manifest = self.build(records + alternate)
        self.assertEqual(sum(manifest["counts"].values()), len(records + alternate))
        self.assertEqual(manifest["label_map"], LABEL_MAP_10)
        self.assertTrue(all(manifest["assignments"][row["image"]] == "excluded"
                            for row in alternate))
        self.assertEqual(set(manifest["exclusion_reasons"].values()),
                         {"unselected_source", "temporal_gap"})
        self.assertTrue(all(n >= 20 for counts in manifest["class_counts"].values()
                            for n in counts.values()))
        self.assertGreater(manifest["separation_audit"]["minimum_cross_split_gap_frames"], 10)

    def test_input_order_invariance_including_single_shared_prefix(self):
        sources = dict.fromkeys(CLASS_NAMES_10, "00")
        records = fixture(sources)
        expected = self.build(records, class_sources=sources)
        random.Random(91).shuffle(records)
        self.assertEqual(self.build(records, class_sources=sources), expected)

    def test_gap_uses_raw_frame_distance_not_sample_ordinals(self):
        manifest = self.build(fixture(spacing=20))
        self.assertEqual(manifest["counts"]["excluded"], 0)
        self.assertEqual(manifest["separation_audit"]["minimum_cross_split_gap_frames"], 20)

    def test_cross_class_transition_is_also_purged(self):
        sources = dict.fromkeys(CLASS_NAMES_10, "00")
        records = fixture(sources)
        # Make adjacent classes contiguous: previous class test abuts next class train.
        for index, row in enumerate(records):
            row["frame"] = index + 1
            row["image"] = f"00{index + 1:05d}"
        manifest = self.build(records, class_sources=sources)
        for frame in range(236, 246):
            self.assertEqual(manifest["exclusion_reasons"][f"00{frame:05d}"], "temporal_gap")
        self.assertGreater(manifest["separation_audit"]["minimum_cross_split_gap_frames"], 10)

    def test_overlapping_embargoes_and_zero_gap(self):
        for gap in (0, 1, 9, 10, 34):
            with self.subTest(gap=gap):
                manifest = self.build(gap_frames=gap, min_samples_per_class=1)
                self.assertGreater(manifest["separation_audit"]["minimum_cross_split_gap_frames"], gap)
        with self.assertRaisesRegex(ValueError, "coverage after temporal embargo"):
            self.build(gap_frames=100)

    def test_interleaved_classes_retain_gap_after_overlapping_exclusions(self):
        sources = dict.fromkeys(CLASS_NAMES_10, "00")
        records = fixture(sources)
        random.Random(27).shuffle(records)
        for index, row in enumerate(records):
            row.update(frame=index + 1, image=f"00{index + 1:05d}")
        manifest = self.build(records, class_sources=sources, min_samples_per_class=1)
        retained = [row for row in records if manifest["assignments"][row["image"]] != "excluded"]
        for left, right in zip(retained, retained[1:]):
            if manifest["assignments"][left["image"]] != manifest["assignments"][right["image"]]:
                self.assertGreater(right["frame"] - left["frame"], 10)

    def test_default_gap_on_sufficiently_long_synthetic_sequences(self):
        manifest = build_temporal_manifest(fixture(count=2000), "a" * 64)
        self.assertGreater(manifest["separation_audit"]["minimum_cross_split_gap_frames"], 150)

    def test_bad_parameters_fail_without_fallback(self):
        changes = [{"gap_frames": -1}, {"gap_frames": True}, {"gap_frames": 1.2},
                   {"min_samples_per_class": 0}, {"min_samples_per_class": True},
                   {"fractions": (0.7, 0.1, 0.1)}, {"fractions": (0.7, 0, 0.3)},
                   {"fractions": (0.7, float("nan"), 0.3)},
                   {"fractions": (True, 0.1, 0.2)}, {"class_sources": {}},
                   {"class_sources": {**DEFAULT_CLASS_SOURCES, "back": "0"}},
                   {"class_sources": {**DEFAULT_CLASS_SOURCES, "back": "99"}}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build(**change)

    def test_invalid_records_and_empty_class_are_rejected(self):
        original = fixture()
        for field, value in (("frame", True), ("frame", 99999), ("position", "invalid"),
                             ("image", "000")):
            records = copy.deepcopy(original)
            records[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.build(records)
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            self.build(original + [original[0]])
        with self.assertRaisesRegex(ValueError, "nonempty chronological"):
            self.build(fixture(count=2))
        with self.assertRaisesRegex(ValueError, "only 0 samples"):
            self.build([r for r in original if r["position"] != "5050_guard"])


if __name__ == "__main__":
    unittest.main()
