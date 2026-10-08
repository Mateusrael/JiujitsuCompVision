"""Deterministic section allocation with separate view and moment holdouts.

The plan's clock mappings are estimates from annotation landmarks, not proof of
exact camera synchronization. All buffers and paired-view checks use that shared
reference clock. No predictions or model scores enter allocation.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import random

import numpy as np

from src.Dataset.annotations import (CLASS_NAMES_10, LABEL_MAP_10, both_poses_present,
                                      record_errors)

DEFAULT_PLAN = Path(__file__).resolve().parents[2] / "docs/multiview_sections.json"
KINDS = ("held_out_view", "unseen_moment")
CATEGORIES = ("train", "val_view", "val_moment", "test_view", "test_moment")
ALGORITHM = "section-coordinate-search-v2"
DEFAULT_MAX_CLASS_DEVIATION = .04


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_plan(plan, records, digest):
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("Section plan must have schema_version 1")
    if plan.get("annotation_sha256") not in (None, digest):
        raise ValueError("Section plan does not match the annotation SHA-256")
    fractions = plan.get("target_fractions")
    if (not isinstance(fractions, list) or len(fractions) != 3 or
            not all(_number(x) and x > 0 for x in fractions) or
            not math.isclose(sum(fractions), 1, abs_tol=1e-9)):
        raise ValueError("Section target_fractions must be three positive fractions summing to 1")
    deviation = plan.get("max_class_deviation", DEFAULT_MAX_CLASS_DEVIATION)
    if not _number(deviation) or not 0 < deviation < 1:
        raise ValueError("Section max_class_deviation must be a finite fraction in (0, 1)")
    for key, minimum in (("buffer_frames", 0), ("paired_view_tolerance_frames", 0),
                         ("min_samples_per_class", 1), ("seed", 0)):
        if not isinstance(plan.get(key), int) or isinstance(plan[key], bool) or plan[key] < minimum:
            raise ValueError(f"Section {key} must be an integer >= {minimum}")
    if not isinstance(plan.get("groups"), list) or not plan["groups"]:
        raise ValueError("Section plan requires nonempty groups")
    observed = {r["image"][:2] for r in records}
    prefixes, ids, group_ids = set(), set(), set()
    for group in plan["groups"]:
        if not isinstance(group, dict) or not isinstance(group.get("id"), str) or group["id"] in group_ids:
            raise ValueError("Groups require unique string IDs")
        group_ids.add(group["id"])
        sources = group.get("sources")
        if (not isinstance(sources, list) or len(sources) < 2
                or any(not isinstance(s, str) or len(s) != 2 or not s.isdigit() for s in sources)
                or len(sources) != len(set(sources))
                or prefixes.intersection(sources)):
            raise ValueError("Every camera source must occur in exactly one multi-view group")
        prefixes.update(sources)
        alignment = group.get("alignment")
        if not isinstance(alignment, dict) or set(alignment) != set(sources):
            raise ValueError("Alignment must cover exactly the group's sources")
        for mapping in alignment.values():
            if (not isinstance(mapping, dict) or not _number(mapping.get("scale"))
                    or mapping["scale"] <= 0 or not _number(mapping.get("offset"))
                    or not isinstance(mapping.get("evidence"), str) or not mapping["evidence"].strip()):
                raise ValueError("Alignment requires a positive scale, finite offset and evidence")
        sections = group.get("sections")
        if not isinstance(sections, list) or not sections:
            raise ValueError("Groups require nonempty sections")
        previous = None
        for section in sections:
            if (not isinstance(section, dict) or not isinstance(section.get("id"), str)
                    or section["id"] in ids or not _number(section.get("start"))
                    or not _number(section.get("end")) or section["start"] >= section["end"]):
                raise ValueError("Sections require unique IDs and finite nonempty intervals")
            if previous is not None and not math.isclose(previous, section["start"], abs_tol=1e-8):
                raise ValueError("Sections must be ordered, contiguous and nonoverlapping")
            previous = section["end"]
            ids.add(section["id"])
    if prefixes != observed:
        raise ValueError("Section sources must cover every observed prefix exactly")


def _prepare(records, plan):
    """Index records once; preserve local and estimated common-clock coordinates."""
    class_index = {name: i for i, name in enumerate(CLASS_NAMES_10)}
    blocks, by_source = [], {}
    for group_index, group in enumerate(plan["groups"]):
        first = len(blocks)
        for section_index, section in enumerate(group["sections"]):
            block = {**section, "group": group_index, "sources": group["sources"], "rows": [],
                     "previous": len(blocks) - 1 if section_index else None,
                     "next": len(blocks) + 1 if section_index + 1 < len(group["sections"]) else None}
            blocks.append(block)
        for source in group["sources"]:
            by_source[source] = (group, first, [s["start"] for s in group["sections"]])
    for index, row in enumerate(records):
        source = row["image"][:2]
        group, first, starts = by_source[source]
        clock = group["alignment"][source]
        time = (row["frame"] - clock["offset"]) / clock["scale"]
        section = bisect_right(starts, time) - 1
        if section < 0 or time >= group["sections"][section]["end"]:
            raise ValueError(f"Image {row['image']} is outside the section plan")
        blocks[first + section]["rows"].append((index, time, source,
                                                class_index[LABEL_MAP_10[row["position"]]],
                                                both_poses_present(row)))
    for block in blocks:
        block["rows"].sort(key=lambda row: (row[1], row[2]))
        block["actions"] = [(0, None), (2, None), (4, None)] + [
            (category, source) for category in (1, 3) for source in block["sources"]]
    return blocks


def _has_neighbor(times, value, tolerance):
    index = bisect_left(times, value - tolerance)
    return index < len(times) and times[index] <= value + tolerance


def _option(block, action, trim, plan, *, assignments=False, spread=None, spread_size=0):
    """One block/action has fixed eligibility once its two boundary states are known."""
    category, held = block["actions"][action]
    radius = plan["buffer_frames"] / 2
    lower = block["start"] + radius if trim & 1 else -math.inf
    upper = block["end"] - radius if trim & 2 else math.inf
    interior = [r for r in block["rows"] if lower < r[1] < upper]
    paired = [defaultdict(list), defaultdict(list)]
    if held is not None:
        for _, time, source, label, complete in interior:
            if source != held:
                paired[0][label].append(time)
                if complete:
                    paired[1][label].append(time)
        for population in paired:
            for times in population.values():
                times.sort()
    counts = np.zeros((2, 5, len(CLASS_NAMES_10)), dtype=np.int64)
    spread_counts = np.zeros(spread_size, dtype=np.int64)
    image_assignments, pose_assignments, reasons, pose_reasons = {}, {}, {}, {}
    for index, time, source, label, complete in block["rows"]:
        image_category = pose_category = -1
        reason = pose_reason = None
        if not lower < time < upper:
            reason = pose_reason = "temporal_buffer"
        else:
            image_category = category if held is None or source == held else 0
            if held == source and not _has_neighbor(paired[0][label], time, plan["paired_view_tolerance_frames"]):
                image_category = -1
                reason = "no_paired_training_view"
            pose_category = image_category
            if not complete:
                pose_category = -1
                pose_reason = "missing_pose"
            elif image_category < 0:
                pose_reason = reason
            elif held == source and not _has_neighbor(paired[1][label], time, plan["paired_view_tolerance_frames"]):
                pose_category = -1
                pose_reason = "no_paired_training_view"
        if image_category >= 0:
            counts[0, image_category, label] += 1
            if image_category > 0 and spread is not None and index in spread:
                spread_counts[spread[index][0]] += 1
        if pose_category >= 0:
            counts[1, pose_category, label] += 1
            if pose_category > 0 and spread is not None and index in spread:
                pose_slot = spread[index][1]
                if pose_slot is not None:
                    spread_counts[pose_slot] += 1
        if assignments:
            image_assignments[index], pose_assignments[index] = image_category, pose_category
            if reason:
                reasons[index] = reason
            if pose_reason:
                pose_reasons[index] = pose_reason
    if assignments:
        return counts, image_assignments, pose_assignments, reasons, pose_reasons
    return (counts, spread_counts) if spread is not None else counts


def _spread_spec(blocks):
    """Both populations use the same image-clock thirds, where data supports them."""
    grouped = defaultdict(list)
    block_counts = Counter()
    for block in blocks:
        counts = Counter(row[3] for row in block["rows"])
        for label, count in counts.items():
            if count >= 20:
                block_counts[block["group"], label] += 1
        for row in block["rows"]:
            grouped[block["group"], row[3]].append(row)
    mapping, slots = {}, []
    for (group, label), rows in sorted(grouped.items()):
        if len(rows) < 600 or block_counts[group, label] < 6:
            continue
        first, last = min(row[1] for row in rows), max(row[1] for row in rows)
        thirds = [min(2, int(3 * (row[1] - first) / max(last - first, 1))) for row in rows]
        if any(thirds.count(third) < 20 for third in range(3)):
            # A discontinuous label can be absent throughout a temporal third.
            # Allocation cannot create evaluation observations in that gap.
            continue
        base = len(slots)
        for third in range(3):
            slots.append({"group_index": group, "class": CLASS_NAMES_10[label],
                          "third": ("early", "middle", "late")[third],
                          "population": "image"})
        pose_slots = {}
        for third in range(3):
            complete = sum(row[4] and row_third == third for row, row_third in zip(rows, thirds))
            if complete >= 20:
                pose_slots[third] = len(slots)
                slots.append({"group_index": group, "class": CLASS_NAMES_10[label],
                              "third": ("early", "middle", "late")[third],
                              "population": "pose"})
        for row, third in zip(rows, thirds):
            mapping[row[0]] = (base + third, pose_slots.get(third) if row[4] else None)
    return mapping, slots


def _trim(blocks, choices, i):
    block = blocks[i]
    active = choices[i] != 0
    left = block["previous"]
    right = block["next"]
    return (int(left is not None and (active or choices[left] != 0)) +
            2 * int(right is not None and (active or choices[right] != 0)))


def _allocate(blocks, plan):
    """Count-based deterministic search, balancing image and complete-pose populations."""
    spread_map, spread_slots = _spread_spec(blocks)
    tables, spread_tables = [], []
    for block in blocks:
        values = [[_option(block, action, trim, plan, spread=spread_map, spread_size=len(spread_slots))
                   for trim in range(4)] for action in range(len(block["actions"]))]
        tables.append(np.array([[v[0] for v in row] for row in values]))
        spread_tables.append(np.array([[v[1] for v in row] for row in values]))
    options = []
    for b, table in zip(blocks, tables):
        # Reject choices with no evaluation examples even before boundary trimming.
        options.append([a for a in range(len(b["actions"])) if a == 0 or
                        np.all(table[a, 0, :, b["actions"][a][0]].sum(axis=-1) > 0)])
    fractions = plan["target_fractions"]
    pooled_target = np.array(fractions)[None, :, None]
    class_tolerance = plan.get("max_class_deviation", DEFAULT_MAX_CLASS_DEVIATION)
    target = np.array([fractions[0], fractions[1] / 2, fractions[1] / 2,
                       fractions[2] / 2, fractions[2] / 2])
    minimum = plan["min_samples_per_class"]
    raw = sum((table[0, 0] for table in tables), start=np.zeros((2, 5, 10), dtype=np.int64))
    if np.any(raw[:, 0] < minimum * 5):
        raise ValueError("Insufficient image/two-pose class coverage for five training/evaluation subsets")
    camera_slots = [(g, source) for g, group in enumerate(plan["groups"]) for source in group["sources"]]
    slot_index = {key: i for i, key in enumerate(camera_slots)}

    def rotations(choices):
        slots = np.zeros(len(camera_slots), dtype=np.int64)
        for i, action in enumerate(choices):
            held = blocks[i]["actions"][action][1]
            if held is not None:
                slots[slot_index[(blocks[i]["group"], held)]] += 1
        return slots

    def objective(total, slots, spread):
        sums = total.sum(axis=2)
        denominator = sums.sum(axis=1)
        if np.any(denominator == 0):
            return 1e9
        share = sums / denominator[:, None]
        coverage = np.maximum(minimum - total, 0) / minimum
        class_denominator = total.sum(axis=1)
        class_share = total / np.maximum(class_denominator[:, None, :], 1)
        class_pooled = np.stack((class_share[:, 0], class_share[:, 1:3].sum(axis=1),
                                 class_share[:, 3:5].sum(axis=1)), axis=1)
        class_error = class_pooled - pooled_target
        # Give every class equal influence in each population, and penalize the
        # worst deviations beyond 2.5 points. Retention also matters: additional
        # buffered frames are a cost, even when they make ratios slightly closer.
        return float(150 * np.square(share - target).sum() +
                     150 * np.square(class_error).sum() / len(CLASS_NAMES_10) +
                     30 * np.square(class_share - target[None, :, None]).sum() / len(CLASS_NAMES_10) +
                     300 * np.square(np.maximum(np.abs(class_error) - .025, 0)).sum() +
                     30 * np.square(coverage).sum() + 4 * np.count_nonzero(slots == 0) +
                     4 * np.square(np.maximum(20 - spread, 0) / 20).sum() +
                     (1 - denominator / raw.sum(axis=(1, 2))).sum())

    best = None
    rng = random.Random(plan["seed"])
    for restart in range(6):
        choices = [0 if restart == 0 or rng.random() < .6 else rng.choice(allowed)
                   for allowed in options]
        contributions = [tables[i][a, _trim(blocks, choices, i)] for i, a in enumerate(choices)]
        spread_contributions = [spread_tables[i][a, _trim(blocks, choices, i)] for i, a in enumerate(choices)]
        total = sum(contributions, start=np.zeros((2, 5, 10), dtype=np.int64))
        spread_total = sum(spread_contributions, start=np.zeros(len(spread_slots), dtype=np.int64))
        slots = rotations(choices)
        score = objective(total, slots, spread_total)
        for sweep in range(25):
            changed = False
            order = list(range(len(blocks)))
            rng.shuffle(order)
            for i in order:
                original = choices[i]
                affected = [j for j in (blocks[i]["previous"], i, blocks[i]["next"]) if j is not None]
                base = total - sum((contributions[j] for j in affected),
                                   start=np.zeros((2, 5, 10), dtype=np.int64))
                spread_base = spread_total - sum((spread_contributions[j] for j in affected),
                                                 start=np.zeros(len(spread_slots), dtype=np.int64))
                local_best = (score, original, total, slots, None, spread_total, None)
                for action in options[i]:
                    if action == original:
                        continue
                    choices[i] = action
                    updates = [tables[j][choices[j], _trim(blocks, choices, j)] for j in affected]
                    candidate = base + sum(updates, start=np.zeros((2, 5, 10), dtype=np.int64))
                    spread_updates = [spread_tables[j][choices[j], _trim(blocks, choices, j)] for j in affected]
                    candidate_spread = spread_base + sum(spread_updates, start=np.zeros(len(spread_slots), dtype=np.int64))
                    candidate_slots = slots.copy()
                    for old_action, delta in ((original, -1), (action, 1)):
                        held = blocks[i]["actions"][old_action][1]
                        if held is not None:
                            candidate_slots[slot_index[(blocks[i]["group"], held)]] += delta
                    candidate_score = objective(candidate, candidate_slots, candidate_spread)
                    if candidate_score < local_best[0] - 1e-12:
                        local_best = (candidate_score, action, candidate, candidate_slots, updates, candidate_spread, spread_updates)
                score, choices[i], total, slots, updates, spread_total, spread_updates = local_best
                if choices[i] != original:
                    changed = True
                    for j, update in zip(affected, updates):
                        contributions[j] = update
                    for j, update in zip(affected, spread_updates):
                        spread_contributions[j] = update
            if not changed:
                # Moving an evaluation role to another section can require two
                # simultaneous changes: neither single move need improve balance.
                # Swap whole section roles within a camera group, preserving the
                # view rotation while recomputing all affected boundary buffers.
                pair_best = None
                for i in range(len(blocks)):
                    for j in range(i + 1, len(blocks)):
                        a, b = choices[i], choices[j]
                        if (blocks[i]["group"] != blocks[j]["group"] or a == b
                                or b not in options[i] or a not in options[j]):
                            continue
                        affected = sorted({k for index in (i, j)
                                           for k in (blocks[index]["previous"], index,
                                                     blocks[index]["next"]) if k is not None})
                        candidate = total - sum((contributions[k] for k in affected),
                                                start=np.zeros((2, 5, 10), dtype=np.int64))
                        candidate_spread = spread_total - sum((spread_contributions[k] for k in affected),
                                                              start=np.zeros(len(spread_slots), dtype=np.int64))
                        choices[i], choices[j] = b, a
                        updates = [tables[k][choices[k], _trim(blocks, choices, k)] for k in affected]
                        spread_updates = [spread_tables[k][choices[k], _trim(blocks, choices, k)] for k in affected]
                        choices[i], choices[j] = a, b
                        candidate += sum(updates, start=np.zeros((2, 5, 10), dtype=np.int64))
                        candidate_spread += sum(spread_updates, start=np.zeros(len(spread_slots), dtype=np.int64))
                        candidate_score = objective(candidate, slots, candidate_spread)
                        if candidate_score < (score if pair_best is None else pair_best[0]) - 1e-12:
                            pair_best = (candidate_score, i, j, candidate, candidate_spread,
                                         affected, updates, spread_updates)
                if pair_best is None:
                    break
                score, i, j, total, spread_total, affected, updates, spread_updates = pair_best
                choices[i], choices[j] = choices[j], choices[i]
                for k, update, spread_update in zip(affected, updates, spread_updates):
                    contributions[k], spread_contributions[k] = update, spread_update
        class_pooled = np.stack((total[:, 0], total[:, 1:3].sum(axis=1),
                                 total[:, 3:5].sum(axis=1)), axis=1)
        class_error = class_pooled / np.maximum(class_pooled.sum(axis=1, keepdims=True), 1) - pooled_target
        pooled = class_pooled.sum(axis=2)
        overall_error = pooled / pooled.sum(axis=1, keepdims=True) - fractions
        feasible = (np.all(total >= minimum) and np.all(rotations(choices) > 0)
                    and np.all(spread_total >= 20) and np.all(np.abs(overall_error) <= .025)
                    and np.all(np.abs(class_error) <= class_tolerance))
        # Prefer a fully valid restart even when an invalid one has a lower soft score.
        candidate = (not feasible, round(score, 12), tuple(choices), total.copy(), spread_total.copy())
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    _, score, choices, total, spread_total = best
    if np.any(total < minimum):
        shortages = [(population, CATEGORIES[category], CLASS_NAMES_10[label], int(total[population, category, label]))
                     for population, category, label in zip(*np.where(total < minimum))]
        raise ValueError(f"Section plan cannot meet image/two-pose class coverage: {shortages}")
    if any(rotations(choices) == 0):
        raise ValueError("Section allocation did not rotate the held-out view across all sources")
    if np.any(spread_total < 20):
        shortages = [{**slot, "evaluation_samples": int(count)}
                     for slot, count in zip(spread_slots, spread_total) if count < 20]
        raise ValueError("Section allocation could not distribute image/pose evaluation "
                         f"throughout supported class thirds: {shortages}")
    pooled = np.stack([total[:, 0].sum(axis=1), total[:, 1:3].sum(axis=(1, 2)),
                       total[:, 3:5].sum(axis=(1, 2))], axis=1)
    shares = pooled / pooled.sum(axis=1, keepdims=True)
    if np.any(np.abs(shares - fractions) > .025):
        raise ValueError(f"Whole-section allocation is too far from target fractions: {shares.tolist()}")
    class_pooled = np.stack((total[:, 0], total[:, 1:3].sum(axis=1),
                             total[:, 3:5].sum(axis=1)), axis=1)
    class_error = class_pooled / class_pooled.sum(axis=1, keepdims=True) - pooled_target
    if np.any(np.abs(class_error) > class_tolerance):
        raise ValueError("Whole-section allocation exceeds max_class_deviation: "
                         f"{100 * np.abs(class_error).max():.2f} percentage points "
                         f"(allowed {100 * class_tolerance:.2f}). Revise the candidate sections.")
    distribution = [{**slot, "evaluation_samples": int(count)} for slot, count in zip(spread_slots, spread_total)]
    return choices, total, score, distribution


def _partition(category):
    return "excluded" if category < 0 else "train" if category == 0 else "val" if category < 3 else "test"


def _kind(category):
    return "held_out_view" if category in (1, 3) else "unseen_moment"


def _population_report(assignments, types, records):
    counts = dict.fromkeys(("train", "val", "test", "excluded"), 0)
    classes = {p: dict.fromkeys(CLASS_NAMES_10, 0) for p in ("train", "val", "test")}
    eval_counts = {p: dict.fromkeys(KINDS, 0) for p in ("val", "test")}
    eval_classes = {p: {kind: dict.fromkeys(CLASS_NAMES_10, 0) for kind in KINDS} for p in ("val", "test")}
    for row in records:
        image = row["image"]
        part = assignments[image]
        counts[part] += 1
        if part != "excluded":
            label = LABEL_MAP_10[row["position"]]
            classes[part][label] += 1
            if part != "train":
                eval_counts[part][types[image]] += 1
                eval_classes[part][types[image]][label] += 1
    return {"counts": counts, "class_counts": classes, "evaluation_counts": eval_counts,
            "evaluation_class_counts": eval_classes}


def _balance_report(report, plan):
    """Audit each class independently; frequent classes cannot mask rare-class drift."""
    partitions = ("train", "val", "test")
    target = np.array(plan["target_fractions"])
    populations = {}
    for population, prefix in (("image", ""), ("pose", "pose_")):
        counts = np.array([[report[prefix + "class_counts"][part][label]
                            for part in partitions] for label in CLASS_NAMES_10])
        fractions = counts / counts.sum(axis=1, keepdims=True)
        errors = 100 * (fractions - target)
        pooled = counts.sum(axis=0)
        populations[population] = {
            "overall_fractions": dict(zip(partitions, (pooled / pooled.sum()).tolist())),
            "class_fractions": {label: dict(zip(partitions, values.tolist()))
                                for label, values in zip(CLASS_NAMES_10, fractions)},
            "maximum_class_deviation_percentage_points": float(np.abs(errors).max()),
            "rms_class_deviation_percentage_points": float(np.sqrt(np.square(errors).mean())),
        }
    return {
        "denominator": "Retained examples of each class, separately for images and both-present poses.",
        "target_fractions": dict(zip(partitions, target.tolist())),
        "maximum_allowed_class_deviation_percentage_points": 100 * plan.get(
            "max_class_deviation", DEFAULT_MAX_CLASS_DEVIATION),
        "populations": populations,
    }


def _audit_separation(records, plan, assignments, pose_assignments, evaluation_types):
    """Check final memberships independently of candidate-option bookkeeping."""
    clocks = {source: (group["id"], group["alignment"][source])
              for group in plan["groups"] for source in group["sources"]}
    group_sources = {group["id"]: group["sources"] for group in plan["groups"]}
    indexed = []
    for row in records:
        image, source = row["image"], row["image"][:2]
        group, clock = clocks[source]
        time = (row["frame"] - clock["offset"]) / clock["scale"]
        indexed.append((image, source, group, time, LABEL_MAP_10[row["position"]]))

    def nearest(times, time):
        index = bisect_left(times, time)
        return min((abs(times[i] - time) for i in (index - 1, index)
                    if 0 <= i < len(times)), default=math.inf)

    def finite_or_none(value):
        return value if math.isfinite(value) else None

    gap = plan["buffer_frames"]
    tolerance = plan["paired_view_tolerance_frames"]
    populations = {}
    for population, membership in (("image", assignments), ("pose", pose_assignments)):
        times, witnesses, by_source = defaultdict(list), defaultdict(list), defaultdict(list)
        for image, source, group, time, label in indexed:
            part = membership[image]
            if part == "excluded":
                continue
            times[group, part].append(time)
            by_source[source].append((time, part))
            if part == "train":
                witnesses[group, source, label].append(time)
        for values in (*times.values(), *witnesses.values()):
            values.sort()
        held_counts = dict.fromkeys(sorted(clocks), 0)
        minimum_unseen = minimum_val_test = minimum_camera = math.inf
        maximum_witness = 0.0
        for image, source, group, time, label in indexed:
            part = membership[image]
            if part not in ("val", "test"):
                continue
            opposite = "test" if part == "val" else "val"
            distance = nearest(times[group, opposite], time)
            minimum_val_test = min(minimum_val_test, distance)
            if distance <= gap:
                raise ValueError(f"{population} validation/test temporal isolation failed: {image}")
            if evaluation_types[image] == "unseen_moment":
                distance = nearest(times[group, "train"], time)
                minimum_unseen = min(minimum_unseen, distance)
                if distance <= gap:
                    raise ValueError(f"{population} unseen-moment training isolation failed: {image}")
            else:
                distance = min(nearest(witnesses[group, other, label], time)
                               for other in group_sources[group] if other != source)
                if distance > tolerance + 1e-9:
                    raise ValueError(f"{population} held-out view has no paired training view: {image}")
                maximum_witness = max(maximum_witness, distance)
                held_counts[source] += 1
        for source, values in by_source.items():
            values.sort()
            for (left, from_part), (right, to_part) in zip(values, values[1:]):
                if from_part != to_part:
                    distance = right - left
                    minimum_camera = min(minimum_camera, distance)
                    if distance <= gap:
                        raise ValueError(f"{population} same-camera temporal isolation failed: {source}")
        missing = [source for source, count in held_counts.items() if not count]
        if missing:
            raise ValueError(f"{population} held-out rotation has no retained samples for sources: {missing}")
        populations[population] = {
            "minimum_unseen_moment_training_gap_reference_frames": finite_or_none(minimum_unseen),
            "minimum_validation_test_gap_reference_frames": finite_or_none(minimum_val_test),
            "minimum_same_camera_cross_partition_gap_reference_frames": finite_or_none(minimum_camera),
            "maximum_nearest_paired_training_distance_reference_frames": maximum_witness,
            "held_out_view_samples_by_source": held_counts,
        }
    return {"required_gap_reference_frames": gap,
            "paired_view_tolerance_reference_frames": tolerance,
            "clock": "Estimated reference frame coordinates; not independent synchronization evidence.",
            "populations": populations}


def build_multiview_manifest(records, annotation_sha256, *, section_plan=None):
    """Allocate complete sections once and audit both image and two-pose cohorts."""
    if not records:
        raise ValueError("Annotations must not be empty")
    seen = set()
    for index, row in enumerate(records):
        errors = record_errors(row)
        if errors:
            raise ValueError(f"Annotation {index}: {'; '.join(errors)}")
        if int(row["image"][2:]) != row["frame"] or row["image"] in seen:
            raise ValueError("Duplicate image or image/frame mismatch")
        seen.add(row["image"])
    records = sorted(records, key=lambda row: row["image"])
    plan = deepcopy(section_plan) if section_plan is not None else json.loads(DEFAULT_PLAN.read_text(encoding="utf-8"))
    _validate_plan(plan, records, annotation_sha256)
    blocks = _prepare(records, plan)
    choices, expected, objective, distribution = _allocate(blocks, plan)
    assignments, pose_assignments, types, reasons, pose_reasons, section_ids = {}, {}, {}, {}, {}, {}
    sections = []
    actual = np.zeros_like(expected)
    for i, (block, action) in enumerate(zip(blocks, choices)):
        trim = _trim(blocks, choices, i)
        counts, image_values, pose_values, image_reasons, pose_values_reasons = _option(block, action, trim, plan, assignments=True)
        actual += counts
        category, held = block["actions"][action]
        group = plan["groups"][block["group"]]
        sections.append({"id": block["id"], "group": group["id"], "start": block["start"], "end": block["end"],
                         "partition": _partition(category), "evaluation_type": _kind(category) if category else None,
                         "held_out_source": held, "buffer_left": bool(trim & 1), "buffer_right": bool(trim & 2),
                         "counts_by_population_category": {pop: {name: int(counts[p, c].sum()) for c, name in enumerate(CATEGORIES)}
                                                           for p, pop in enumerate(("image", "pose"))}})
        for index, category in image_values.items():
            image = records[index]["image"]
            assignments[image] = _partition(category)
            pose_assignments[image] = _partition(pose_values[index])
            section_ids[image] = block["id"]
            if category > 0:
                types[image] = _kind(category)
            if index in image_reasons:
                reasons[image] = image_reasons[index]
            if index in pose_values_reasons:
                pose_reasons[image] = pose_values_reasons[index]
    if not np.array_equal(actual, expected) or len(assignments) != len(records):
        raise ValueError("Section allocation audit failed")
    if any(part != "excluded" and part != assignments[image] for image, part in pose_assignments.items()):
        raise ValueError("Pose assignments must be a subset of image assignments")
    report = {"schema_version": 3, "split_method": "multiview-sections", "annotation_sha256": annotation_sha256,
              "class_names": list(CLASS_NAMES_10), "label_map": dict(LABEL_MAP_10), "section_plan": plan,
              "allocation": {"algorithm": ALGORITHM, "seed": plan["seed"], "objective": objective,
                             "ratio_denominator": "Retained samples after buffers and population eligibility; balanced separately for images and poses."},
              "evaluation_scope": "Distributed sections within known recordings/athletes. Held-out views have another training camera within the estimated time tolerance; unseen moments withhold all angles. Camera alignment is estimated from annotation landmarks, not independently verified exact synchronization.",
              "assignments": dict(sorted(assignments.items())), "pose_assignments": dict(sorted(pose_assignments.items())),
              "evaluation_types": dict(sorted(types.items())), "section_ids": dict(sorted(section_ids.items())),
              "exclusion_reasons": dict(sorted(reasons.items())), "pose_exclusion_reasons": dict(sorted(pose_reasons.items())),
              "sections": sections, "pose_filter": "both_present"}
    distribution_by_population = {
        population: [{key: value for key, value in row.items() if key != "population"}
                     for row in distribution if row["population"] == population]
        for population in ("image", "pose")}
    report["distribution_audit"] = {"scope": "Image evaluation (validation and test together) in early/middle/late thirds of each substantial group/class stretch (>=600 annotations, >=6 sections with >=20 examples, and >=20 available annotations in every third).",
                                    "pose_scope": "Pose evaluation uses the same image-clock thirds. Each third with >=20 raw complete-pose observations must retain >=20 pose evaluation samples; unsupported pose thirds are omitted rather than assigned fabricated observations.",
                                    "minimum_per_third": 20,
                                    "coverage": distribution_by_population["image"],
                                    "pose_coverage": distribution_by_population["pose"]}
    report.update(_population_report(assignments, types, records))
    report.update({"pose_" + key: value for key, value in _population_report(pose_assignments, types, records).items()})
    report["balance_audit"] = _balance_report(report, plan)
    buffered = [row for row in records if reasons.get(row["image"]) == "temporal_buffer"]
    complete_in_buffer = sum(both_poses_present(row) for row in buffered)
    report["pose_buffer_audit"] = {
        "annotated_images_in_buffers": len(buffered),
        "already_ineligible_missing_athlete": len(buffered) - complete_in_buffer,
        "otherwise_eligible_pose_samples": complete_in_buffer,
    }
    report["separation_audit"] = _audit_separation(records, plan, assignments, pose_assignments, types)
    return report
