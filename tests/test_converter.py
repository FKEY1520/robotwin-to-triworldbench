"""Regression tests for instruction consistency and movable bundle paths.

Run from the converter directory with:
    .venv/Scripts/python.exe -m unittest discover -s tests -v

Fixtures are deliberately synthetic: tests do not read or modify RoboTwin data.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "convert_robotwin_to_triworld.py"
SPEC = importlib.util.spec_from_file_location("robotwin_converter_under_test", SCRIPT)
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class InstructionCandidateTests(unittest.TestCase):
    def test_seen_candidates_keep_order_original_words_and_provenance(self):
        first = "Use the right arm to lift the bottle."
        second = "Carefully lift the bottle using your left hand."
        candidates = converter.instruction_candidates({
            "seen": [first, second], "unseen": ["Move the bottle."],
        })
        self.assertEqual([item["text"] for item in candidates], [
            first, second, "Move the bottle.",
        ])
        self.assertTrue(all(isinstance(item["json_path"], str) for item in candidates))
        self.assertEqual(len({item["json_path"] for item in candidates}), 3)
        self.assertIn("seen", candidates[0]["json_path"])
        self.assertIn("unseen", candidates[-1]["json_path"])

    def test_empty_values_do_not_hide_later_instruction(self):
        candidates = converter.instruction_candidates({
            "seen": ["", "   ", None, {"instruction": "Lift with the left arm."}],
        })
        self.assertEqual([item["text"] for item in candidates], ["Lift with the left arm."])

    def test_plain_instruction_is_supported(self):
        candidates = converter.instruction_candidates({"instruction": "Move the bottle."})
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["text"], "Move the bottle.")


class InstructionArmTests(unittest.TestCase):
    def test_explicit_effector_names_are_recognized(self):
        examples = {
            "Use the LEFT ARM.": {"left"},
            "Use your right hand to lift it.": {"right"},
            "Close the left gripper.": {"left"},
            "Lift with the right arm and support with the left hand.": {"left", "right"},
            "Use both arms to lift the box.": {"left", "right"},
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(converter.instruction_arms(text), expected)

    def test_direction_words_are_not_effector_claims(self):
        for text in (
            "Move the bottle to the right side of the table.",
            "Place it on the left of the box.",
            "Turn the bottle right, then move it left.",
            "Lift the upright bottle carefully.",
        ):
            with self.subTest(text=text):
                self.assertEqual(converter.instruction_arms(text), set())


class InstructionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.right = "Use the right arm to lift the bottle."
        self.left = "Lift the bottle with your left arm."
        self.neutral = "Carefully lift the bottle and hold it upright."

    def choose(self, payload, arm="left", policy="consistent"):
        result = converter.select_instruction(payload, expected_arm=arm, policy=policy)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        text, audit = result
        self.assertIsInstance(audit, dict)
        return text, audit

    def test_consistent_first_instruction_is_preserved(self):
        text, _ = self.choose({"seen": [self.left, self.neutral]})
        self.assertEqual(text, self.left)

    def test_neutral_first_instruction_is_preserved(self):
        text, _ = self.choose({"seen": [self.neutral, self.left]})
        self.assertEqual(text, self.neutral)

    def test_conflict_selects_exact_original_matching_text(self):
        text, _ = self.choose({"seen": [self.right, self.left]})
        self.assertEqual(text, self.left)
        self.assertNotEqual(text, self.right.replace("right", "left"))

    def test_matching_effector_is_preferred_over_earlier_neutral(self):
        text, _ = self.choose({"seen": [self.right, self.neutral, self.left]})
        self.assertEqual(text, self.left)

    def test_neutral_used_when_no_explicitly_matching_candidate_exists(self):
        text, _ = self.choose({"seen": [self.right, self.neutral]})
        self.assertEqual(text, self.neutral)

    def test_spatial_right_is_neutral_even_for_left_arm(self):
        spatial = "Move the bottle to the right side of the tray."
        text, _ = self.choose({"seen": [self.right, spatial]})
        self.assertEqual(text, spatial)

    def test_right_arm_selection_is_symmetric(self):
        text, _ = self.choose({"seen": [self.left, self.right]}, arm="right")
        self.assertEqual(text, self.right)

    def test_unknown_arm_preserves_original_choice(self):
        text, _ = self.choose({"seen": [self.right, self.left]}, arm=None)
        self.assertEqual(text, self.right)

    def test_conflicting_seen_must_not_silently_select_unseen(self):
        with self.assertRaises(ValueError):
            self.choose({"seen": [self.right], "unseen": [self.left, self.neutral]})

    def test_neutral_seen_is_preferred_to_matching_unseen(self):
        text, _ = self.choose({
            "seen": [self.right, self.neutral], "unseen": [self.left],
        })
        self.assertEqual(text, self.neutral)

    def test_unseen_only_payload_can_select_within_its_original_split(self):
        text, _ = self.choose({"unseen": [self.right, self.left]})
        self.assertEqual(text, self.left)

    def test_unseen_only_conflict_fails_if_no_compatible_original_exists(self):
        with self.assertRaises(ValueError):
            self.choose({"unseen": [self.right]})

    def test_no_matching_original_text_fails_without_rewriting(self):
        payload = {"seen": [self.right]}
        with self.assertRaises(ValueError):
            self.choose(payload)
        self.assertEqual(payload, {"seen": [self.right]})

    def test_both_arms_is_not_neutral_for_single_arm_task(self):
        both = "Use both arms to lift the bottle."
        text, _ = self.choose({"seen": [self.right, both, self.left]})
        self.assertEqual(text, self.left)

    def test_strict_rejects_first_conflict_despite_later_match(self):
        with self.assertRaises(ValueError):
            self.choose({"seen": [self.right, self.left]}, policy="strict")

    def test_strict_accepts_first_neutral(self):
        text, _ = self.choose({"seen": [self.neutral, self.left]}, policy="strict")
        self.assertEqual(text, self.neutral)

    def test_first_keeps_conflict_and_emits_warning(self):
        text, audit = self.choose({"seen": [self.right, self.left]}, policy="first")
        self.assertEqual(text, self.right)
        warning_values = [value for key, value in audit.items() if "warning" in key.lower()]
        self.assertTrue(any(warning_values), "Retaining a known conflict must leave an audit warning")

    def test_missing_text_fails(self):
        with self.assertRaises(ValueError):
            self.choose({"seen": [None, "", " "]})

    def test_selection_does_not_mutate_source_payload(self):
        import copy
        payload = {"seen": [self.right, self.left], "unseen": [self.neutral]}
        original = copy.deepcopy(payload)
        self.choose(payload)
        self.assertEqual(payload, original)


class BundlePathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="robotwin-path-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def test_relative_path_resolves_inside_bundle(self):
        actual = converter.resolve_bundle_path(self.root, "test_dataset/data/episode1.hdf5")
        self.assertEqual(actual, self.root / "test_dataset" / "data" / "episode1.hdf5")

    def test_parent_traversal_is_rejected(self):
        for relative in ("../outside.json", "STATE/../../outside.json", "..\\outside.json"):
            with self.subTest(relative=relative):
                with self.assertRaises(ValueError):
                    converter.resolve_bundle_path(self.root, relative)

    def test_absolute_path_is_rejected_even_if_inside_bundle(self):
        for absolute in (self.root / "STATE" / "episode1.json", self.root.parent / "outside.json"):
            with self.subTest(absolute=absolute):
                with self.assertRaises(ValueError):
                    converter.resolve_bundle_path(self.root, str(absolute))

    def test_windows_absolute_and_unc_paths_are_rejected(self):
        for relative in ("C:\\other\\episode1.json", "\\\\server\\share\\episode1.json"):
            with self.subTest(relative=relative):
                with self.assertRaises(ValueError):
                    converter.resolve_bundle_path(self.root, relative)


class BundleRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="robotwin-repair-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.sandbox = Path(self.temporary.name).resolve()
        self.root = self.sandbox / "bundle_after_move"
        self.root.mkdir()
        self.mapping_path = self.root / "robotwin_episode_mapping.json"

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Original compact formatting checks that backup retains the exact bytes.
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def make_bundle(self, modern=True, episodes=1):
        old_root = self.sandbox / "bundle_before_move"
        original_root = self.sandbox / "legacy_original_sources"
        entries = []
        for index in range(1, episodes + 1):
            episode = f"episode{index}"
            relative_hdf5 = f"source_data/task/config/data/{episode}.hdf5"
            relative_instruction = f"source_data/task/config/instructions/{episode}.json"
            hdf5 = self.root / relative_hdf5 if modern else original_root / f"{episode}.hdf5"
            instruction = self.root / relative_instruction if modern else original_root / f"{episode}.json"
            hdf5.parent.mkdir(parents=True, exist_ok=True)
            hdf5.write_bytes(b"synthetic HDF5 sentinel: unchanged")
            self.write_json(instruction, {"seen": ["Lift using your left arm."], "unseen": ["Lift it."]})
            entry = {
                "new_episode": episode,
                "source_hdf5": str(old_root / relative_hdf5) if modern else str(hdf5),
                "source_instruction_json": str(old_root / relative_instruction) if modern else str(instruction),
                "instruction": "Lift using your left arm.",
                "frame_count": 3,
                "instruction_selection": {"changed": True, "original": "Lift using your right arm."},
            }
            if modern:
                entry["source_hdf5_rel"] = relative_hdf5
                entry["source_instruction_json_rel"] = relative_instruction
            entries.append(entry)
            self.write_json(self.root / "STATE" / f"{episode}.json", {
                "hdf5_path": entry["source_hdf5"],
                "output_json": str(old_root / "STATE" / f"{episode}.json"),
                "num_frames": 3,
                "phases": [{"start_frame": 0, "end_frame": 2, "label": "抬起"}],
            })
            self.write_json(self.root / "gt_dataset" / episode / f"{episode}.json", {
                "source_hdf5": entry["source_hdf5"],
                "source_instruction_json": entry["source_instruction_json"],
                "instruction": entry["instruction"],
                "source_episode": f"episode{index - 1}",
            })
        mapping = {
            "entries": entries,
            "source_root": str(old_root / "source_data") if modern else str(original_root),
            "target_root": str(old_root / "gt_dataset"),
            "task": "adjust_bottle",
        }
        if modern:
            mapping["source_root_rel"] = "source_data"
        self.write_json(self.mapping_path, mapping)
        image_path = self.root / "gt_dataset" / "episode1" / "head" / "frames" / "frame_00000.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"synthetic image sentinel: unchanged")
        return mapping

    def snapshot(self):
        return {path.relative_to(self.root): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def test_moved_bundle_updates_paths_and_preserves_annotations_and_binary_files(self):
        old_mapping = self.make_bundle()
        before = self.snapshot()
        report = converter.repair_bundle_paths(self.root)
        self.assertEqual(report["updated_json_files"], 3)
        backup = Path(report["backup_root"])
        self.assertTrue(backup.is_dir())
        mapping = self.read_json(self.mapping_path)
        self.assertEqual(mapping["target_root"], str(self.root / "gt_dataset"))
        self.assertEqual(mapping["source_root"], str(self.root / "source_data"))
        for key in ("instruction", "frame_count", "instruction_selection"):
            self.assertEqual(mapping["entries"][0][key], old_mapping["entries"][0][key])
        entry = mapping["entries"][0]
        self.assertEqual(entry["source_hdf5"], str(self.root / entry["source_hdf5_rel"]))
        self.assertEqual(entry["source_instruction_json"], str(self.root / entry["source_instruction_json_rel"]))
        state = self.read_json(self.root / "STATE" / "episode1.json")
        self.assertEqual(state["hdf5_path"], entry["source_hdf5"])
        self.assertEqual(state["output_json"], str(self.root / "STATE" / "episode1.json"))
        self.assertEqual(state["phases"], [{"start_frame": 0, "end_frame": 2, "label": "抬起"}])
        gt = self.read_json(self.root / "gt_dataset" / "episode1" / "episode1.json")
        self.assertEqual(gt["source_hdf5"], entry["source_hdf5"])
        self.assertEqual(gt["source_instruction_json"], entry["source_instruction_json"])
        self.assertEqual(gt["instruction"], entry["instruction"])
        for relative, content in before.items():
            if relative in (
                Path("robotwin_episode_mapping.json"),
                Path("STATE/episode1.json"),
                Path("gt_dataset/episode1/episode1.json"),
            ):
                self.assertEqual((backup / relative).read_bytes(), content)
            else:
                self.assertEqual((self.root / relative).read_bytes(), content)

    def test_legacy_bundle_keeps_valid_external_source_paths(self):
        old_mapping = self.make_bundle(modern=False)
        source_path = Path(old_mapping["entries"][0]["source_hdf5"])
        before_source = source_path.read_bytes()
        converter.repair_bundle_paths(self.root)
        mapping = self.read_json(self.mapping_path)
        self.assertEqual(mapping["source_root"], old_mapping["source_root"])
        for key in ("source_hdf5", "source_instruction_json"):
            self.assertEqual(mapping["entries"][0][key], old_mapping["entries"][0][key])
        self.assertEqual(source_path.read_bytes(), before_source)
        self.assertEqual(mapping["target_root"], str(self.root / "gt_dataset"))

    def test_second_episode_missing_source_prevents_all_writes(self):
        mapping = self.make_bundle(episodes=2)
        (self.root / mapping["entries"][1]["source_hdf5_rel"]).unlink()
        before = self.snapshot()
        with self.assertRaises(FileNotFoundError):
            converter.repair_bundle_paths(self.root)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.root / "path_repair_backups").exists())

    def test_relative_source_path_cannot_escape_bundle(self):
        mapping = self.make_bundle()
        external_source = self.sandbox / "external.hdf5"
        external_source.write_bytes(b"external source must stay untouched")
        mapping["entries"][0]["source_hdf5_rel"] = "../external.hdf5"
        self.write_json(self.mapping_path, mapping)
        before = self.snapshot()
        with self.assertRaises(ValueError):
            converter.repair_bundle_paths(self.root)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(external_source.read_bytes(), b"external source must stay untouched")

    def test_second_repair_is_noop_and_creates_no_extra_backup(self):
        self.make_bundle()
        converter.repair_bundle_paths(self.root)
        before = self.snapshot()
        report = converter.repair_bundle_paths(self.root)
        self.assertEqual(report["updated_json_files"], 0)
        self.assertIsNone(report["backup_root"])
        self.assertEqual(self.snapshot(), before)

    def test_transient_replace_failure_rolls_back_cleans_temporary_files_and_allows_retry(self):
        original_replace = Path.replace
        test_root = self.root
        for failure_at in (1, 2):
            with self.subTest(failure_at=failure_at):
                self.root = test_root / f"failure_at_{failure_at}"
                self.root.mkdir()
                self.mapping_path = self.root / "robotwin_episode_mapping.json"
                self.make_bundle()
                before = self.snapshot()
                replace_calls = 0

                def locked_replace(path, target):
                    nonlocal replace_calls
                    replace_calls += 1
                    if replace_calls == failure_at:
                        raise OSError("Simulated transient Windows file lock")
                    return original_replace(path, target)

                with mock.patch.object(Path, "replace", new=locked_replace):
                    with self.assertRaisesRegex(OSError, "transient Windows file lock"):
                        converter.repair_bundle_paths(self.root)
                self.assertEqual(replace_calls, failure_at)
                after = {relative: content for relative, content in self.snapshot().items()
                         if relative.parts[0] != "path_repair_backups"}
                self.assertEqual(after, before, "Partial JSON replacements must roll back exactly")
                self.assertEqual(list(self.root.rglob("*.rebase-tmp")), [])
                report = converter.repair_bundle_paths(self.root)
                self.assertEqual(report["updated_json_files"], 3)
                self.assertEqual(list(self.root.rglob("*.rebase-tmp")), [])
                state = self.read_json(self.root / "STATE" / "episode1.json")
                self.assertEqual(state["output_json"], str(self.root / "STATE" / "episode1.json"))


if __name__ == "__main__":
    unittest.main()
