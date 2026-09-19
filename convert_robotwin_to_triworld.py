#!/usr/bin/env python3
"""RoboTwin legacy HDF5 -> TriWorldBench-style custom validation bundle.

Python >= 3.10. Install: python -m pip install numpy h5py Pillow opencv-python
Start with --dry-run, then --limit 1; use --limit 0 for all source episodes.
The first conversion downloads two pinned, hash-checked official helpers.
Source files are read-only. Existing output directories are never overwritten.
Standard images remain JPEG for official compatibility; lossless PNGs and
complete source HDF5/instruction/scene metadata are preserved separately.
After moving a bundle, run --repair-paths NEW_BUNDLE_ROOT to refresh the absolute
paths required by official consumers, using the saved bundle-relative paths.

Verified reference (2026-09-16): TriWorldBench/Dataset/val_dataset.tar.gz;
test_dataset HDF5 has endpose + joint_action, first_frame uses episodeN_view.jpg.
Official GT/STATE contract:
https://github.com/TriWorldBench/TriWorldBench/blob/main/test_data/data_readme.md

This is a custom converter, not an official TriWorldBench script. It produces
custom data in the released layout, not the official validation split or VQA QA.
It does not normalize, shift, resample, or reinterpret the robot trajectories.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import shlex
import shutil
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import h5py
    import numpy as np
    from PIL import Image
except ImportError as exc:
    repair_args = "-m pip install --force-reinstall --only-binary=:all: numpy h5py Pillow opencv-python"
    if sys.platform == "win32":
        powershell_python = sys.executable.replace("'", "''")
        repair_commands = (
            f"PowerShell: & '{powershell_python}' {repair_args}\n"
            f'CMD: "{sys.executable}" {repair_args}'
        )
    else:
        repair_commands = f"{shlex.quote(sys.executable)} {repair_args}"
    raise SystemExit(
        f"Dependency import failed: {exc}\n"
        f"Python {sys.version.split()[0]}: {sys.executable}\n"
        "A package may be missing, damaged, or built for a different Python version.\n"
        "Recreating an existing virtual environment with another Python version can leave incompatible packages.\n"
        "'Requirement already satisfied' does not verify that binary extensions can be imported.\n"
        "Reinstall the dependencies using this environment's Python, then rerun conversion:\n"
        f"{repair_commands}"
    )

HELPERS = {
    "decode_image_bit": (
        "https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/"
        "bd5636810602b59c36c86dcf1bba42e5e15c1f3f/data/decode_image_bit.py",
        "90df99c52d3f127a09f5960680113f17e8b3e25b2e3cb9117abdeb249d7a337e",
    ),
    "segment_episode_phases": (
        "https://raw.githubusercontent.com/TriWorldBench/TriWorldBench/"
        "0d3603ec48df2448bd29d716c8de8cb6b45d3114/scripts/lib/segment_episode_phases.py",
        "c3ea93bc716710d19e2b7db9ee665ffbed7ca88eae566232487c0d6c6b26da25",
    ),
}
VIEWS = {
    "head": "observation/head_camera/rgb",
    "left": "observation/left_camera/rgb",
    "right": "observation/right_camera/rgb",
}
# Exact numeric schema observed in the released ALOHA validation sample.
FIELDS = {
    "endpose/left_endpose": (7,),
    "endpose/left_gripper": (),
    "endpose/right_endpose": (7,),
    "endpose/right_gripper": (),
    "joint_action/left_arm": (6,),
    "joint_action/left_gripper": (),
    "joint_action/right_arm": (6,),
    "joint_action/right_gripper": (),
    "joint_action/vector": (14,),
}
INSTRUCTION_KEYS = ("instruction", "prompt", "task_instruction", "text", "description",
                    "seen", "unseen", "instructions")
SCHEMA_VERSION = 2


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_helper(name):
    url, expected = HELPERS[name]
    cache = Path(__file__).resolve().parent / "_triworld_helpers"
    path = cache / f"{name}.py"
    if path.exists():
        content = path.read_bytes()
    else:
        print(f"[download] official helper: {name}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot download {name}: {exc}\nOpen this exact URL in your browser:\n"
                f"{url}\nSave its raw content as: {path}"
            ) from exc
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError(f"Helper checksum mismatch: {path}; use the pinned URL {url}")
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    spec = importlib.util.spec_from_file_location(f"twb_official_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def instruction_candidates(value, json_path="$"):
    """Enumerate original strings in preference order, retaining their provenance."""
    if isinstance(value, str):
        return [{"text": value, "json_path": json_path}] if value.strip() else []
    if isinstance(value, list):
        return [candidate for i, item in enumerate(value)
                for candidate in instruction_candidates(item, f"{json_path}[{i}]")]
    if isinstance(value, dict):
        return [candidate for key in INSTRUCTION_KEYS if key in value
                for candidate in instruction_candidates(value[key], f"{json_path}.{key}")]
    return []


def instruction_arms(text):
    arms = set(re.findall(
        r"\b(left|right)[\s-]+(?:(?:robotic|robot)[\s-]+)?(?:arms?|hands?|grippers?)\b",
        text.lower()))
    if re.search(r"\b(?:both|two)[\s-]+(?:(?:robotic|robot)[\s-]+)?(?:arms?|hands?|grippers?)\b",
                 text.lower()):
        arms.update(("left", "right"))
    return arms


def instruction_split(candidate):
    match = re.search(r"\.(seen|unseen)(?:\[|\.|$)", candidate["json_path"])
    return match.group(1) if match else "unspecified"


def select_instruction(payload, expected_arm, policy="consistent"):
    """Reject explicit arm conflicts; never rewrite text or silently switch splits."""
    if policy not in ("consistent", "strict", "first"):
        raise ValueError(f"Unknown instruction policy: {policy}")
    if expected_arm not in (None, "left", "right"):
        raise ValueError(f"Expected a single arm or None, got {expected_arm!r}")
    candidates = instruction_candidates(payload)
    if not candidates:
        raise ValueError("No non-empty instruction string found")
    first = candidates[0]
    audit = {"policy": policy, "expected_arm": expected_arm, "candidate_count": len(candidates),
             "original_first": first, "selected": first, "changed": False,
             "reason": "first_candidate_compatible", "warnings": []}
    if expected_arm is None:
        audit["reason"] = "arm_not_unambiguously_known"
        audit["warnings"].append("Arm not unambiguously known; instruction kept without arm correction.")
        return first["text"], audit
    eligible = [c for c in candidates if instruction_split(c) == instruction_split(first)]
    conflicts = [c for c in eligible if instruction_arms(c["text"]) - {expected_arm}]
    audit["conflicting_candidate_paths"] = [c["json_path"] for c in conflicts]
    if not instruction_arms(first["text"]) - {expected_arm}:
        return first["text"], audit
    warning = f"First instruction names another arm; trajectory expects {expected_arm}."
    audit["warnings"].append(warning)
    if policy == "first":
        audit["reason"] = "conflict_explicitly_kept"
        return first["text"], audit
    if policy == "strict":
        raise ValueError(warning + " Fix the source annotation or use --instruction-policy consistent.")
    matching = [c for c in eligible if instruction_arms(c["text"]) == {expected_arm}]
    neutral = [c for c in eligible if not instruction_arms(c["text"])]
    if not matching and not neutral:
        raise ValueError(warning + " No compatible original candidate in the same instruction split.")
    selected = (matching or neutral)[0]
    audit.update(selected=selected, changed=True,
                 reason="selected_matching_arm_original" if matching else "selected_arm_neutral_original")
    return selected["text"], audit


def scene_for_episode(source, episode):
    path = source / "scene_info.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a scene-info object: {path}")
    index = int(episode.removeprefix("episode"))
    scene = payload.get(f"episode_{index}", payload.get(episode))
    if scene is not None and not isinstance(scene, dict):
        raise ValueError(f"Invalid scene-info entry: {path}/{episode}")
    return scene


def arm_evidence(handle, scene):
    # Drive-target ranges, not tiny physical end-effector jitter on the idle arm.
    spans = {arm: float(np.ptp(np.column_stack((handle[f"joint_action/{arm}_arm"][:],
                                               handle[f"joint_action/{arm}_gripper"][:])), axis=0).max())
             for arm in ("left", "right")}
    moving = [arm for arm, span in spans.items() if span > 1e-4]
    info = (scene or {}).get("info", {})
    scene_arm = info.get("{a}") if isinstance(info, dict) else None
    scene_arm = scene_arm if scene_arm in ("left", "right") else None
    expected = moving[0] if len(moving) == 1 else None
    if scene_arm and expected and scene_arm != expected:
        raise ValueError(f"Scene arm {scene_arm} conflicts with moving arm {expected}; manual review required.")
    return {"expected_arm": expected, "scene_arm": scene_arm, "moving_arms": moving,
            "max_joint_or_gripper_span": spans, "motion_threshold": 1e-4}


def resolve_bundle_path(root, relative):
    """Resolve stored paths against the bundle, refusing traversal and absolute paths."""
    root = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise ValueError(f"Not a safe bundle-relative path: {relative}")
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Path escapes bundle root: {relative}")
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preserve_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = sha256_file(source)
    if sha256_file(destination) != digest:
        raise ValueError(f"Preserved copy checksum mismatch: {source}")
    return digest


def preflight(args):
    for label, value in (("task", args.task), ("config", args.config)):
        if Path(value).name != value or value in ("", ".", ".."):
            raise ValueError(f"--{label} must be a single directory name")
    source = (args.source_root / args.task / args.config).resolve()
    if not (source / "data").is_dir():
        raise FileNotFoundError(f"Expected source folder: {source / 'data'}")
    paths = [p for p in (source / "data").glob("episode*.hdf5")
             if re.fullmatch(r"episode\d+", p.stem)]
    paths.sort(key=lambda p: int(p.stem.removeprefix("episode")))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise ValueError("No episodeN.hdf5 files found. Check the root/config or source format.")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new output folder: {output}")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the source folder.")
    records = []
    for index, path in enumerate(paths, start=1):
        instruction_path = source / "instructions" / f"{path.stem}.json"
        if not instruction_path.is_file():
            raise FileNotFoundError(f"Missing matching instruction: {instruction_path}")
        instruction_payload = json.loads(instruction_path.read_text(encoding="utf-8-sig"))
        scene = scene_for_episode(source, path.stem)
        with h5py.File(path, "r") as f:
            for key in (*VIEWS.values(), *FIELDS):
                if key not in f:
                    raise KeyError(f"{path.name}: missing /{key}; source is not the expected legacy layout")
            t = len(f[next(iter(VIEWS.values()))])
            if not t:
                raise ValueError(f"{path.name}: empty episode")
            for key in VIEWS.values():
                if f[key].ndim < 1 or len(f[key]) != t:
                    raise ValueError(f"{path.name}: camera length mismatch at {key}")
            for key, shape in FIELDS.items():
                if f[key].shape != (t, *shape):
                    raise ValueError(f"{path.name}: /{key} shape={f[key].shape}; expected {(t, *shape)}")
                if not np.isfinite(f[key][:]).all():
                    raise ValueError(f"{path.name}: nonfinite values at {key}")
            vector = f["joint_action/vector"][:]
            joined = np.column_stack((f["joint_action/left_arm"][:], f["joint_action/left_gripper"][:],
                                      f["joint_action/right_arm"][:], f["joint_action/right_gripper"][:]))
            if not np.array_equal(vector, joined):
                raise ValueError(f"{path.name}: vector/component order mismatch")
            evidence = arm_evidence(f, scene)
        instruction, audit = select_instruction(instruction_payload, evidence["expected_arm"],
                                                 args.instruction_policy)
        source_relative = Path("source_data") / args.task / args.config
        record = {
            "new_episode": f"episode{index}", "task": args.task,
            "config_name": args.config, "source_episode": path.stem,
            "source_hdf5": str(path), "source_instruction_json": str(instruction_path),
            "instruction": instruction, "frame_count": t,
            "original_source_hdf5": str(path), "original_source_instruction_json": str(instruction_path),
            "source_hdf5_rel": (source_relative / "data" / path.name).as_posix(),
            "source_instruction_json_rel": (source_relative / "instructions" / instruction_path.name).as_posix(),
            "instruction_selection": audit, "arm_evidence": evidence,
            "scene_info": scene,
        }
        records.append(record)
        print(f"[plan] {record['new_episode']} <- {path.stem}; frames={t}", flush=True)
        if audit["changed"]:
            print(f"[instruction] {path.stem}: {audit['original_first']['json_path']} -> "
                  f"{audit['selected']['json_path']} ({audit['reason']})", flush=True)
        for warning in audit["warnings"]:
            print(f"[warning] {path.stem}: {warning}", flush=True)
    return output, records


def convert_episode(record, work, final, decode, segment, *, lossless_images=True, jpeg_quality=95):
    episode = record["new_episode"]
    t = record["frame_count"]
    gt = work / "gt_dataset" / episode
    test = work / "test_dataset"
    for folder in (test / "data", test / "first_frame", test / "instructions", gt):
        folder.mkdir(parents=True, exist_ok=True)
    source_hdf5 = Path(record["original_source_hdf5"])
    preserved_hdf5 = resolve_bundle_path(work, record["source_hdf5_rel"])
    record["source_hdf5_sha256"] = preserve_file(source_hdf5, preserved_hdf5)
    record["source_instruction_sha256"] = preserve_file(
        Path(record["original_source_instruction_json"]),
        resolve_bundle_path(work, record["source_instruction_json_rel"]))
    # Official consumers expect absolute paths. Relative companions allow safe rebasing.
    record["source_hdf5"] = str(resolve_bundle_path(final, record["source_hdf5_rel"]))
    record["source_instruction_json"] = str(resolve_bundle_path(final, record["source_instruction_json_rel"]))
    numeric_path = test / "data" / f"{episode}.hdf5"
    with h5py.File(preserved_hdf5, "r") as src, h5py.File(numeric_path, "x") as dst:
        for key in FIELDS:
            group, name = key.split("/")
            # HDF5 copy preserves values, dtype, and dataset attributes, without reinterpretation.
            src.copy(src[key], dst.require_group(group), name=name)
        sizes = {}
        for view, key in VIEWS.items():
            frames = gt / view / "frames"
            frames.mkdir(parents=True)
            lossless_frames = work / "lossless" / episode / view / "frames"
            if lossless_images:
                lossless_frames.mkdir(parents=True)
            first_shape = None
            for idx in range(t):
                rgb = np.asarray(decode(src[key][idx]))
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                    raise ValueError(f"Invalid RGB array: {episode}/{view}/{idx}: {rgb.shape}, {rgb.dtype}")
                if rgb.shape != (240, 320, 3):
                    raise ValueError(
                        f"{episode}/{view}/{idx}: expected 320x240 RGB for this validation layout, "
                        f"got HWC={rgb.shape}. No automatic resize is applied."
                    )
                if first_shape is None:
                    first_shape = rgb.shape
                if rgb.shape != first_shape:
                    raise ValueError(f"Image size changes inside {episode}/{view}")
                target = frames / f"frame_{idx:05d}.jpg"
                Image.fromarray(rgb).save(target, format="JPEG", quality=jpeg_quality)
                if lossless_images:
                    png = lossless_frames / f"frame_{idx:05d}.png"
                    Image.fromarray(rgb).save(png, format="PNG", compress_level=1)
                    with Image.open(png) as saved:
                        if not np.array_equal(rgb, np.asarray(saved)):
                            raise ValueError(f"Lossless image mismatch: {episode}/{view}/{idx}")
                if idx == 0:
                    shutil.copyfile(target, test / "first_frame" / f"{episode}_{view}.jpg")
            sizes[view] = list(first_shape)
    dump_json(test / "instructions" / f"{episode}.json", {"instruction": record["instruction"]})
    dump_json(gt / f"{episode}.json", {
        "instruction": record["instruction"], "task": record["task"], "episode": episode,
        "source_episode": record["source_episode"], "source_hdf5": record["source_hdf5"],
        "source_instruction_json": record["source_instruction_json"],
        "source_hdf5_rel": record["source_hdf5_rel"],
        "source_instruction_json_rel": record["source_instruction_json_rel"],
        "original_source_hdf5": record["original_source_hdf5"],
        "instruction_selection": record["instruction_selection"],
        "scene_info": record["scene_info"],
    })
    state = segment(preserved_hdf5, record["task"], episode,
                    record["source_episode"], record["instruction"], None)
    if state["num_frames"] != t:
        raise ValueError(f"STATE/GT length mismatch: {episode}")
    if record["scene_info"] is not None:
        state["scene_info"] = copy.deepcopy(record["scene_info"])
    state["hdf5_path"] = record["source_hdf5"]
    state["hdf5_path_rel"] = record["source_hdf5_rel"]
    state["output_json_rel"] = f"STATE/{episode}.json"
    state["output_json"] = str(final / "STATE" / f"{episode}.json")
    state["original_source_hdf5"] = record["original_source_hdf5"]
    state["instruction_selection"] = record["instruction_selection"]
    dump_json(work / "STATE" / f"{episode}.json", state)
    # Verify saved numerical output against source, rather than trusting file presence.
    with h5py.File(source_hdf5, "r") as src, h5py.File(numeric_path, "r") as dst:
        for key in FIELDS:
            if (src[key].shape != dst[key].shape or src[key].dtype != dst[key].dtype
                    or src[key][:].tobytes() != dst[key][:].tobytes()):
                raise ValueError(f"Numerical copy mismatch: {episode}/{key}")
    record["view_counts"] = dict.fromkeys(VIEWS, t)
    record["view_shapes_hwc"] = sizes
    record["lossless_frames_rel"] = f"lossless/{episode}" if lossless_images else None
    print(f"[ok] {episode}: 3 x {t} RGB frames, numeric HDF5, instruction, STATE", flush=True)


def repair_bundle_paths(bundle_root):
    """Refresh official absolute-path fields after a move; back up JSON before edits.

    Old bundles without source copies keep their existing, valid original source
    paths. This does not regenerate annotations or modify HDF5/image content.
    """
    root = Path(bundle_root).resolve()
    mapping_path = resolve_bundle_path(root, "robotwin_episode_mapping.json")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    entries = mapping.get("entries", [])
    if not entries:
        raise ValueError(f"No mapping entries: {mapping_path}")
    planned = []
    seen = set()
    for entry in entries:
        episode = entry.get("new_episode", "")
        if not re.fullmatch(r"episode[1-9]\d*", episode) or episode in seen:
            raise ValueError(f"Invalid or duplicate episode id: {episode!r}")
        seen.add(episode)
        for key in ("source_hdf5", "source_instruction_json"):
            relative = entry.get(key + "_rel")
            if relative:
                path = resolve_bundle_path(root, relative)
            else:
                path = Path(entry.get(key, ""))
                if not path.is_absolute():
                    raise ValueError(f"{episode}: legacy {key} must be an existing absolute path")
            if not path.is_file():
                raise FileNotFoundError(f"{episode}: cannot repair missing {key}: {path}")
            entry[key] = str(path.resolve())
        state_path = resolve_bundle_path(root, f"STATE/{episode}.json")
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        state["hdf5_path"] = entry["source_hdf5"]
        if entry.get("source_hdf5_rel"):
            state["hdf5_path_rel"] = entry["source_hdf5_rel"]
        state["output_json_rel"] = f"STATE/{episode}.json"
        state["output_json"] = str(state_path)
        planned.append((state_path, state))
        gt_path = resolve_bundle_path(root, f"gt_dataset/{episode}/{episode}.json")
        gt = json.loads(gt_path.read_text(encoding="utf-8-sig"))
        for key in ("source_hdf5", "source_instruction_json"):
            gt[key] = entry[key]
            if entry.get(key + "_rel"):
                gt[key + "_rel"] = entry[key + "_rel"]
        planned.append((gt_path, gt))
    mapping["target_root_rel"] = "gt_dataset"
    mapping["target_root"] = str(root / "gt_dataset")
    if mapping.get("source_root_rel"):
        mapping["source_root"] = str(resolve_bundle_path(root, mapping["source_root_rel"]))
    planned.append((mapping_path, mapping))
    # Validate every input before writing anything, including legacy bundles.
    changed = [(path, value) for path, value in planned
               if json.loads(path.read_text(encoding="utf-8-sig")) != value]
    if not changed:
        return {"bundle": str(root), "updated_json_files": 0, "backup_root": None}
    backup = resolve_bundle_path(root, "path_repair_backups") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup.mkdir(parents=True, exist_ok=False)
    for path, _ in changed:
        saved = backup / path.relative_to(root)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
    temporary_paths = []
    try:
        for path, value in changed:
            # Per-file atomic replacement; restore all saved JSON on failure.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                             dir=path.parent, prefix=f".{path.name}.",
                                             suffix=".rebase-tmp") as stream:
                temporary = Path(stream.name)
                temporary_paths.append(temporary)
                stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(path)
    except Exception:
        for path, _ in changed:
            shutil.copy2(backup / path.relative_to(root), path)
        raise
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return {"bundle": str(root), "updated_json_files": len(changed), "backup_root": str(backup)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=Path(r"P:\RoboTwin数据集\RoboTwin_Raw"))
    parser.add_argument("--task", default="adjust_bottle")
    parser.add_argument("--config", default="aloha-agilex_clean_50")
    parser.add_argument("--output-root", type=Path, default=Path(
        r"P:\RoboTwin数据集\RoboTwin_convert\adjust_bottle_convert\TriWorldBench_adjust_bottle_smoke_v2"))
    parser.add_argument("--limit", type=int, default=1, help="Default 1 for a smoke run; 0 converts all episodes")
    parser.add_argument("--dry-run", action="store_true", help="Check paths, instructions, keys, shapes, values; write nothing")
    parser.add_argument("--instruction-policy", choices=("consistent", "strict", "first"), default="consistent",
                        help="consistent selects an original same-split alternative on arm conflict; strict errors; first keeps and warns")
    parser.add_argument("--no-lossless-images", action="store_true",
                        help="Skip supplementary PNGs; complete original HDF5 is still preserved")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Quality of official-compatible JPEGs (1..100)")
    parser.add_argument("--repair-paths", type=Path, metavar="BUNDLE_ROOT",
                        help="Refresh mapping/STATE/GT absolute paths after moving a bundle; backs up JSON first")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in 1..100")
    if args.repair_paths and args.dry_run:
        parser.error("--repair-paths cannot be combined with --dry-run")
    work = None
    try:
        if args.repair_paths:
            print(json.dumps(repair_bundle_paths(args.repair_paths), ensure_ascii=False, indent=2))
            return 0
        output, records = preflight(args)
        print(f"[checked] episodes={len(records)}; output={output}", flush=True)
        if args.dry_run:
            print("Dry run passed. Image decoding and output creation have not run yet.")
            return 0
        decoder = load_helper("decode_image_bit").decode_image_bit
        segment = load_helper("segment_episode_phases").segment_hdf5
        output.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
        source = (args.source_root / args.task / args.config).resolve()
        source_relative = Path("source_data") / args.task / args.config
        preserved_metadata = {}
        for name in ("scene_info.json", "seed.txt", "config.json", "config.yaml", "config.yml"):
            path = source / name
            if path.is_file():
                relative = (source_relative / name).as_posix()
                preserved_metadata[relative] = preserve_file(path, resolve_bundle_path(work, relative))
        for record in records:
            convert_episode(record, work, output, decoder, segment,
                            lossless_images=not args.no_lossless_images, jpeg_quality=args.jpeg_quality)
        stamp = datetime.now(timezone.utc).isoformat()
        dump_json(work / "robotwin_episode_mapping.json", {
            "schema_version": SCHEMA_VERSION, "generated_at": stamp,
            "source_root": str(output / "source_data"), "source_root_rel": "source_data",
            "original_source_root": str(args.source_root.resolve()),
            "target_root": str(output / "gt_dataset"), "target_root_rel": "gt_dataset", "config_name": args.config,
            "relative_paths_base": "bundle_root (directory containing this mapping)",
            "after_move": "Run converter --repair-paths NEW_BUNDLE_ROOT before official evaluation",
            "start_index": 1, "entries": records,
        })
        dump_json(work / "STATE" / "manifest.json", {
            "generated_at": stamp, "written": len(records), "failed": 0,
            "records": [{"episode": r["new_episode"], "num_frames": r["frame_count"]} for r in records],
        })
        dump_json(work / "conversion_report.json", {
            "created_at": stamp, "episode_count": len(records), "format": "TriWorldBench-style custom bundle",
            "schema_version": SCHEMA_VERSION,
            "source_files_modified": False, "numeric_values_and_dtype_preserved": True,
            "image_resize": False, "frame_subsampling": False, "jpeg_quality": args.jpeg_quality,
            "standard_jpeg_lossless": False,
            "lossless_pngs": not args.no_lossless_images,
            "lossless_png_semantics": "Exact RGB pixels after official decoding; source JPEG was already lossy",
            "source_hdf5_and_instructions_preserved": True,
            "source_metadata_sha256": preserved_metadata,
            "source_items_not_copied": ["_traj_data", "video", "environment/venv"],
            "scene_info_episodes_preserved": sum(r["scene_info"] is not None for r in records),
            "instruction_policy": args.instruction_policy,
            "instruction_selection_changes": [r["new_episode"] for r in records if r["instruction_selection"]["changed"]],
            "instruction_warnings": {r["new_episode"]: r["instruction_selection"]["warnings"]
                                     for r in records if r["instruction_selection"]["warnings"]},
            "path_portability": "Bundle-relative companion paths; run --repair-paths after moving for official absolute-path consumers",
            "color_decoder": "Official RoboTwin marker-aware decode_image_bit",
            "official_helpers": {name: {"url": spec[0], "sha256": spec[1]} for name, spec in HELPERS.items()},
            "vqa_included": False, "state_annotation": "Official heuristic phase segmentation, not manual labels",
        })
        if output.exists():
            raise FileExistsError(f"Output appeared during conversion; refusing overwrite: {output}")
        work.rename(output)
        work = None
        print(f"\nSUCCESS: {output}\nConverted {len(records)} episodes; source files unchanged.")
        print("Source HDF5, original instructions and scene metadata preserved in source_data/.")
        if not args.no_lossless_images:
            print("Lossless decoded RGB images saved separately in lossless/; standard outputs remain JPEG.")
        print("Custom VQA questions are not generated. This is not the official validation split.")
        return 0
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if work is not None:
            # Only remove this run's staging tree, after checking its resolved boundary.
            staging = work.resolve()
            if staging.parent == output.parent.resolve() and staging.name.startswith(f".{output.name}.partial-"):
                shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
