[English](README.md) | [简体中文](README_CN.md)

# RoboTwin → TriWorldBench Converter

Convert **legacy RoboTwin ALOHA HDF5 episodes** into a **custom dataset bundle with the TriWorldBench directory layout**. The converter copies robot trajectories without changing their numerical values, exports three camera views, selects an original task instruction, and generates phase annotations using an official TriWorldBench helper.

This is a community converter. Its output is your own converted dataset; it does not become the official TriWorldBench validation/test split. The script does not download datasets, train a model, run inference, run evaluation, or generate VQA questions and answers.

For a first run: **install dependencies → prepare one task → run a dry run → convert one episode → convert the entire task/configuration**. The examples below provide every command.

## Contents

- [1. Basic terms and supported data](#1-basic-terms-and-supported-data)
- [2. Install the project](#2-install-the-project)
- [3. Prepare the source dataset](#3-prepare-the-source-dataset)
- [4. Convert your first dataset](#4-convert-your-first-dataset)
- [5. Understand the output](#5-understand-the-output)
- [6. Numerical values, actions, and image quality](#6-numerical-values-actions-and-image-quality)
- [7. Instruction selection and its limits](#7-instruction-selection-and-its-limits)
- [8. All command-line options](#8-all-command-line-options)
- [9. Move a converted dataset](#9-move-a-converted-dataset)
- [10. Troubleshooting](#10-troubleshooting)
- [11. Tests and verification scope](#11-tests-and-verification-scope)
- [12. Official references and helper files](#12-official-references-and-helper-files)

## 1. Basic terms and supported data

| Term | Meaning in this project |
| --- | --- |
| **Task** | A manipulation task, such as `adjust_bottle`. |
| **Configuration** | A dataset subdirectory describing a robot/setup/collection variant, such as `aloha-agilex_clean_50`. |
| **Episode / trajectory** | One complete recorded attempt at a task. It contains a sequence of `T` time steps, not just one image. |
| **HDF5** | A `.hdf5` file containing named groups and arrays, much like folders and tables inside one file. |
| **Frame** | A camera image at one time step. Each supported episode contains head, left, and right camera sequences. |
| **GT** | Ground truth: the recorded image sequences used as reference targets for training or evaluation. |
| **STATE** | JSON files with automatically estimated action phases and events, such as approaching, grasping, and releasing. These are heuristic annotations, not manually verified labels or the robot's raw joint-state arrays. |
| **Bundle** | The complete output directory, containing model-facing files, ground truth, metadata, and preserved source files. |

### Supported input

The converter currently targets **legacy ALOHA data with 14-dimensional joint actions**. Each episode must have:

- The nine numeric datasets listed in [Section 6](#6-numerical-values-actions-and-image-quality).
- Three camera datasets: `observation/head_camera/rgb`, `observation/left_camera/rgb`, and `observation/right_camera/rgb`.
- Images that decode to **320 × 240 RGB** pixels, stored as `uint8` arrays of shape `(240, 320, 3)` after decoding.
- The same nonzero number of time steps across the three cameras and all numeric datasets.
- A separate, matching `instructions/episodeN.json` file.

The script checks that numeric values are finite and that `joint_action/vector` exactly matches the documented component order. Additional source HDF5 fields are allowed and remain available in the preserved source copy.

**Not directly supported:** newer XPolicyLab `state/action/vision` layouts, LeRobot datasets, different joint counts, missing camera views, or other image sizes. A RoboTwin download is not guaranteed to work merely because it is called “RoboTwin 2.0”; inspect its internal layout. The converter does not automatically resize images or reinterpret a different robot's actions.

### How many episodes can it convert?

There is no fixed episode-count limit in the script. One invocation processes **one task and one configuration**. `--limit 1` converts one episode; `--limit 0` selects all matching episodes in that task/configuration directory. It does not traverse every task on the official website.

The number that successfully converts depends on the files actually present, their format, annotation consistency, and available disk space. No success count is claimed for the entire official RoboTwin collection. Start with the checks below for each new task/configuration.

## 2. Install the project

You need Git, **Python 3.10 or newer**, and enough space for the original HDF5 files plus exported JPEGs and optional PNGs. No GPU or running RoboTwin simulator is needed for conversion.

The required Python packages are `numpy`, `h5py`, `Pillow`, and `opencv-python`. There is currently no `requirements.txt`; install them using the command below.

The commands run Python directly from a virtual environment, an isolated directory for this project's dependencies. You do not need to activate it or change PowerShell's execution policy.

### Windows: PowerShell

Run this in a parent directory where you want to keep the project:

```powershell
git -c core.autocrlf=false clone https://github.com/FKEY1520/robotwin-to-triworldbench.git robotwin-to-triworldbench
Set-Location .\robotwin-to-triworldbench
git config core.autocrlf false
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy h5py Pillow opencv-python
```

Check that `py -3 --version` reports Python 3.10 or newer before creating the environment. If `py` is unavailable, check `python --version`, then use `python -m venv .venv` to create the environment.

### Linux / macOS: shell

These are equivalent setup commands; this project has been tested on Windows, and Linux/macOS have not been verified in this project.

```bash
git -c core.autocrlf=false clone https://github.com/FKEY1520/robotwin-to-triworldbench.git robotwin-to-triworldbench
cd robotwin-to-triworldbench
git config core.autocrlf false
python3 --version
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install numpy h5py Pillow opencv-python
```

Check that `python3 --version` reports Python 3.10 or newer. Some Linux installations require their distribution's Python `venv` package first.

**Why disable automatic line-ending conversion?** The two official helper `.py` files are checked against SHA-256 hashes before use. Changing their original LF line endings to Windows CRLF changes their bytes and fails that check, even if the code looks identical. Keep the helper files unchanged; see [Section 12](#12-official-references-and-helper-files) if a checksum fails.

## 3. Prepare the source dataset

Download and extract the desired legacy dataset yourself. Put it **outside the Git repository**. These examples use a sibling `RoboTwin_Raw` directory:

```text
your_workspace/
├── robotwin-to-triworldbench/          # This repository; run commands here
│   ├── convert_robotwin_to_triworld.py
│   ├── README.md
│   ├── README_CN.md
│   ├── tests/
│   └── _triworld_helpers/
└── RoboTwin_Raw/                       # Source data, outside the repository
    └── adjust_bottle/                  # --task
        └── aloha-agilex_clean_50/      # --config
            ├── data/
            │   ├── episode0.hdf5
            │   ├── episode1.hdf5
            │   └── ...
            ├── instructions/
            │   ├── episode0.json
            │   ├── episode1.json
            │   └── ...
            ├── scene_info.json        # Optional
            ├── seed.txt               # Optional
            └── config.yaml            # Optional; config.json/config.yml also copied
```

`--source-root` must point to `RoboTwin_Raw`, not to `data/` or directly to an HDF5 file. The script constructs the complete path as:

```text
SOURCE_ROOT / TASK / CONFIG / data / episodeN.hdf5
```

Only filenames matching `episode` + digits + `.hdf5` are selected. Every selected HDF5 file needs an instruction JSON with the same episode number.

A typical instruction file is:

```json
{
  "seen": [
    "Use the left arm to pick up the bottle.",
    "Lift the bottle from the table."
  ],
  "unseen": [
    "Raise the bottle with the left arm."
  ]
}
```

This illustrates the structure; use the original instructions corresponding to your actual recordings. A JSON string, a list of strings, or an object such as `{"instruction": "..."}` is also supported. Do not invent matching labels merely to make a missing-file check pass.

Optional metadata files must be at the configuration root shown above. The converter copies `scene_info.json`, `seed.txt`, `config.json`, `config.yaml`, and `config.yml` when present. It does not scan arbitrary directories for other metadata.

## 4. Convert your first dataset

Run all commands from the repository directory. Relative paths below are resolved from that current working directory. Replace the source root, task, and configuration if your dataset differs.

**Always specify your source and output paths.** The script's built-in defaults still contain the original developer's `P:` drive paths. The commands here override them and are portable.

### Step 1: Check all selected episodes without generating output

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_robotwin_to_triworld.py --source-root '../RoboTwin_Raw' --task adjust_bottle --config aloha-agilex_clean_50 --output-root './outputs/adjust_bottle_smoke' --limit 0 --dry-run
```

Linux / macOS:

```bash
./.venv/bin/python -X utf8 ./convert_robotwin_to_triworld.py --source-root '../RoboTwin_Raw' --task adjust_bottle --config aloha-agilex_clean_50 --output-root './outputs/adjust_bottle_smoke' --limit 0 --dry-run
```

The dry run checks file paths, matching instructions, required fields, array shapes, finite numeric values, vector order, and arm/instruction consistency for every selected episode. It reads the numeric arrays, so it may take time on a large dataset.

It **does not decode images, check their decoded resolution, load/download helpers, generate STATE, or write converted files**. A successful dry run is a useful first check, not proof that every image will decode successfully. The chosen output directory must not already exist, even for a dry run.

### Step 2: Fully convert one episode

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_robotwin_to_triworld.py --source-root '../RoboTwin_Raw' --task adjust_bottle --config aloha-agilex_clean_50 --output-root './outputs/adjust_bottle_smoke' --limit 1
```

Linux / macOS:

```bash
./.venv/bin/python -X utf8 ./convert_robotwin_to_triworld.py --source-root '../RoboTwin_Raw' --task adjust_bottle --config aloha-agilex_clean_50 --output-root './outputs/adjust_bottle_smoke' --limit 1
```

This small trial is often called a **smoke run**. It performs the actual image decoding and export, numeric copy verification, source preservation, and STATE generation. If the official helpers are missing, the script downloads the exact pinned versions and verifies their hashes.

A completed run prints `SUCCESS:` followed by the output location. Open the three images in `outputs/adjust_bottle_smoke/test_dataset/first_frame/` and inspect `conversion_report.json` and `robotwin_episode_mapping.json` before continuing. For the first example run, `conversion_report.json` should contain `"episode_count": 1`, `"numeric_values_and_dtype_preserved": true`, and `"lossless_pngs": true`. Review any entries under `instruction_warnings`; a successful run can still report annotation warnings.

### Step 3: Convert the entire selected task/configuration

Use a **new output directory** so that it does not conflict with the smoke run:

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_robotwin_to_triworld.py --source-root '../RoboTwin_Raw' --task adjust_bottle --config aloha-agilex_clean_50 --output-root './outputs/adjust_bottle_clean50' --limit 0
```

Linux / macOS:

```bash
./.venv/bin/python -X utf8 ./convert_robotwin_to_triworld.py --source-root '../RoboTwin_Raw' --task adjust_bottle --config aloha-agilex_clean_50 --output-root './outputs/adjust_bottle_clean50' --limit 0
```

`--limit 0` means all matching files currently present, including the episode already used for the smoke run. This creates a separate complete bundle. A folder named `clean_50` is not by itself proof that 50 files are available. For another task, change `--task`, `--config` as needed, and `--output-root`; run the same three-step workflow again.

The script reads source files without modifying them and refuses to overwrite existing output directories. There is no append, resume, episode-offset, automatic skipping, or multi-task mode. A failed episode stops the whole batch. Files are built in a temporary sibling directory named like `.adjust_bottle_clean50.partial-...`, then renamed to the final output only after success. Ordinary conversion errors clean up that temporary directory; forced process termination or power loss may leave a partial directory.

## 5. Understand the output

A successful conversion produces a bundle like this. `episode1` is the new output number; `episode0` is an example original source number.

```text
outputs/adjust_bottle_clean50/
├── test_dataset/
│   ├── data/
│   │   └── episode1.hdf5                  # Nine numeric datasets only
│   ├── first_frame/
│   │   ├── episode1_head.jpg
│   │   ├── episode1_left.jpg
│   │   └── episode1_right.jpg
│   └── instructions/
│       └── episode1.json                  # One selected original instruction
├── gt_dataset/
│   └── episode1/
│       ├── episode1.json                  # Instruction and source metadata
│       ├── head/frames/frame_00000.jpg
│       ├── left/frames/frame_00000.jpg
│       └── right/frames/frame_00000.jpg
├── STATE/
│   ├── episode1.json                      # Heuristic phases/events and source paths
│   └── manifest.json
├── lossless/                              # Omitted with --no-lossless-images
│   └── episode1/
│       ├── head/frames/frame_00000.png
│       ├── left/frames/frame_00000.png
│       └── right/frames/frame_00000.png
├── source_data/
│   └── adjust_bottle/aloha-agilex_clean_50/
│       ├── data/episode0.hdf5             # Complete, byte-identical source copy
│       ├── instructions/episode0.json     # Complete original instruction JSON
│       ├── scene_info.json                # Optional source metadata
│       ├── seed.txt                       # Optional
│       └── config.yaml                    # Optional; other accepted configs also copied
├── robotwin_episode_mapping.json
└── conversion_report.json
```

Each `frames/` directory contains all `T` images, from `frame_00000` to the frame at index `T - 1`, with at least five digits in the filename. There is no frame subsampling. Each first-frame JPEG is an exact file copy of the corresponding GT frame-zero JPEG.

Source episodes are sorted **numerically** by their original numbers and assigned new numbers starting at `episode1`. For example, source `episode0`, `episode2`, and `episode10` become output `episode1`, `episode2`, and `episode3`. Use `robotwin_episode_mapping.json` to recover the original task, configuration, episode number, selected instruction, and source paths.

The preserved source HDF5 retains everything it originally contained, including a front camera, camera calibration, and extra fields when present. These are not added to the three-view `test_dataset` HDF5. The converter verifies complete HDF5/instruction file copies using SHA-256 and records hashes for copied metadata. It does not copy `_traj_data`, source video directories, virtual environments, or arbitrary other source files. Thus, `source_data` preserves the selected recordings and supported metadata, not an entire raw directory backup.

`conversion_report.json` records the episode count, preservation settings, JPEG quality, helper versions, instruction changes/warnings, and omitted source items. The mapping's path fields ending in `_rel` are relative to the **bundle root**, not the JSON file's own subdirectory.

### Which files should a model read?

For a world model conditioned on actions, a typical interface is:

```text
three initial camera images + instruction + allowed action sequence
                           ↓
              predicted future camera images
```

Use the required files under `test_dataset/` to build your model adapter. The adapter must explicitly choose the numeric fields required by its protocol; the converter supplies both joint-action and end-effector arrays and does not choose a model's representation for it.

The full future image sequences in `gt_dataset/` and `lossless/` are targets for training or references for evaluation. `STATE/` contains information derived from the complete trajectory. Do not feed future target images, full-trajectory annotations, or hidden source/scene information to a prediction model unless your evaluation protocol explicitly permits it. Extra files are preserved for traceability and evaluation, not automatic permission to use them as inputs.

Standard TriWorldBench readers continue to use `.jpg`. A custom pipeline that wants the lossless decoded pixels must explicitly read `lossless/`; its frame zero is the PNG equivalent of the initial observation. Do not simply rename PNG files to `.jpg`.

## 6. Numerical values, actions, and image quality

### Numeric schema

`T` is the number of time steps in the episode. It can differ between episodes.

| Dataset inside the HDF5 file | Required shape | Meaning for supported legacy ALOHA data |
| --- | --- | --- |
| `endpose/left_endpose` | `(T, 7)` | Left end-effector pose: `x, y, z, qw, qx, qy, qz` |
| `endpose/left_gripper` | `(T,)` | Left gripper value stored with the end-effector data |
| `endpose/right_endpose` | `(T, 7)` | Right end-effector pose in the same order |
| `endpose/right_gripper` | `(T,)` | Right gripper value stored with the end-effector data |
| `joint_action/left_arm` | `(T, 6)` | Six left-arm joint target positions |
| `joint_action/left_gripper` | `(T,)` | Left gripper action value |
| `joint_action/right_arm` | `(T, 6)` | Six right-arm joint target positions |
| `joint_action/right_gripper` | `(T,)` | Right gripper action value |
| `joint_action/vector` | `(T, 14)` | Left arm 6 + left gripper 1 + right arm 6 + right gripper 1 |

For this legacy ALOHA source, joint actions are **absolute joint drive-target positions**, not differences from the preceding frame and not measured joint positions (`qpos`). An absolute target gives the desired position; a relative action would instead give a change such as `target[t] - target[t-1]`. This converter never computes that difference.

The end-effector pose uses world coordinates and a quaternion in **`qw, qx, qy, qz`** order. A quaternion represents orientation with four values. If your own model concatenates both 7-value poses and their two gripper values, the result is **16 dimensions**. That is a separate representation from the **14-dimensional joint-action vector**; the script does not transform one into the other.

The nine numeric datasets are copied directly. The converter then verifies that their shapes, dtypes, and array bytes match the source exactly. It does **not** normalize, change units, change dtype, reorder quaternion components, calculate relative actions, interpolate, shift the sequence in time, or subsample frames. The checked ALOHA samples used `float64`; the script preserves the source dtype instead of forcing every input to `float64`.

Consequently, compatible source data needs a storage-layout conversion, not a numerical transformation merely to match this bundle. If a particular model expects normalized or relative actions, implement that separately in its documented model adapter.

### Why are both JPEG and PNG produced?

```text
Image bytes stored in source HDF5
                ↓ Official RoboTwin decoding
          Decoded RGB pixel array
            ├── JPEG, quality 95 by default → standard TriWorldBench layout
            └── PNG, lossless compression  → supplementary lossless/ directory
```

Both output formats are generated **independently from the same decoded RGB array**. The PNG is not produced by reading the newly written JPEG.

- **JPEG** is lossy. Re-encoding may slightly change pixel values, even at quality `100`. It is retained because the reference layout/readers expect JPEG filenames.
- **PNG** uses lossless compression. It can reduce file size without changing the decoded pixels. Each written PNG is reopened and checked for exact pixel equality. It cannot recover information already lost when the source image was originally JPEG-compressed.
- Neither branch resizes images or drops frames. The complete original encoded images also remain in the copied source HDF5.
- The official decoder handles the RGB/BGR convention of supported legacy buffers. Do not add an extra red/blue channel swap after decoding.

PNG export and full source copies increase disk usage and conversion time. Add `--no-lossless-images` to a conversion command if you do not need the supplementary PNGs; this still keeps standard JPEGs and complete source HDF5 copies. Change JPEG quality with `--jpeg-quality`, from `1` to `100`, if your downstream workflow allows it.

## 7. Instruction selection and its limits

Only one instruction is written to `test_dataset/instructions/episodeN.json`, while the complete original instruction JSON is preserved in `source_data/`. The output instruction file has this structure:

```json
{
  "instruction": "Use the left arm to pick up the bottle."
}
```

The converter searches recognized keys in this order: `instruction`, `prompt`, `task_instruction`, `text`, `description`, `seen`, `unseen`, `instructions`. Lists retain their original order; blank strings are ignored. For the usual `seen`/`unseen` structure, a nonempty `seen` candidate is considered first.

The script estimates whether exactly one arm moves from changes in joint/gripper target values. An arm is considered moving when its largest target-value range exceeds `1e-4`. It also checks the optional scene arm label in `scene_info.json`: for source `episode0`, it reads the `episode_0` entry, or falls back to `episode0`, then looks at `info["{a}"]`. If that label is `left` or `right` and contradicts an unambiguous moving arm, conversion stops under **all** instruction policies.

| `--instruction-policy` | Behavior when the first instruction explicitly names the wrong arm |
| --- | --- |
| `consistent` — default | Prefer an original alternative explicitly naming the expected arm; otherwise use an original candidate with no recognized arm claim. Search only within the first candidate's same `seen`/`unseen` group. Stop if none is compatible. |
| `strict` | Stop and request manual review, even if another compatible candidate exists. |
| `first` | Keep the first original candidate and record a warning. This deliberately retains the annotation conflict. |

If the first instruction has no recognized conflict, it stays unchanged. The script never rewrites words or silently switches from `seen` to `unseen` to resolve a conflict. It records the original candidate, selected candidate, JSON location, reason, and warnings in the mapping and related metadata.

This is a narrow consistency check, **not general language understanding**. It recognizes explicit English expressions such as “left arm”, “right gripper”, or “both hands” using text patterns. It cannot verify every object name, color, spatial relation, paraphrase, or non-English instruction. When both arms move or neither clearly moves, it does not infer a single expected arm and keeps the first candidate with a warning. Human review is still needed for annotation quality.

`STATE` annotations have a separate limitation: the official phase helper uses motion/gripper heuristics and task categories. Unknown task names fall back to its `pick_place` category. Successful export does not prove that those phases describe every new task correctly; inspect them before using them as labels.

## 8. All command-line options

| Option | Default | What it does |
| --- | --- | --- |
| `--source-root PATH` | Developer-specific source path below | Root containing task directories. |
| `--task NAME` | `adjust_bottle` | One task directory name, not a path. |
| `--config NAME` | `aloha-agilex_clean_50` | One configuration directory name, not a path. |
| `--output-root PATH` | Developer-specific output path below | New bundle directory; it must not exist and must be separate from the selected source directory. |
| `--limit N` | `1` | First `N` episodes in numeric order; `0` means all selected episodes. Negative values are rejected. |
| `--dry-run` | Off | Run the preflight checks without image decoding or output generation. |
| `--instruction-policy POLICY` | `consistent` | Choose `consistent`, `strict`, or `first`; see Section 7. |
| `--no-lossless-images` | Off | Skip supplementary PNGs. PNGs are generated by default. |
| `--jpeg-quality N` | `95` | JPEG quality from `1` through `100`; none is guaranteed lossless. |
| `--repair-paths BUNDLE_ROOT` | Unset | Repair path metadata in an existing bundle instead of converting data. |
| `-h`, `--help` | — | Print usage and exit. |

The exact built-in path defaults are:

```text
--source-root
P:\RoboTwin数据集\RoboTwin_Raw

--output-root
P:\RoboTwin数据集\RoboTwin_convert\adjust_bottle_convert\TriWorldBench_adjust_bottle_smoke_v2
```

These are local development defaults, not required folder names or shared download locations. Always override both paths on another computer. Paths containing spaces should be quoted. `--task` and `--config` must each be a single directory name.

To display the script's built-in help:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_robotwin_to_triworld.py --help
```

```bash
./.venv/bin/python -X utf8 ./convert_robotwin_to_triworld.py --help
```

## 9. Move a converted dataset

Move or copy the **whole bundle**, including `source_data/`, before repairing paths. Some official consumers use absolute paths, particularly `STATE.hdf5_path` when reading source camera calibration. The converter also saves relative paths so that the absolute ones can be rebuilt after a move.

From the repository directory, pass the bundle's **new** root:

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_robotwin_to_triworld.py --repair-paths '../moved_datasets/adjust_bottle_clean50'
```

Linux / macOS:

```bash
./.venv/bin/python -X utf8 ./convert_robotwin_to_triworld.py --repair-paths '../moved_datasets/adjust_bottle_clean50'
```

Replace the example path with the directory you actually moved the bundle to. Repair mode does not use `--source-root`, `--task`, `--config`, or `--output-root`. **Do not combine it with `--dry-run`; that combination is rejected.**

The command validates required paths, backs up changed JSON files under `path_repair_backups/<timestamp>/`, and updates the mapping, per-episode GT metadata, and STATE paths. It does not re-encode images, change numeric arrays, reselect instructions, or regenerate phases. If all paths are already correct, it reports zero updates and creates no new backup. Historical `original_source_*` fields remain a record of where conversion originally read the data.

Older bundles without preserved source copies can also have their paths repaired, but their original source HDF5 and instruction files must still exist at the recorded absolute paths. Repair does not upgrade old bundles with PNGs, source copies, new scene metadata, or corrected instruction choices. To obtain those additions, rerun conversion from the source into a new output directory.

## 10. Troubleshooting

| Message or symptom | Cause and next step |
| --- | --- |
| `Missing dependency` / `No module named ...` | Install the four packages using the same virtual-environment Python used to run conversion. Check that the terminal is in the repository directory. |
| `Expected source folder` / `No episodeN.hdf5 files found` | Check `SOURCE_ROOT/TASK/CONFIG/data/`, archive extraction nesting, and exact file names. The script does not read ZIP files directly. |
| `Missing matching instruction` | Supply the actual `instructions/episodeN.json` paired with that recording. HDF5 files alone are insufficient. |
| `Output already exists` | Choose a new `--output-root`. Rerunning in an existing bundle does not resume or overwrite it; this check also applies to dry runs. |
| `missing /...`, unexpected shape, or `vector/component order mismatch` | The source does not satisfy the expected legacy ALOHA contract. Inspect its schema rather than renaming unrelated fields or reshaping values to bypass the check. |
| `nonfinite values` | The source contains NaN or infinity in a required numeric array. Review the source episode. The converter does not silently replace values. |
| `Failed to decode image bits` / expected `320x240 RGB` | An encoded frame is damaged, unsupported, or the wrong size. A dry run cannot detect these image problems; a full conversion can. |
| Instruction conflict / no compatible original candidate | Review the original text and arm evidence. `consistent` can select an eligible original alternative; `strict` stops at the first conflict. Use `first` only if deliberately retaining that text and its warning is acceptable. |
| Scene arm conflicts with moving arm | Review the source scene label and trajectory together. This error is not bypassed by `--instruction-policy first`. |
| `Cannot download ...` | The pinned helper is absent and cannot be downloaded. Download the exact raw URL printed by the error and save its bytes to the indicated `_triworld_helpers/` file, then rerun. |
| `Helper checksum mismatch` | A helper differs from its pinned original, possibly because of LF/CRLF conversion. Restore the exact raw file from Section 12; do not disable checksum checking or change the expected hash. |
| Missing source file after `--repair-paths` | Restore the complete moved bundle including `source_data/`. An old bundle may still require its external original HDF5 and instruction files. |
| Disk full / many small image files | Free enough space or choose another output location. `--no-lossless-images` omits PNGs but still retains JPEGs and full source copies. |
| A hidden `.partial-...` directory remains after a crash | It is unfinished output. Ensure conversion is no longer running and inspect the exact directory before removing only that abandoned run's temporary files. Restart with a fresh output directory. |

Successful completion returns exit code `0`; a caught conversion failure returns `1`. Argument errors are reported by Python's argument parser. Inspect the first error rather than assuming earlier `[ok]` messages mean the entire batch finished.

## 11. Tests and verification scope

Run the included regression tests from the repository root. Windows is the verified environment:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

The equivalent Linux/macOS command is shown for reference. The current suite includes Windows-specific path tests and has not been verified on those systems, so do not assume the same pass result there:

```bash
./.venv/bin/python -X utf8 -m unittest discover -s tests -v
```

The current suite contains **34 tests**, covering instruction selection, path safety, and repair/backup/rollback behavior. The previously recorded local run passed all 34. These tests do not establish that every official dataset archive is compatible.

The local verification recorded for the revised converter includes:

- A full preflight check of **50 `adjust_bottle` episodes** in `aloha-agilex_clean_50`.
- A complete conversion of **one real episode**, with numeric values unchanged and **420 PNGs** checked against decoded source pixels.
- A directory-move/path-repair check.

Separately, the **older converter's 50-episode output** was compared against the raw recordings: 450 numeric arrays matched exactly, and 21,714 images had correct frame/view correspondence. That older image check does not mean JPEG pixels were lossless. The earlier 50-episode audit and the revised converter's one-episode full run are different verification scopes.

For your own data, confirm `SUCCESS`, review `conversion_report.json`, inspect instruction warnings and representative frames, and validate the resulting bundle in your own training/evaluation loader. Neither a folder name nor this verification history guarantees every task, robot, or later data release.

## 12. Official references and helper files

- [TriWorldBench repository](https://github.com/TriWorldBench/TriWorldBench) and [released data layout documentation](https://github.com/TriWorldBench/TriWorldBench/blob/main/test_data/data_readme.md).
- [TriWorldBench dataset repository](https://huggingface.co/datasets/TriWorldBench/Dataset).
- [Official RoboTwin dataset repository](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0), [collection/data-format documentation](https://robotwin-platform.github.io/doc/usage/collect-data.html), and [configuration documentation](https://robotwin-platform.github.io/doc/usage/configurations.html).
- [Legacy RoboTwin joint-target implementation](https://github.com/RoboTwin-Platform/RoboTwin/blob/7662179861ddf667dd51aa773451e8995874b0c0/envs/robot/robot.py#L498-L526), which explains the action semantics of the supported source format.

The converter uses these exact pinned helper files rather than whatever is currently on an upstream default branch:

| Local helper | Pinned source |
| --- | --- |
| `_triworld_helpers/decode_image_bit.py` | [RoboTwin image decoder at `bd563681`](https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/bd5636810602b59c36c86dcf1bba42e5e15c1f3f/data/decode_image_bit.py) |
| `_triworld_helpers/segment_episode_phases.py` | [TriWorldBench phase segmentation at `0d3603ec`](https://raw.githubusercontent.com/TriWorldBench/TriWorldBench/0d3603ec48df2448bd29d716c8de8cb6b45d3114/scripts/lib/segment_episode_phases.py) |

Their SHA-256 values are recorded in `HELPERS` inside `convert_robotwin_to_triworld.py` and in each successful conversion report. Existing helper files are verified; missing helpers are downloaded and verified before use. An existing file with the wrong hash stops conversion and is not automatically replaced. Restore its exact pinned original, or move that specific invalid helper out of `_triworld_helpers/` and rerun so it can be downloaded again. Downloads require network access only when those files are absent. Retain the original file bytes, including line endings.

The repository's `.gitignore` excludes virtual environments, generated `outputs/`, dataset files, archives, and backups. Keep downloaded data outside the repository and review your Git changes before publishing.
