# RoboTwin → TriWorldBench 转换器

适用于当前 `RoboTwin_Raw/<task>/<config>/data/episodeN.hdf5` 旧格式，当前数值契约是 ALOHA 双臂14维关节目标。原始数据以只读方式使用，已有输出目录不会被覆盖。

## 本次修复

- **指令臂冲突检查**：对照原始场景信息和关节/夹爪变化，发现第一条指令显式写错机械臂时，优先从同一 seen/unseen 集合选择明确匹配的原始句子；再尝试不指定机械臂的原始句子。没有合适候选则报错，不直接替换文字，也不自动把 seen 改成 unseen。原始全部指令保留，并记录原候选、最终候选索引和选择原因。此检查只能识别明确的臂冲突，不是所有文本细节的语义验证。
- **数值不变**：仍直接复制9个数值字段，保存后检查 shape、dtype 和数组字节相同。不做差分、归一化、单位转换、抽帧或时间移位。
- **图像无损副本**：标准三视角 GT 和首帧仍输出真正的 RGB JPEG，以兼容官方固定 `.jpg` 的读取代码；另外默认导出 PNG，保证与官方解码后的 RGB 像素逐值一致。PNG不能恢复源 JPEG 压缩之前的信息，标准 JPEG 重新压缩仍有损。
- **完整 HDF5 与元数据**：逐字节复制源 HDF5，保留 front 相机、全部相机标定和其他字段；保留原始 instruction JSON、scene_info.json、seed.txt 及存在的 config 文件。STATE 带入该 episode 的原始 scene_info。副本路径用于 STATE 的相机参数读取。
- **移动后修复路径**：记录相对 bundle 根目录的路径，同时保留官方工具要求的绝对路径；`--repair-paths` 会更新 mapping、STATE、GT 元数据，修改前备份 JSON。支持旧转换包中仍有效的原始源文件路径。

## 运行

在 PowerShell 中进入本目录，用现有虚拟环境执行。

```powershell
Set-Location 'P:\RoboTwin数据集\convert_robotwin_to_triworld'

# 全部50条预检查，不生成数据。
.\.venv\Scripts\python.exe -X utf8 .\convert_robotwin_to_triworld.py --dry-run --limit 0

# 新目录中试转换第一条。
.\.venv\Scripts\python.exe -X utf8 .\convert_robotwin_to_triworld.py --limit 1

# 全量转换；输出名称用v2，与原先结果分开。
.\.venv\Scripts\python.exe -X utf8 .\convert_robotwin_to_triworld.py --limit 0 --output-root 'P:\RoboTwin数据集\RoboTwin_convert\adjust_bottle_convert\TriWorldBench_adjust_bottle_clean50_v2'
```

如果输出目录已存在，应指定另一个新目录。脚本不会自动删除或覆盖它。

默认源位置：`P:\RoboTwin数据集\RoboTwin_Raw`；默认任务 `adjust_bottle`；默认配置 `aloha-agilex_clean_50`。其他任务可通过 `--source-root`、`--task`、`--config` 指定，但必须符合同样的 HDF5 数值维度契约。

## 输出

```text
bundle/
  test_dataset/                 官方同类模型输入：9数值字段、三首帧JPEG、单条选定指令
  gt_dataset/                   官方布局的三视角JPEG序列和GT元数据
  STATE/                        阶段标注、原始scene_info、完整HDF5路径
  source_data/<task>/<config>/
    data/episode0.hdf5          完整原HDF5的独立副本，含四相机和标定
    instructions/episode0.json  原始全部语言候选，不更改原文
    scene_info.json            完整原场景信息（源文件存在时）
    seed.txt                   原始seed文件（源文件存在时）
  lossless/episode1/head/frames/frame_00000.png
  lossless/episode1/left/frames/frame_00000.png
  lossless/episode1/right/frames/frame_00000.png
  robotwin_episode_mapping.json
  conversion_report.json
```

`source_data` 使用原始 episode 编号，其他输出使用从1开始的扁平编号；以 mapping 为准。自定义训练若需要像素不再受有损压缩，应显式读取 `lossless/` 的 PNG，第0张就是无损首帧；现有官方读取器仍读标准 JPEG。

原始 `_traj_data`、源视频和虚拟环境不复制，记录在报告中。新增 source_data 和 PNG 会增加磁盘占用。可以用 `--no-lossless-images` 跳过 PNG；完整原 HDF5 和全部原始指令仍保留。

## 目录移动后

将整个 bundle 一起移动，保留其中 source_data。然后执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_robotwin_to_triworld.py --repair-paths 'P:\新的位置\TriWorldBench_adjust_bottle_clean50_v2'
```

该命令不重新转图，也不修改动作、指令、阶段或场景信息；只刷新 JSON 路径，旧 JSON 保存在 bundle 内 `path_repair_backups/`。官方几何评估会使用 `STATE.hdf5_path` 读取相机参数，移动后运行评估前需要修复。相对路径始终以 mapping 所在的 bundle 根目录为基准。

对**旧版结果**也可以使用同一命令修正旧目标路径，但它不会补回源副本、PNG、场景信息，也不会纠正旧指令；完整修复需用新版脚本重新生成到新目录。旧结果若原始源 HDF5 已不存在，命令会报错，不会把源路径伪装成不含相机标定的数值输入 HDF5。

## 选项和边界

- `--instruction-policy consistent`：默认，第一候选无明确冲突则保留；冲突时从同一集合选择原始替代句。
- `--instruction-policy strict`：第一候选与已知运动臂冲突就报错，交给人工审核。
- `--instruction-policy first`：显式保留第一候选，冲突写入日志和报告。
- `--jpeg-quality 95`：标准 JPEG 的质量参数，范围1–100；100也不能保证无损。
- 场景信息与明确的单臂运动相互冲突时直接报错；双臂/静止情况下不猜单臂标签，会记录不能确定的提示。
- 不自动生成 VQA 答案；自定义 VQA 需另行准备匹配的问答。新版 XPolicyLab `state/action/vision` 文件不适用于此转换器。

## 回归测试

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

本次修改前的脚本备份位于 `backups/`。
