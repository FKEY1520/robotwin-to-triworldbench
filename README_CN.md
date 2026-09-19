[English](README.md) | 简体中文

# RoboTwin → TriWorldBench 数据格式转换工具

这个项目把**旧版 RoboTwin ALOHA 机器人数据**转换成**采用 TriWorldBench 目录布局的自定义数据集**。它保留机器人的原始数值轨迹，导出三个相机视角的图像，选择一条原始任务指令，并使用 TriWorldBench 官方辅助脚本生成动作阶段标注。

这是社区转换工具。转换后的数据仍然是你自己的数据集，不会因此变成官方验证集或测试集。脚本不负责下载数据集、训练模型、模型推理、评估，也不生成 VQA（视觉问答）的题目和答案。

第一次使用，按这条顺序操作即可：**安装环境 → 整理一个任务的数据 → 预检查 → 试转一条 → 用新目录全量转换**。下面提供可直接修改使用的完整命令。

## 目录

- [1. 先理解几个概念与支持范围](#basics)
- [2. 安装项目与依赖](#installation)
- [3. 准备原始数据](#source-data)
- [4. 完成第一次转换](#first-conversion)
- [5. 看懂输出目录](#output-layout)
- [6. 动作数值、数据维度与图片质量](#data-semantics)
- [7. 指令怎么选择，有哪些限制](#instructions)
- [8. 完整命令行参数](#cli-options)
- [9. 移动数据集后修复路径](#move-bundle)
- [10. 常见问题与排查](#troubleshooting)
- [11. 测试方法与已验证范围](#verification)
- [12. 官方资料与辅助脚本](#references)

<a name="basics"></a>

## 1. 先理解几个概念与支持范围

| 名称 | 在这个项目中的含义 |
| --- | --- |
| **任务（task）** | 机器人要完成的操作，例如 `adjust_bottle`。 |
| **配置（config）** | 表示机器人、场景或采集条件的数据子目录，例如 `aloha-agilex_clean_50`。 |
| **轨迹（episode）** | 一次完整的任务执行记录，包含连续的 `T` 个时间步，不是一张图片。 |
| **HDF5** | 后缀为 `.hdf5` 的文件，可以在一个文件中保存多组有名称的数组，类似把文件夹和表格装进同一个容器。 |
| **帧（frame）** | 某个时刻的一张相机图像。本工具导出头部、左侧、右侧三个相机的图像序列。 |
| **GT** | Ground Truth，即真实记录的图像序列，用作训练目标或评估参照。 |
| **STATE** | 描述靠近、抓取、松开等动作阶段和事件的 JSON 标注，由规则自动估计；不是人工标注，也不是原始关节状态数组。 |
| **转换包（bundle）** | 一次转换产生的完整输出目录，包含模型输入、GT、标注、映射和保留的源文件。 |

### 支持哪些输入

当前脚本针对**旧版 ALOHA 双臂、14 维关节动作数据**。每条轨迹需要同时满足：

- HDF5 内包含[第 6 节](#data-semantics)列出的 9 个数值字段。
- 包含三个相机字段：`observation/head_camera/rgb`、`observation/left_camera/rgb`、`observation/right_camera/rgb`。
- 每帧解码后都是 **320 × 240 的 RGB 图像**，数组形状为 `(240, 320, 3)`，类型为 `uint8`。
- 三个相机序列和全部数值字段具有相同的非零时间步数 `T`。
- 每个 `episodeN.hdf5` 都有对应的 `instructions/episodeN.json`。

脚本还会检查数值中没有 NaN 或无穷大，并检查 `joint_action/vector` 与左右臂、夹爪字段按规定顺序拼接的结果完全相同。HDF5 可以含有更多字段，这些内容会保留在完整源文件副本中。

**目前不能直接转换：**新版 XPolicyLab 的 `state/action/vision` 布局、LeRobot 数据、不同关节数量的数据、缺少所需相机的数据，以及其他分辨率的图像。不能只看下载文件写着“RoboTwin 2.0”就判断兼容，必须检查内部结构。脚本不会自动缩放图片，也不会把另一种机器人的动作含义猜成 ALOHA 动作。

### 一共能转换多少条

脚本没有固定的轨迹条数上限。**一次运行只处理一个任务、一个配置目录。**`--limit 1` 选择一条轨迹，`--limit 0` 选择这个目录中全部符合文件名规则的轨迹；它不会自动遍历官方网站上的所有任务。

实际能成功转换多少条，取决于文件是否齐全、格式是否匹配、标注是否存在冲突，以及磁盘空间是否足够。本项目没有声称已经成功转换整个官方数据集。对于新的任务或配置，都应先预检查，再试转一条。

<a name="installation"></a>

## 2. 安装项目与依赖

需要安装 Git 和 **Python 3.10 或更新版本**。转换不需要 GPU，也不需要启动 RoboTwin 仿真器。请预留空间保存原始 HDF5 副本、JPEG，以及默认额外生成的 PNG。

Python 依赖只有这四项：`numpy`、`h5py`、`Pillow`、`opencv-python`。当前仓库没有 `requirements.txt`，直接使用下面的安装命令。

虚拟环境 `.venv` 用来单独存放本项目的依赖。下面始终直接调用虚拟环境中的 Python，**不需要激活环境，也不需要修改 PowerShell 执行策略**。

### Windows：使用 PowerShell

在准备存放项目的上级目录打开 PowerShell，执行：

```powershell
git -c core.autocrlf=false clone https://github.com/FKEY1520/robotwin-to-triworldbench.git robotwin-to-triworldbench
Set-Location .\robotwin-to-triworldbench
git config core.autocrlf false
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy h5py Pillow opencv-python
```

先确认 `py -3 --version` 显示的版本不低于 3.10，再创建虚拟环境。如果系统没有 `py` 命令，可以先用 `python --version` 确认版本，然后把创建环境的命令换成 `python -m venv .venv`。


**为什么克隆时关闭自动换行转换？**仓库中的两个官方辅助脚本会经过 SHA-256 校验，也就是检查文件字节是否与固定版本完全相同。Git 如果把 LF 换行自动改成 Windows CRLF，即使代码看起来没变，校验也会失败。请保持辅助脚本原始字节不变；遇到校验问题时见[第 12 节](#references)。

<a name="source-data"></a>

## 3. 准备原始数据

自行下载并解压符合要求的旧版数据，建议把它放在 **Git 仓库外面**。下面所有命令假设原始数据与项目目录并列：

```text
your_workspace/
├── robotwin-to-triworldbench/          # 项目目录；在这里执行命令
│   ├── convert_robotwin_to_triworld.py
│   ├── README.md
│   ├── README_CN.md
│   ├── tests/
│   └── _triworld_helpers/
└── RoboTwin_Raw/                       # --source-root 指向这里
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
            ├── scene_info.json        # 可选
            ├── seed.txt               # 可选
            └── config.yaml            # 可选；也会复制 config.json/config.yml
```

这个例子只说明文件结构，实际应使用与轨迹对应的原始指令。脚本也支持 JSON 字符串、字符串列表，以及 `{"instruction": "..."}` 形式。不要为了通过文件检查而随意编写与轨迹不对应的描述。

可选元数据必须放在上图的配置目录下。存在时，脚本会复制 `scene_info.json`、`seed.txt`、`config.json`、`config.yaml`、`config.yml`；不会搜索任意其他目录中的元数据。


## 4. 完成第一次转换

以下命令都在项目根目录执行，相对路径也从这里计算。`../RoboTwin_Raw` 表示上一级目录中的原始数据，`./outputs/...` 表示项目中的输出目录。如果你的数据在别处，请替换路径；任务和配置名称也要与实际文件夹一致。


### 第一步：预检查全部轨迹

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_robotwin_to_triworld.py --source-root '../RoboTwin_Raw' --task adjust_bottle --config aloha-agilex_clean_50 --output-root './outputs/adjust_bottle_smoke' --limit 0 --dry-run
```

`--dry-run` 会检查路径、配套指令、所需字段、数组形状、数值是否有限、动作向量拼接顺序，以及能够识别的机械臂和指令冲突。它会读取数值数组，数据量大时也需要等待。

预检查**不会解码图片，不会检查解码后的分辨率，不会加载或下载辅助脚本，也不会生成 STATE 或转换文件**。因此，预检查通过并不代表每张图片都能成功解码。即使是预检查，指定的输出目录也必须尚不存在。

### 第二步：实际试转一条轨迹

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_robotwin_to_triworld.py --source-root '../RoboTwin_Raw' --task adjust_bottle --config aloha-agilex_clean_50 --output-root './outputs/adjust_bottle_smoke' --limit 1
```

这类小规模试运行通常叫 **smoke run**。它会实际解码并导出图像、复制和校验数值、保留源文件、生成阶段标注。如果官方辅助脚本缺失，会先下载固定版本并校验。

完成后终端会打印 `SUCCESS:` 和输出位置。继续全量转换前，可以打开 `outputs/adjust_bottle_smoke/test_dataset/first_frame/` 中的三张首帧图片，并查看 `conversion_report.json` 和 `robotwin_episode_mapping.json`。按上面命令试转一条后，报告应包含 `"episode_count": 1`、`"numeric_values_and_dtype_preserved": true`、`"lossless_pngs": true`。还要检查 `instruction_warnings`，成功转换也可能带有标注警告。

### 第三步：用新目录转换全部轨迹

这里使用另一个输出目录，不能继续写入已经存在的试转目录。全量转换会重新包含已经试转过的那条轨迹，不是只转换“剩余轨迹”。

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_robotwin_to_triworld.py --source-root '../RoboTwin_Raw' --task adjust_bottle --config aloha-agilex_clean_50 --output-root './outputs/adjust_bottle_clean50' --limit 0
```

`--limit 0` 表示当前所选任务和配置中实际存在的全部匹配文件。文件夹名称含 `clean_50` 不代表里面一定有 50 个文件。换一个任务时，修改 `--task`、必要时修改 `--config`，再指定新的 `--output-root`，重复上面的步骤。

脚本只读原始数据，不覆盖已有输出。当前没有追加、断点续转、指定起始轨迹编号、自动跳过错误轨迹或自动遍历多个任务的模式。**任何一条选中轨迹转换失败，整批任务都会停止。**

转换先写入输出旁的临时目录，例如 `.adjust_bottle_clean50.partial-...`，全部成功后才改名成正式输出目录。正常捕获的转换错误会清理临时目录；强制结束进程或断电可能留下未完成的目录。


## 5. 看懂输出目录

下面以原始 `episode0` 转成新编号 `episode1` 为例：

```text
outputs/adjust_bottle_clean50/
├── test_dataset/
│   ├── data/
│   │   └── episode1.hdf5                  # 仅包含 9 个数值字段
│   ├── first_frame/
│   │   ├── episode1_head.jpg
│   │   ├── episode1_left.jpg
│   │   └── episode1_right.jpg
│   └── instructions/
│       └── episode1.json                  # 选定的一条原始指令
├── gt_dataset/
│   └── episode1/
│       ├── episode1.json                  # 指令与来源元数据
│       ├── head/frames/frame_00000.jpg
│       ├── left/frames/frame_00000.jpg
│       └── right/frames/frame_00000.jpg
├── STATE/
│   ├── episode1.json                      # 自动阶段/事件标注与源路径
│   └── manifest.json
├── lossless/                              # --no-lossless-images 可关闭
│   └── episode1/
│       ├── head/frames/frame_00000.png
│       ├── left/frames/frame_00000.png
│       └── right/frames/frame_00000.png
├── source_data/
│   └── adjust_bottle/aloha-agilex_clean_50/
│       ├── data/episode0.hdf5             # 完整源 HDF5，文件字节相同
│       ├── instructions/episode0.json     # 完整原始指令 JSON
│       ├── scene_info.json                # 源文件存在时保留
│       ├── seed.txt                       # 源文件存在时保留
│       └── config.yaml                    # 源文件存在时保留；其他支持的配置文件同理
├── robotwin_episode_mapping.json
└── conversion_report.json
```

每个 `frames/` 目录会保存全部 `T` 帧，不抽帧。帧号从 `0` 到 `T - 1`，文件名中的数字至少补足五位。`first_frame/` 的 JPEG 直接复制对应 GT 目录中的第 0 帧 JPEG，所以这两份文件的字节一致。

原始轨迹按编号**数值大小排序**，输出从 `episode1` 连续编号。例如原始 `episode0`、`episode2`、`episode10`，对应新编号 `episode1`、`episode2`、`episode3`。源文件副本仍保留原编号，具体对应关系以 `robotwin_episode_mapping.json` 为准。

完整源 HDF5 中原本存在的 front 相机、相机标定和其他字段会全部保留，但不会额外加入只包含 9 个数值字段的 `test_dataset/data/`。源 HDF5、原始指令和指定元数据副本经过 SHA-256 校验，相关哈希会记录下来。

脚本不复制 `_traj_data`、源视频目录、虚拟环境或任意其他源文件。因此，`source_data/` 保存的是**本次选中的轨迹及支持的元数据**，不是整个原始数据目录的完整备份。

`conversion_report.json` 记录转换条数、保留设置、JPEG 质量、辅助脚本版本、指令变化和警告等信息。映射文件中以 `_rel` 结尾的路径都以**整个转换包根目录**为基准，不是以各 JSON 所在子目录为基准。

### 模型应该读取哪些文件

对于以动作作为条件的世界模型，典型流程可以理解为：

```text
三个相机的初始图像 + 任务指令 + 评估协议允许使用的动作序列
                            ↓
                  模型预测后续图像序列
```

通常从 `test_dataset/` 中读取协议要求的输入。模型适配代码必须明确选择关节动作或末端位姿等所需字段；转换器同时保留了这些数组，不会替模型决定应该使用哪种表示。

`gt_dataset/` 与 `lossless/` 中完整的后续图像可用于训练目标或评估参照。`STATE/` 来自完整轨迹。除非你的评估协议明确允许，否则不要把未来目标图像、完整轨迹标注或隐藏的源场景信息当成预测模型的输入。

官方标准读取器仍然读取 `.jpg`。如果自己的训练流程希望使用解码后像素完全一致的图片，需要显式读取 `lossless/` 中的 PNG；其中第 0 帧就是对应的无损首帧。不要仅把 PNG 后缀改成 `.jpg`。



## 6. 动作数值、数据维度与图片质量

### 数值字段

`T` 是一条轨迹的时间步数，不同轨迹的 `T` 可以不同。

| HDF5 内部字段 | 必需形状 | 对受支持旧版 ALOHA 数据的含义 |
| --- | --- | --- |
| `endpose/left_endpose` | `(T, 7)` | 左末端位姿：`x, y, z, qw, qx, qy, qz` |
| `endpose/left_gripper` | `(T,)` | 与末端位姿一起记录的左夹爪值 |
| `endpose/right_endpose` | `(T, 7)` | 右末端位姿，顺序相同 |
| `endpose/right_gripper` | `(T,)` | 与末端位姿一起记录的右夹爪值 |
| `joint_action/left_arm` | `(T, 6)` | 左臂 6 个关节的目标位置 |
| `joint_action/left_gripper` | `(T,)` | 左夹爪动作值 |
| `joint_action/right_arm` | `(T, 6)` | 右臂 6 个关节的目标位置 |
| `joint_action/right_gripper` | `(T,)` | 右夹爪动作值 |
| `joint_action/vector` | `(T, 14)` | 左臂 6 + 左夹爪 1 + 右臂 6 + 右夹爪 1 |

对于这里适配的旧版 ALOHA 数据，关节动作是**绝对关节驱动目标位置**，不是前后帧差值，也不是传感器测得的实际关节位置 `qpos`。绝对目标表示“希望关节到达哪个位置”；相对动作则表示“在原位置上变化多少”，例如 `target[t] - target[t-1]`。本脚本不计算这种差分。

末端位姿采用世界坐标，姿态四元数的顺序是 **`qw, qx, qy, qz`**。四元数是使用四个数表示旋转的一种方法。如果在自己的模型中拼接左右两个 7 维位姿和两个夹爪值，会得到 **16 维**表示。这与 **14 维关节动作**是不同的表示，脚本不会把二者互相转换。

9 个数值字段直接从源 HDF5 复制，保存后逐字段检查形状、数据类型（dtype）和数组字节完全一致。脚本不进行归一化、单位转换、dtype 转换、四元数重排、相对动作计算、插值、时间移位或抽帧。之前检查的 ALOHA 样本是 `float64`，但脚本的规则是保留源类型，并不是强制所有输入都必须为 `float64`。

因此，对于符合要求的源数据，匹配这里的转换包只需要整理存储布局，**不需要为了格式转换额外修改动作数值**。如果某个模型要求归一化动作或相对动作，应在该模型的适配代码中单独、明确地处理。

### 为什么同时生成 JPEG 和 PNG

```text
原始 HDF5 中保存的图像内容
            ↓ RoboTwin 官方解码
        解码后的 RGB 像素数组
          ├─→ JPEG：默认质量 95，保存到标准目录
          └─→ PNG：无损压缩，保存到额外的 lossless/ 目录
```

两种图片都**直接从同一份解码后的 RGB 数组生成**。PNG 不是读取新生成的 JPEG 后再转换出来的。

- **JPEG 是有损压缩。**重新编码可能使像素值发生少量变化，即使质量设为 `100` 也不保证无损。保留 JPEG 是为了匹配参考布局及读取器要求的文件名。
- **PNG 是无损压缩。**它可以压缩文件体积，但不改变保存的像素值。脚本会重新打开每张 PNG，检查它与解码结果逐像素一致。PNG 无法恢复源图像最初经过 JPEG 压缩时已经丢失的信息。
- 两条导出路径都不缩放图片，也不丢帧。原始编码图像还保留在完整源 HDF5 副本中。
- 官方解码器会处理受支持旧数据的 RGB/BGR 通道约定，不要在解码后额外交换红蓝通道。

PNG 和完整源文件副本会增加磁盘占用和转换时间。如果不需要额外 PNG，可在转换命令末尾加 `--no-lossless-images`；标准 JPEG 与完整源 HDF5 仍会保留。JPEG 质量可通过 `--jpeg-quality` 设置为 `1` 到 `100`。



## 7. 指令怎么选择，有哪些限制

`test_dataset/instructions/episodeN.json` 最终只写入**一条选中的原始指令**。完整原始 JSON 中的其他候选句仍保留在 `source_data/`。输出指令文件的结构如下，实际句子取决于输入：

```json
{
  "instruction": "Use the left arm to pick up the bottle."
}
```

脚本按以下字段顺序寻找候选句：`instruction`、`prompt`、`task_instruction`、`text`、`description`、`seen`、`unseen`、`instructions`。列表内部保持原顺序，空白字符串会跳过。对于常见的 `seen`/`unseen` 结构，非空 `seen` 候选会优先出现。

脚本根据关节与夹爪目标值的变化，判断是否只有一只机械臂明确运动。某侧所有相关目标值中，最大的变化范围超过 `1e-4`，就认为这只臂在运动。存在 `scene_info.json` 时，对于源 `episode0`，先查找 `episode_0` 条目，不存在时再找 `episode0`，然后读取其中的 `info["{a}"]`。该标签为 `left` 或 `right` 且与明确的单臂运动冲突时，**所有指令策略都会报错。**

| `--instruction-policy` | 第一条指令明确写错机械臂时的处理 |
| --- | --- |
| `consistent`，默认 | 优先选择同组中明确写对机械臂的原始候选；没有时，尝试没有识别到机械臂声明的原始候选。只在第一候选所属的同一 `seen`/`unseen` 组内查找，没有合适候选就报错。 |
| `strict` | 立即停止并要求人工检查，即使后面存在可用候选也不会自动选择。 |
| `first` | 保留第一条原始句子并记录警告，相当于明确保留这个标注冲突。 |

第一候选没有识别到冲突时，直接保留。脚本不会直接把句子中的 `right` 改成 `left`，也不会为了消除冲突而悄悄从 `seen` 切换到 `unseen`。原候选、最终候选、JSON 中的位置、选择原因和警告都会记录到映射及相关元数据。

这是有限的文本一致性检查，**不是通用语义理解**。它使用文本规则识别英文中的 “left arm”“right gripper”“both hands” 等表达，不能检查全部物体名称、颜色、空间关系、其他表达方式或非英文指令。双臂同时运动、或两侧都没有明确运动时，不推断单个运动臂，会保留第一候选并记录警告。

`STATE` 还有独立限制：官方辅助脚本根据运动、夹爪变化和任务类别生成阶段，属于启发式规则。它不认识的任务名会回退到 `pick_place` 类别。成功导出并不代表所有新任务的阶段都准确，使用前应检查。

<a name="cli-options"></a>

## 8. 完整命令行参数

| 参数 | 默认值 | 用法 |
| --- | --- | --- |
| `--source-root PATH` | 下方列出的开发环境路径 | 包含任务目录的原始数据根目录。 |
| `--task NAME` | `adjust_bottle` | 一个任务目录名，不是完整路径。 |
| `--config NAME` | `aloha-agilex_clean_50` | 一个配置目录名，不是完整路径。 |
| `--output-root PATH` | 下方列出的开发环境路径 | 新的完整转换包目录，必须尚不存在，且与所选源目录分开。 |
| `--limit N` | `1` | 按原编号数值排序，取前 `N` 条；`0` 表示全部，不能为负数。不是选择“编号为 N”的轨迹。 |
| `--dry-run` | 关闭 | 只做预检查，不解码图片、不生成输出。 |
| `--instruction-policy POLICY` | `consistent` | 可选 `consistent`、`strict`、`first`，详见第 7 节。 |
| `--no-lossless-images` | 关闭 | 不生成额外 PNG；未指定时默认生成 PNG。 |
| `--jpeg-quality N` | `95` | JPEG 质量，范围 `1` 到 `100`，所有取值都不保证像素无损。 |
| `--repair-paths BUNDLE_ROOT` | 未指定 | 对已有转换包修复路径元数据，而不执行新转换。 |
| `-h`、`--help` | — | 显示帮助并退出。 |

脚本当前两个路径默认值分别是：

```text
--source-root
P:\RoboTwin数据集\RoboTwin_Raw

--output-root
P:\RoboTwin数据集\RoboTwin_convert\adjust_bottle_convert\TriWorldBench_adjust_bottle_smoke_v2
```

它们是本地开发默认值，不是别人电脑上必须使用的目录，也不是下载地址。在其他电脑上应显式覆盖这两个参数。含空格的路径需要加引号；`--task` 和 `--config` 各自只能是单个目录名。

查看脚本自带帮助：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_robotwin_to_triworld.py --help
```

```bash
./.venv/bin/python -X utf8 ./convert_robotwin_to_triworld.py --help
```

<a name="move-bundle"></a>

## 9. 移动数据集后修复路径

请移动或复制**整个转换包**，包括 `source_data/`。部分官方工具使用绝对路径，尤其会通过 `STATE.hdf5_path` 读取源 HDF5 中的相机标定。转换包同时保存了相对路径，移动后可以据此重新生成绝对路径。

移动完成后，在项目目录中运行下面的命令，参数填写转换包的**新位置**：

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -X utf8 .\convert_robotwin_to_triworld.py --repair-paths '../moved_datasets/adjust_bottle_clean50'
```

Linux / macOS：

```bash
./.venv/bin/python -X utf8 ./convert_robotwin_to_triworld.py --repair-paths '../moved_datasets/adjust_bottle_clean50'
```

这里的路径只是示例，应替换为实际移动后的位置。修复模式不使用 `--source-root`、`--task`、`--config` 和 `--output-root`。**不能与 `--dry-run` 同时使用**，这种组合会报错。

命令会先检查所需文件，把需要修改的旧 JSON 备份到 `path_repair_backups/<时间戳>/`，再更新 mapping、每条 GT 元数据和 STATE 中的路径。它不重新编码图片，不修改数值，不重新选择指令，也不重新生成阶段。若路径已经正确，则报告更新数为 0，不创建多余备份。`original_source_*` 仍保留最初读取数据的位置，作为历史记录。

新版转换包的相对路径使用 `/` 分隔，移动到另一台电脑后应在目标电脑运行修复；Linux/macOS 尚未做实际迁移验证。旧版转换包如果没有完整源文件副本，修复仍要求原来记录的绝对路径下存在 HDF5 和指令文件。只有 Windows 绝对源路径的旧包，不能直接在 Linux 上恢复那些源路径。

修复路径不会给旧包补上 PNG、源文件副本、场景元数据或新的指令选择。需要这些内容时，应从原始数据重新转换到一个新输出目录。


## 10. 常见问题与排查

| 报错或现象 | 原因与处理方式 |
| --- | --- |
| `Missing dependency` / `No module named ...` | 使用运行脚本的同一个虚拟环境 Python 安装四个依赖，确认终端位于项目目录。 |
| `Expected source folder` / `No episodeN.hdf5 files found` | 检查 `SOURCE_ROOT/TASK/CONFIG/data/`、压缩包是否多套了一层目录，以及文件名。脚本不能直接读取 ZIP。 |
| `Missing matching instruction` | 缺少与轨迹配套的 `instructions/episodeN.json`。仅有 HDF5 不够，应找回真正对应的原始指令。 |
| `Output already exists` | 更换一个不存在的 `--output-root`。已有目录不能续传或覆盖；预检查也有这个限制。 |
| `missing /...`、形状不符或 `vector/component order mismatch` | 输入不符合旧版 ALOHA 的字段和维度要求。先检查数据结构，不要随意重命名字段或重塑数组来绕过检查。 |
| `nonfinite values` | 必需数值字段中存在 NaN 或无穷大。需要检查源数据，脚本不会静默填补数值。 |
| `Failed to decode image bits` 或要求 `320x240 RGB` | 存在损坏、格式不支持或尺寸不符的图片。预检查不解码，实际转换时才会发现此类问题。 |
| 指令冲突、没有可用原始候选 | 对照源指令和运动臂检查。`consistent` 只能选择符合条件的原始候选；`strict` 会停止。只有接受保留冲突文本及警告时才使用 `first`。 |
| 场景机械臂与运动臂冲突 | 同时检查源场景标签和轨迹；`--instruction-policy first` 也不会绕过此错误。 |
| `Cannot download ...` | 缺少辅助脚本且网络下载失败。打开错误提示中的固定 raw URL，把原始内容保存到提示的 `_triworld_helpers/` 文件后重试。 |
| `Helper checksum mismatch` | 辅助脚本与固定版本字节不一致，可能是 LF/CRLF 换行变化。按第 12 节恢复原始文件，不要关闭校验或改写预期哈希。 |
| 修复路径时提示缺少源文件 | 恢复包含 `source_data/` 的完整转换包。旧转换包可能仍依赖外部原始 HDF5 和指令文件。 |
| 磁盘不足、图片文件太多 | 清理空间或改用其他输出位置。`--no-lossless-images` 只省略 PNG，仍保存 JPEG 和完整源 HDF5。 |
| 意外中断后遗留隐藏的 `.partial-...` 目录 | 这是未完成的输出。确认转换进程已结束，并检查具体路径后，只清理该次运行留下的临时目录，再使用新输出目录重试。 |

成功完成的退出码为 `0`，捕获到的转换错误返回 `1`。命令行参数错误由 Python 参数解析器报告。应查看第一个错误，而不能因为前面出现过 `[ok]` 就认为整批已经成功。



## 11. 测试方法与已验证范围

在项目根目录运行回归测试。Windows 是已经验证的环境：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

Linux/macOS 等价命令仅供参考。当前测试包含 Windows 特有的路径用例，尚未在这两个系统上验证，不能假定会得到相同通过结果：

```bash
./.venv/bin/python -X utf8 -m unittest discover -s tests -v
```

现有 **34 项测试**检查指令选择、路径安全、路径修复、备份和失败回滚。已在 Windows 本地运行并全部通过。测试使用合成样例，不代表已经逐包验证全部官方数据，也不是完整图像转换的真实数据测试。

此前对新版转换器记录的真实数据验证包括：

- `adjust_bottle/aloha-agilex_clean_50` 的 **50 条轨迹全部预检查通过**。
- **1 条真实轨迹完整转换通过**，数值保持不变，**420 张 PNG** 与源解码像素一致。
- 完成转换目录移动和路径修复检查。

另外，**旧版转换器的 50 条输出**曾与原始数据全量核对：450 个数值数组完全一致，21,714 张图片的帧和视角对应正确。这个旧版图片检查并不表示 JPEG 像素无损。旧版 50 条全量核对，与新版 1 条实际完整转换，是不同的验证范围。

对于自己的数据，应确认终端输出 `SUCCESS`，查看 `conversion_report.json`，检查指令警告和代表性图片，并让自己的训练或评估读取器实际读取转换包。目录名称及上述验证记录都不能保证其他任务、机器人或后续发布版本一定兼容。



## 12. 官方资料与辅助脚本

- [TriWorldBench 官方仓库](https://github.com/TriWorldBench/TriWorldBench)与[发布数据布局说明](https://github.com/TriWorldBench/TriWorldBench/blob/main/test_data/data_readme.md)。
- [TriWorldBench 数据集仓库](https://huggingface.co/datasets/TriWorldBench/Dataset)。
- [RoboTwin 官方数据集仓库](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)、[采集与数据格式文档](https://robotwin-platform.github.io/doc/usage/collect-data.html)及[配置文档](https://robotwin-platform.github.io/doc/usage/configurations.html)。
- [旧版 RoboTwin 关节目标实现](https://github.com/RoboTwin-Platform/RoboTwin/blob/7662179861ddf667dd51aa773451e8995874b0c0/envs/robot/robot.py#L498-L526)，用于理解受支持源格式的动作含义。

转换器使用下面的固定版本辅助脚本，不会自动跟随上游默认分支的最新改动：

| 本地辅助脚本 | 固定来源 |
| --- | --- |
| `_triworld_helpers/decode_image_bit.py` | [RoboTwin 图像解码器，版本 `bd563681`](https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/bd5636810602b59c36c86dcf1bba42e5e15c1f3f/data/decode_image_bit.py) |
| `_triworld_helpers/segment_episode_phases.py` | [TriWorldBench 阶段划分脚本，版本 `0d3603ec`](https://raw.githubusercontent.com/TriWorldBench/TriWorldBench/0d3603ec48df2448bd29d716c8de8cb6b45d3114/scripts/lib/segment_episode_phases.py) |

SHA-256 值写在主脚本 `convert_robotwin_to_triworld.py` 的 `HELPERS` 常量中，成功转换时也会记入报告。辅助脚本存在时先检查哈希；缺失时下载并检查后再使用。**文件存在但哈希不匹配时，会停止，不会自动覆盖下载。**

处理方法是恢复表格链接中的精确原始文件，或者把那一个校验失败的辅助文件移出 `_triworld_helpers/`，让脚本在下次转换时重新下载。只有辅助文件缺失时才需要联网下载。保留原始字节，包括换行格式。

仓库的 `.gitignore` 会排除虚拟环境、生成的 `outputs/`、数据集文件、压缩包和备份。建议原始数据放在仓库外，发布前检查 Git 变更。
