# 在另一台服务器复现

## 0. 先区分可以复现什么

本仓库包含项目源码、查看器、地图脚手架和部分冻结配置，**不包含**模型权重、
训练数据、已训练 adapter checkpoint、参考图片或生成的 GLB/PLY。
目前可以按下文准备环境并复用原生 TRELLIS.2 推理入口；不能仅凭克隆仓库就声称
复现了训练、Map10 对齐或在线生成实验。

|入口|还需要什么|性质|
|---|---|---|
|`tools/native_smoke.py`|原生环境、三个权重目录、预处理输入图|复用 v17 的原生图像→3D，不是地图控制|
|`v22_server.py`|v21 GLB、页面引用的配置/输入图|只浏览、无 GPU 生成接口|
|`v27_live.py`|本地 Python/SSH、远端同路径代码、全部注册实验输入|相机触发的有界在线实验，不是开箱即用服务|
|历史 map adapter / LoRA 训练入口|对应 PLAN、合法 TRAIN 配对、父 checkpoint、数据 split|原实验研究代码，不应自动启动|

本次核查不安装新环境、不启动训练或推理。新安装命令来自工作环境的版本与构建记录，
**尚未在另一台全新服务器上做过端到端重装验证**。

## 1. 代码与实际版本

```bash
git clone https://github.com/menyunlei/map2world-4D-.git
cd map2world-4D-
export PROJECT_ROOT="$PWD"
export NATIVE_ROOT="$PROJECT_ROOT/experiments/recursive_map_research_v1/trellis2_native_quality_v1"
```

2026-10-15 对工作服务器进行了只读核验：Linux、Python **3.10.22**、
PyTorch **2.6.0+cu124**、torchvision **0.21.0+cu124**、CUDA Toolkit **12.4.99**、
FlashAttention **2.7.3**、Triton **3.2.0**；TRELLIS.2 worktree 当时是干净的。
需要支持 CUDA 12.4 的驱动、CUDA 编译器、C/C++ 构建工具、Git 和 Ninja。
工作硬件为 A800 80GB；这不是其他显存容量的保证。

精确源码 commit 在 [`reproduction-lock.json`](reproduction-lock.json)。
注意：旧 PLAN 的 TRELLIS.2 commit 有 **41 位误写**；正确值是：

```text
75fbf0183001ed9876c8dbb35de6b68552ee08bd
```

保留旧文件作为历史证据，不使用错误 commit checkout。

## 2. 安装原生 CUDA 环境

已有 Conda/Mamba 的 Linux 主机可以按下面的布局安装。没有 Conda 时先按其官方说明安装，
不要使用旧实验里针对原机器的网络代理/安装恢复脚本。

```bash
mkdir -p "$NATIVE_ROOT/vendor" "$NATIVE_ROOT/weights"
conda create -y -p "$NATIVE_ROOT/environment" python=3.10.22
conda activate "$NATIVE_ROOT/environment"
# 如果系统已有完整 CUDA 12.4 Toolkit，可将 CUDA_HOME 指向那个目录。
# 否则按工作环境布局安装 Toolkit；是否能解析到旧包取决于软件源。
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.0  # A800；换 GPU 时按实际架构设置
export MAX_JOBS=4
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$PROJECT_ROOT/requirements/native.txt"
```

克隆并锁定第三方源码，不能以最新 main 替代固定版本：

```bash
python - <<'PY'
import json, os, pathlib, subprocess
lock=json.loads((pathlib.Path(os.environ['PROJECT_ROOT'])/'docs/reproduction-lock.json').read_text())
vendor=pathlib.Path(os.environ['NATIVE_ROOT'])/'vendor'
sources={'TRELLIS.2': [lock['backbone']['url'], lock['backbone']['commit']], **lock['native_dependencies']}
for name,(url,commit) in sources.items():
    dest=vendor/name
    if dest.exists():
        raise SystemExit(f'{dest} already exists; inspect it rather than resetting it')
    subprocess.run(['git','clone',url,str(dest)],check=True)
    subprocess.run(['git','-C',str(dest),'checkout','--detach',commit],check=True)
    subprocess.run(['git','-C',str(dest),'submodule','update','--init','--recursive'],check=True)
PY
python -m pip install --no-build-isolation "$NATIVE_ROOT/vendor/utils3d"
python -m pip install --no-build-isolation "$NATIVE_ROOT/vendor/nvdiffrast"
# 工作记录最终从 nvdiffrec_v3 根目录构建，不使用旧脚本中的 /render 安装子目录。
python -m pip install --no-build-isolation "$NATIVE_ROOT/vendor/nvdiffrec_v3"
python -m pip install --no-build-isolation "$NATIVE_ROOT/vendor/CuMesh"
python -m pip install --no-build-isolation "$NATIVE_ROOT/vendor/FlexGEMM"
python -m pip install --no-build-isolation 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
python -m pip install --no-build-isolation "$NATIVE_ROOT/vendor/TRELLIS.2/o-voxel"
```

该 FlashAttention wheel 仅用于 CPython 3.10/Linux x86_64、torch 2.6/CUDA12、
CXX11 ABI FALSE；其他 ABI 不要强装。工作机曾遇到 CUDA 头文件/NVRTC linker 路径问题：
若使用 pip NVIDIA 库与不完整 Toolkit，需要让编译器找到各 `site-packages/nvidia/*/include`
和 `*/lib`。优先安装完整 Toolkit，不要修改系统库或盲目复用旧机器的 symlink。

将原生包加入环境（本次保留上游源码，不在其内修改文件）：

```bash
export PYTHONPATH="$NATIVE_ROOT/vendor/TRELLIS.2${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import importlib.util; print({n:importlib.util.find_spec(n) is not None for n in ['flash_attn','cumesh','flex_gemm','o_voxel','nvdiffrast','utils3d','nvdiffrec_render']})"
```

这一步检查是否安装，**不等于 CUDA 运算已验证**。FlexGEMM/Triton 的实际 import/算子
可能初始化 CUDA；只能在已授权空闲 GPU 上做运行检查。

## 3. 权重目录与运行配置

需要保留 checkpoint 的 `.json` 网络配置和同名 `.safetensors`，不能只拷贝权重文件。

```text
$NATIVE_ROOT/weights/
  TRELLIS.2-4B/pipeline.json
  TRELLIS.2-4B/ckpts/{模型名}.json + {模型名}.safetensors
  TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.{json,safetensors}
  dinov3/config.json + model.safetensors (+ 原模型其余配置文件)
```

权重来源分别是 `microsoft/TRELLIS.2-4B`、`microsoft/TRELLIS-image-large` 和
`facebook/dinov3-vitl16-pretrain-lvd1689m`。DINO 可能需要用户先取得访问许可并登录；
不在仓库写 token。允许下载时，用环境中的 Hugging Face CLI：

```bash
hf download microsoft/TRELLIS.2-4B --include pipeline.json 'ckpts/*.json' 'ckpts/*.safetensors' --local-dir "$NATIVE_ROOT/weights/TRELLIS.2-4B"
hf download microsoft/TRELLIS-image-large --include 'ckpts/ss_dec_conv3d_16l8_fp16.*' --local-dir "$NATIVE_ROOT/weights/TRELLIS-image-large"
hf download facebook/dinov3-vitl16-pretrain-lvd1689m --local-dir "$NATIVE_ROOT/weights/dinov3"
```

原工作机对应位置（用于找原资产，不要求新机使用相同绝对路径）：

```text
JOB=/hpc2hdd/home/xhe989/Documents/2541/project2/DreamX-World-master/experiments/recursive_map_research_v1
TRELLIS2=$JOB/trellis2_native_quality_v1/weights/TRELLIS.2-4B
SS_DECODER=$JOB/trellis2_native_quality_v1/weights/TRELLIS-image-large
DINO=/hpc2hdd/home/xhe989/Documents/ExGRPO/qwen/facebook/dinov3-vitl16-pretrain-lvd1689m
RUNTIME=$JOB/map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json
```

原工作机 8 组推理模型的 JSON/权重均已只读确认存在；完整名称见
`tools/reproduce.py` 的 `expected` 表。DINO 权重 SHA256：
`dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`。
其他模型最初未登记下载 revision；严格复现应复制原权重并分别做 SHA256 清单，不能承诺
今天下载的最新 snapshot 与原权重逐字节一致。

使用工具生成新运行配置，不编辑历史配置、不会覆盖已有文件：

```bash
python tools/reproduce.py configure \
  --template "$NATIVE_ROOT/weights/TRELLIS.2-4B/pipeline.json" \
  --weights "$NATIVE_ROOT/weights" --dino "$NATIVE_ROOT/weights/dinov3" \
  --output "$PROJECT_ROOT/experiments/recursive_map_research_v1/map_object_world_v1/repro_runtime/pipeline.json"
```

工具检查 8 组文件及 DINO hash，保留原生 Shape/Texture normalization，复用已核实
sampler 参数和 `low_vram=true`。`UnusedBackgroundRemoval` 依赖现有 native 入口的
shim，不能把这个 pipeline.json 直接当作未经改动的官方 demo 配置。

## 4. 一次原生推理检查（不训练、不声称地图控制）

输入须是预先固定、经过原实验规则白底合成/正方形补边的 RGB 图。
源图片和处理后的图片都不在本次代码上传中；可以从原运行包复制，或用官方示例按
`trellis2_native_street_v17.prepare()` 中的规则处理。自行换图不是历史输出的精确复现。

先 CPU 检查路径，不触发模型推理：

```bash
python tools/native_smoke.py --native-root "$NATIVE_ROOT" \
  --runtime "$PROJECT_ROOT/experiments/recursive_map_research_v1/map_object_world_v1/repro_runtime" \
  --input /absolute/path/to/fixed_padded_input.png
```

人工确认 GPU 所有权后，查询 UUID，仅选授权空闲卡：

```bash
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
# 替换 UUID，不要原样复制占位符。最多一次；失败后检查日志，不自动重试。
timeout --signal=TERM 1200s python -B -u tools/native_smoke.py \
  --native-root "$NATIVE_ROOT" \
  --runtime "$PROJECT_ROOT/experiments/recursive_map_research_v1/map_object_world_v1/repro_runtime" \
  --input /absolute/path/to/fixed_padded_input.png \
  --seed 942217 --output-name reproduction_native_smoke --gpu GPU-REPLACE-WITH-AUTHORIZED-UUID --run
```

wrapper 只在当前进程覆盖旧路径，调用原 v17 generator；只推理一次，不改模型权重。
它拒绝已有输出目录和已有 compute owner。这个新增 wrapper 已做 CPU 参数/代码检查，
尚未重新跑 GPU。运行后应有 `native_street_basecolor.glb`、`RAW_MESH.npz`、`RAW_NATIVE.pt`、
`ASSET.json`；GLB 是原生全部三角形+顶点基色预览，**不是完整 UV/PBR 成品**。
读取资产的 hash/面数和新视角验证后才能判断质量，不能只看进程退出码。

## 5. 查看器和在线服务

本地客户端使用 **Python 3.11+** 的独立环境，然后安装 `requirements/client.txt`。
原生 CUDA 环境是 Python 3.10，不要混用：旧查看器/服务使用了 3.11 才有的
`hashlib.file_digest`，直接用原生 3.10 启动可能报错。

```bash
conda create -y -n map2world-client python=3.11
conda activate map2world-client
python -m pip install -r "$PROJECT_ROOT/requirements/client.txt"
```

Three.js/OrbitControls 及 MIT license 已在仓库内，不需要前端 npm 构建。

离线 v21 浏览：先从原机器传入 `trellis2_long_street_v21/large_street_100m.glb`，
以及 `trellis2_streaming_v22/STREAM_STATE.json`（保留原资产 hash）、`map10_reference.png`。
相邻块模式另需 `trellis2_streaming_v22/chunk/native-adjacent-001/chunk_100m.glb`；
没有该资产就只检查 v21 和地图，不声称相邻块模式可用。在项目脚本目录运行：

```bash
cd "$PROJECT_ROOT/experiments/recursive_map_research_v1"
python -B -u v22_server.py --port 8767
```

服务器只监听 loopback。浏览器在自己电脑时，需要转发，不要公开裸 HTTP：

```bash
ssh -N -L 8767:127.0.0.1:8767 YOUR_USER@YOUR_SERVER -p YOUR_SSH_PORT
# 本地浏览器 http://127.0.0.1:8767/
```

v27 在线模式不能仅凭代码启动。需同步 **同一个注册实验**的输入与状态：
`CONTRACT.json`、`SOURCE_FREEZE.json`、`BASELINE_CACHE.json`、`EVALUATION_QUERIES.npz`、
`MAP10_PROVENANCE.json`、`map10_scaffold.json`、`map10_reference.png`、两个 attempt 的
`PLAN.json/input.png/ROAD_OBSERVATION.npz`，以及对应 v21/v24 父资产。
生成后已有的 `STATE.json`、报告、hash 和预算记录也要一起保留；不能删状态重置预算。
脚本会严格核验源 hash；当前查看器有显示修正，不能谎称与旧 source freeze 完全相同。

迁移 v27 需要显式适配以下硬编码项，而不是全局替换所有历史文件：

1. `paths.py` 的 ROOT/JOB（历史训练/worker 使用）。
2. `v23_server.py` 和 `v27_server.py` 的 REMOTE/PYTHON/SSH 地址、用户名和端口。
3. `v26_resident.py`/owner 中 GPU UUID 与中央 lease 路径，核实新机所有权。
4. 注册 PLAN 的输入/runtime 路径以及有效资产 hash；迁移配置另存，不覆盖原记录。
5. 在客户端 `work_known_hosts` 放经带外确认的主机公钥；使用 `getpass` 输入密码或自行配置密钥，
   不禁用 host-key 校验，不把密码写进脚本。

这些 prerequisites 成立才运行 `python -B -u v27_live.py`，浏览 `http://127.0.0.1:8772/`。
此入口有相机触发的 GPU 工作，和 v22 只读查看器不同。当前已结束实验不会因刷新或移动
自动获得新预算。本仓库没有偷偷启用新在线实验或训练。

## 6. 代码完整性与已知边界

```bash
python tools/reproduce.py audit
python -m unittest discover -s tools -p 'test_reproduce.py' -v
```

audit 使用 AST 检查 tracked Python 的语法及本机可见的 sibling-module 依赖，
不 import 执行训练脚本。它不证明外部 DreamX/TRELLIS 包、动态载入的数据、参考图和
checkpoint 完整。部分历史 G2-B/原 TRELLIS 实验还依赖原 DreamX 仓库及其数据；
那部分不是本次 TRELLIS.2 原生复现的前提，也不能宣称已独立打包。

地理配准、外观真实感、局部编辑与地图控制仍应分开评价。克隆、安装和 GLB 导出成功
不改变历史失败结论。大权重、大资产和私人认证信息持续由 `.gitignore` 排除。
