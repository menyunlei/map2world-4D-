# Reproducing on Another Server

## 0. Understand What Can Be Reproduced

This repository contains the project source code, viewers, map scaffolding, and some frozen configurations. It **does not include** model weights,
training data, trained adapter checkpoints, reference images, or generated GLB/PLY files.
You can currently follow the instructions below to prepare the environment and reuse the native TRELLIS.2 inference entry point. Cloning the repository alone does not establish
that training, Map10 alignment, or online generation experiments have been reproduced.

|Entry point|Additional requirements|Scope|
|---|---|---|
|`tools/native_smoke.py`|Native environment, three weight directories, preprocessed input image|Reuses v17 native image-to-3D generation; does not provide map control|
|`v22_server.py`|v21 GLB, configurations/input images referenced by the page|Viewing only; no GPU generation interface|
|`v27_live.py`|Local Python/SSH, remote code at the same paths, all registered experiment inputs|Camera-triggered online experiment with defined limits; requires setup before use|
|Historical map adapter / LoRA training entry points|Corresponding PLAN, valid TRAIN pairs, parent checkpoint, data split|Research code from the original experiments; must not start automatically|

This review does not install a new environment or start training or inference. The fresh-install commands come from version and build records for the working environment;
**an end-to-end reinstall on another fresh server has not yet been verified**.

## 1. Code and Verified Versions

```bash
git clone https://github.com/menyunlei/map2world-4D-.git
cd map2world-4D-
export PROJECT_ROOT="$PWD"
export NATIVE_ROOT="$PROJECT_ROOT/experiments/recursive_map_research_v1/trellis2_native_quality_v1"
```

The working server was checked read-only on 2026-10-15: Linux, Python **3.10.22**,
PyTorch **2.6.0+cu124**, torchvision **0.21.0+cu124**, CUDA Toolkit **12.4.99**,
FlashAttention **2.7.3**, and Triton **3.2.0**. The TRELLIS.2 worktree was clean at that time.
You need a driver supporting CUDA 12.4, a CUDA compiler, C/C++ build tools, Git, and Ninja.
The working hardware was an A800 80GB; compatibility with other VRAM capacities is not guaranteed.

Exact source commits are recorded in [`reproduction-lock.json`](reproduction-lock.json).
Note: the TRELLIS.2 commit in the old PLAN contains an **incorrect 41-character hash**. The correct value is:

```text
75fbf0183001ed9876c8dbb35de6b68552ee08bd
```

Keep the old files as historical evidence; do not check out the incorrect commit.

## 2. Install the Native CUDA Environment

On a Linux host with Conda/Mamba already installed, use the layout below. If Conda is unavailable, install it using its official instructions first.
Do not use the network proxy or installation recovery scripts from old experiments that were specific to the original machine.

```bash
mkdir -p "$NATIVE_ROOT/vendor" "$NATIVE_ROOT/weights"
conda create -y -p "$NATIVE_ROOT/environment" python=3.10.22
conda activate "$NATIVE_ROOT/environment"
# If a complete CUDA 12.4 Toolkit is already installed, you can point CUDA_HOME to that directory.
# Otherwise, install the Toolkit using the working environment's layout; availability of older packages depends on the package source.
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.0  # A800; set this to the actual architecture when using a different GPU
export MAX_JOBS=4
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$PROJECT_ROOT/requirements/native.txt"
```

Clone and pin the third-party sources; do not replace the pinned versions with the latest main branch:

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
# The working records show the final build used the nvdiffrec_v3 root directory, not the /render installation subdirectory in old scripts.
python -m pip install --no-build-isolation "$NATIVE_ROOT/vendor/nvdiffrec_v3"
python -m pip install --no-build-isolation "$NATIVE_ROOT/vendor/CuMesh"
python -m pip install --no-build-isolation "$NATIVE_ROOT/vendor/FlexGEMM"
python -m pip install --no-build-isolation 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
python -m pip install --no-build-isolation "$NATIVE_ROOT/vendor/TRELLIS.2/o-voxel"
```

This FlashAttention wheel is only for CPython 3.10/Linux x86_64, torch 2.6/CUDA12,
and CXX11 ABI FALSE; do not force installation with other ABIs. The working machine previously encountered CUDA header/NVRTC linker path issues:
if you use pip NVIDIA libraries with an incomplete Toolkit, the compiler must be able to find the relevant `site-packages/nvidia/*/include`
and `*/lib` directories. Prefer installing a complete Toolkit; do not modify system libraries or blindly reuse symlinks from the old machine.

Add the native package to the environment (this review preserves the upstream source without modifying files inside it):

```bash
export PYTHONPATH="$NATIVE_ROOT/vendor/TRELLIS.2${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import importlib.util; print({n:importlib.util.find_spec(n) is not None for n in ['flash_attn','cumesh','flex_gemm','o_voxel','nvdiffrast','utils3d','nvdiffrec_render']})"
```

This step checks whether the packages are installed; **it does not verify CUDA computation**. Actual FlexGEMM/Triton imports or operators
may initialize CUDA; perform runtime checks only on an authorized, idle GPU.

## 3. Weight Directories and Runtime Configuration

Keep each checkpoint's `.json` network configuration and matching `.safetensors` file; do not copy only the weights.

```text
$NATIVE_ROOT/weights/
  TRELLIS.2-4B/pipeline.json
  TRELLIS.2-4B/ckpts/{model_name}.json + {model_name}.safetensors
  TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.{json,safetensors}
  dinov3/config.json + model.safetensors (+ the original model's other configuration files)
```

The respective weight sources are `microsoft/TRELLIS.2-4B`, `microsoft/TRELLIS-image-large`, and
`facebook/dinov3-vitl16-pretrain-lvd1689m`. DINO may require the user to obtain access approval and log in first;
do not write tokens into the repository. When downloads are authorized, use the Hugging Face CLI in the environment:

```bash
hf download microsoft/TRELLIS.2-4B --include pipeline.json 'ckpts/*.json' 'ckpts/*.safetensors' --local-dir "$NATIVE_ROOT/weights/TRELLIS.2-4B"
hf download microsoft/TRELLIS-image-large --include 'ckpts/ss_dec_conv3d_16l8_fp16.*' --local-dir "$NATIVE_ROOT/weights/TRELLIS-image-large"
hf download facebook/dinov3-vitl16-pretrain-lvd1689m --local-dir "$NATIVE_ROOT/weights/dinov3"
```

Corresponding locations on the original working machine (for locating the original assets; the new machine does not need to use the same absolute paths):

```text
JOB=/hpc2hdd/home/xhe989/Documents/2541/project2/DreamX-World-master/experiments/recursive_map_research_v1
TRELLIS2=$JOB/trellis2_native_quality_v1/weights/TRELLIS.2-4B
SS_DECODER=$JOB/trellis2_native_quality_v1/weights/TRELLIS-image-large
DINO=/hpc2hdd/home/xhe989/Documents/ExGRPO/qwen/facebook/dinov3-vitl16-pretrain-lvd1689m
RUNTIME=$JOB/map_object_world_v1/trellis2_resume_v3/runtime/pipeline.json
```

Read-only checks confirmed that all eight sets of inference model JSON/weight files existed on the original working machine. For the full names, see
the `expected` table in `tools/reproduce.py`. DINO weights SHA256:
`dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
Download revisions were not originally recorded for the other models. Strict reproduction requires copying the original weights and creating a SHA256 manifest for each;
the latest snapshot downloaded today cannot be guaranteed to match the original weights byte for byte.

Use the tool to generate a new runtime configuration; it does not edit historical configurations or overwrite existing files:

```bash
python tools/reproduce.py configure \
  --template "$NATIVE_ROOT/weights/TRELLIS.2-4B/pipeline.json" \
  --weights "$NATIVE_ROOT/weights" --dino "$NATIVE_ROOT/weights/dinov3" \
  --output "$PROJECT_ROOT/experiments/recursive_map_research_v1/map_object_world_v1/repro_runtime/pipeline.json"
```

The tool checks all eight file sets and the DINO hash, preserves native Shape/Texture normalization, and reuses the verified
sampler parameters and `low_vram=true`. `UnusedBackgroundRemoval` depends on the shim in the existing native entry point;
this pipeline.json cannot be used directly as an unmodified official demo configuration.

## 4. One Native Inference Check (No Training or Claim of Map Control)

The input must be a fixed RGB image prepared in advance, composited onto a white background and padded to a square according to the original experiment's rules.
Neither source images nor processed images are included in this code upload. Copy them from the original run package, or process an official example using
the rules in `trellis2_native_street_v17.prepare()`. Substituting your own image does not exactly reproduce the historical output.

First check paths on the CPU without triggering model inference:

```bash
python tools/native_smoke.py --native-root "$NATIVE_ROOT" \
  --runtime "$PROJECT_ROOT/experiments/recursive_map_research_v1/map_object_world_v1/repro_runtime" \
  --input /absolute/path/to/fixed_padded_input.png
```

After manually confirming GPU ownership, query UUIDs and select only an authorized, idle GPU:

```bash
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
# Replace the UUID; do not copy the placeholder unchanged. Run at most once; inspect logs after a failure and do not retry automatically.
timeout --signal=TERM 1200s python -B -u tools/native_smoke.py \
  --native-root "$NATIVE_ROOT" \
  --runtime "$PROJECT_ROOT/experiments/recursive_map_research_v1/map_object_world_v1/repro_runtime" \
  --input /absolute/path/to/fixed_padded_input.png \
  --seed 942217 --output-name reproduction_native_smoke --gpu GPU-REPLACE-WITH-AUTHORIZED-UUID --run
```

The wrapper overrides old paths only within the current process and calls the original v17 generator; it runs inference once without modifying model weights.
It rejects existing output directories and GPUs with an existing compute owner. This new wrapper has passed CPU argument/code checks,
but has not yet been rerun on a GPU. A run should produce `native_street_basecolor.glb`, `RAW_MESH.npz`, `RAW_NATIVE.pt`, and
`ASSET.json`. The GLB is a preview containing all native triangles and vertex base colors, **not a complete UV/PBR asset**.
Assess quality only after reading the asset hashes/face counts and validating new viewpoints; the process exit code alone is insufficient.

## 5. Viewers and Online Services

Use a separate **Python 3.11+** environment for the local client, then install `requirements/client.txt`.
The native CUDA environment uses Python 3.10; do not mix the environments. The old viewers/services use
`hashlib.file_digest`, which is available only from 3.11 onward, so starting them directly with the native 3.10 environment may fail.

```bash
conda create -y -n map2world-client python=3.11
conda activate map2world-client
python -m pip install -r "$PROJECT_ROOT/requirements/client.txt"
```

Three.js/OrbitControls and the MIT license are included in the repository; no frontend npm build is required.

For offline v21 viewing, first transfer `trellis2_long_street_v21/large_street_100m.glb`
from the original machine, along with `trellis2_streaming_v22/STREAM_STATE.json` (preserving the original asset hash) and `map10_reference.png`.
Adjacent-chunk mode also requires `trellis2_streaming_v22/chunk/native-adjacent-001/chunk_100m.glb`;
without that asset, check only v21 and the map, and do not claim adjacent-chunk mode is available. Run from the project script directory:

```bash
cd "$PROJECT_ROOT/experiments/recursive_map_research_v1"
python -B -u v22_server.py --port 8767
```

The server listens only on loopback. If the browser is on your own computer, use port forwarding; do not expose plain HTTP publicly:

```bash
ssh -N -L 8767:127.0.0.1:8767 YOUR_USER@YOUR_SERVER -p YOUR_SSH_PORT
# Local browser: http://127.0.0.1:8767/
```

The code alone is insufficient to start v27 online mode. Synchronize inputs and state from **the same registered experiment**:
`CONTRACT.json`, `SOURCE_FREEZE.json`, `BASELINE_CACHE.json`, `EVALUATION_QUERIES.npz`,
`MAP10_PROVENANCE.json`, `map10_scaffold.json`, `map10_reference.png`,
`PLAN.json/input.png/ROAD_OBSERVATION.npz` for both attempts, and the corresponding v21/v24 parent assets.
Also preserve any existing `STATE.json`, reports, hashes, and budget records from generation; do not delete state to reset the budget.
The scripts strictly verify source hashes. The current viewer includes display fixes, so do not falsely claim it is identical to the old source freeze.

Migrating v27 requires explicitly adapting the following hardcoded items; do not perform global replacements across all historical files:

1. ROOT/JOB in `paths.py` (used by historical training/workers).
2. REMOTE/PYTHON/SSH addresses, usernames, and ports in `v23_server.py` and `v27_server.py`.
3. The GPU UUID and central lease path in `v26_resident.py`/owner; verify ownership on the new machine.
4. Input/runtime paths and valid asset hashes in the registered PLAN; save migration configurations separately without overwriting the original records.
5. Put the host public key, verified through an out-of-band channel, in the client's `work_known_hosts`; enter passwords using `getpass` or configure keys yourself.
   Do not disable host-key verification or write passwords into scripts.

Only after these prerequisites are met should you run `python -B -u v27_live.py` and browse `http://127.0.0.1:8772/`.
This entry point triggers GPU work through camera activity, unlike the read-only v22 viewer. Refreshing or moving the camera does not
automatically grant a new budget to experiments that have already ended. This repository does not silently enable new online experiments or training.

## 6. Code Integrity and Known Limitations

```bash
python tools/reproduce.py audit
python -m unittest discover -s tools -p 'test_reproduce.py' -v
```

The audit uses AST to check the syntax of tracked Python files and sibling-module dependencies visible on the local machine;
it does not import or execute training scripts. It does not establish that external DreamX/TRELLIS packages, dynamically loaded data, reference images, or
checkpoints are complete. Some historical G2-B/original TRELLIS experiments still depend on the original DreamX repository and its data;
that portion is not a prerequisite for this native TRELLIS.2 reproduction and cannot be claimed to be independently packaged.

Georegistration, visual realism, local editing, and map control must still be evaluated separately. Successful cloning, installation, and GLB export
do not change the historical failure conclusions. Large weights, large assets, and private authentication information remain excluded by `.gitignore`.
