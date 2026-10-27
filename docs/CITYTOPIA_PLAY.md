# CityTopia Map10 player and four-GPU native renderer

This is the source of the player served at `http://127.0.0.1:18936/` on the
original workstation. It uses the preserved CityDreamer4D CityTopia background
and building checkpoints, Map10 spatial conditions, and fixed building styles.

## Entry points

- Local HTTP/SSH relay: `experiments/recursive_map_research_v1/citytopia_play_bridge.py`.
- Remote native service: `experiments/recursive_map_research_v1/map_object_world_v1/citytopia_play_v1/native_balanced_v5.py`.
- Browser navigation and camera/screenshot capture: `citytopia_play_v1/play.js`.
- Saved-view RGB-D preview preparation: `citytopia_play_v1/prepare_preview.py`.
- Same-camera measurements: `citytopia_play_v1/SPEED_CHECK_V4.json`.

The versioned renderer imports v4 (visible-building regions), v3 (device transfer),
v2 (cached encoders and dynamic tile queue), and `native_service.py` (API and
world setup). Keep these files together. `native_four_gpu.py` is the previous
row-per-GPU baseline. The service listens only on remote localhost port 18935.

## Existing environment and data

Use the verified CityDreamer4D environment and original CityDreamer4D checkout
beside `DreamX-World-master`; the source overlay retains upstream headers and is
covered by its accompanying `UPSTREAM_LICENSE.txt` (S-Lab License 1.0,
non-commercial use). This does not relicense the surrounding project.

The required external artifacts are not committed:

- `DreamX-World-master/inputs/citypth/CityDreamer4D-BG.pth` and `CityDreamer4D-BLDG.pth`.
- `citydreamer4d_building_capability_20261026_v1/city_sample_map10/`, including
  projections, footprints and its manifest.
- `citytopia_map10_checkpoint_trial_20261027_v1/run_v1/CONTRACT.json`,
  `WORLD_STYLES.pt` and `CAMERAS.json`, with the preserved sources they reference.
- Existing native route images/depths in `citytopia_road_diagnosis_20261027_v1/route_density5_v2/`
  for navigation preview preparation, or its already prepared `citytopia_play_v1/preview/`.

Paths are resolved relative to the existing server layout; the local relay also
retains the original SSH address, username and remote project path. Adjust these
and the GPU UUIDs in `native_service.py` when moving machines. Provide a verified
SSH known-host entry; the relay does not accept unknown host keys automatically.

## Launch on the existing setup

With the preserved scene and model environment available, start the remote
service (only one service may own port 18935):

```bash
python -B -u experiments/recursive_map_research_v1/map_object_world_v1/citytopia_play_v1/native_balanced_v5.py
```

On the workstation, run the relay using the client environment with Paramiko:

```python
import getpass, os, runpy
os.environ["MAP_CITY_SSH_PASSWORD"] = getpass.getpass("SSH password: ")
runpy.run_path("experiments/recursive_map_research_v1/citytopia_play_bridge.py", run_name="__main__")
```

Open `http://127.0.0.1:18936/`. WASD moves, mouse dragging rotates, Q/E changes
altitude in free-flight mode, and R requests the current native view. Camera JSON
and screenshots can be saved from the toolbar. Moving submits the latest pose
while one native render is in flight; navigation remains responsive.

## What was measured

At the two matched saved cameras in `SPEED_CHECK_V4.json`, warm native frame
computation was approximately 0.85–1.12 seconds, versus approximately 5–6 seconds
in the row-per-GPU baseline. Later browser-movement requests were approximately
0.66–0.69 seconds at their respective poses. These are server times, not an
end-to-end latency guarantee or 60 FPS native rendering. Loading the released
world takes approximately 12 seconds; the service releases its four worker
allocations after 120 idle seconds and checks that GPU0–3 are idle before loading.

The matched-camera numerical differences were at most two uint8 levels, mean
absolute difference below 0.003/255, and PSNR above 73 dB. Cropping changes CUDA
operation shapes and their rounding; the outputs are not bit-identical. The
original resolution, ray sampling and eight-pixel CNN context are retained.

Navigation uses saved RGB with coarse depth and was observed at about 28–32 FPS
on the original workstation. It is a preview of the generated scene. Native
rendering retains the model's road and facade defects. Faster rendering does
not establish improved geometry, photorealism, new persistent detail generation,
or learned 4D dynamics.
