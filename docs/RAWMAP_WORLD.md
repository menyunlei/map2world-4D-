# Experiment B, raw-map branch: RGB map image → learned world → player

Code: `experiments/recursive_map_research_v1/map_object_world_v1/rawmap_world_v1/`.
The CityDreamer4D baseline, its player path and Experiment A are unchanged.

**Status: code and CPU checks only.** No GPU was available when this was written, so
there is no trained checkpoint and no playable candidate yet. Nothing here is
evidence that the model understands maps until the GPU steps below are run.

## Question

Can a representation learned from a raw RGB map image directly control a
generated street, with no semantic-map conversion in the inference path?

## Why the input encoder alone could not be swapped

In the local overlay, structured map inputs are used far beyond the first layer:

| Dependency | Where |
|---|---|
| Height field + one-hot semantics as encoder input | `get_hf_seg_tensor`, `GanCraftGenerator.encoder` |
| 3D support: footprints extruded by heights into a seg volume; only those voxels are ever sampled | `get_seg_volume`, `extrude_footprint`, `get_voxel_intersection_perspective` |
| Per-voxel class one-hot inside the MLP | `_forward_perpix` (`seg_map_pov_onehot`) → `RenderMLP.fc_m_a` |
| Building instances route rays to a separate generator, frame and style | `get_visible_buildings`, `render_bldg`, `ftp_stats` |
| Road-only blur keyed by class | `render_bg` road mask |
| Traffic placement from structured lanes | `traffic_scenario_generator` |

So the new branch replaces the whole path from map to 3D, and keeps CityDreamer4D
only as an offline teacher and as the baseline in the player.

## Method

- **Map memory** `Z_M = E(I_map)`: frozen DINOv3/DINOv2 patch grid (`--backbone hf:<path>`),
  a trainable full-resolution RGB stem (/4, /8) for thin roads, and a relational
  transformer over map tokens with absolute 2D positions. Computed once per map.
- **World decoder** `W = G(Z_M, z)`: one query per 1 m map column of the region,
  reading the memory with deformable attention around its own location, then a
  stack of height tokens with column self-attention. Each token emits K Gaussians
  (position within its cell, scale, rotation, opacity, colour). Heights and
  geometry are predicted, never supplied.
- **World code** `z`: fixed per world (`--seed`, `-1` = prior mean). Camera motion
  does not resample content. Training uses one learned code per training scene.
- **Rendering** `I = R(W, C)`: gsplat on GPU (torch reference renderer for tests)
  plus a small sky model conditioned on `z`. Moving the camera only re-renders the saved world.

Out of scope for this first branch: progressive refinement, learned traffic, new
viewer. Existing dynamic demos stay separate and are not part of the raw-map claim.

## Data

One folder per scene; format in `data.py`. The loader rejects a scene unless
`student_inputs == ["map.png"]`. Splits are by scene and checked for leaks.

`export_teacher.py` produces scenes from the preserved Map10 world with the
unmodified CityDreamer4D renderer. Teacher depth is the first-voxel entry of the
extruded map volume: a proxy of the teacher's geometry, labelled as such. It also
checks the camera convention against the native caster's ray directions and stops
if it cannot reproduce them (min cosine > 0.99999).

The **RGB map image must be supplied** (`--map-image`) with its bounds. A
semantic-colour rendering of the Map10 layout would leak the teacher's inputs and
does not count as a raw map. If no aligned raw map exists for Map10, this
experiment is blocked on data, not on code.

Held-out streets from the same city test layout generalisation within Map10
only; they are not evidence for new cities.

## CPU checks (done)

```bash
cd experiments/recursive_map_research_v1/map_object_world_v1/rawmap_world_v1
python -m unittest test_rawmap -v
```

Nine tests: camera convention and projection, convention recovery from native
rays, reference splatting and gradients, world generated once and unchanged by
camera motion, map dependence of the untrained network, gradients reaching every
trainable part, input contract and split checks, controls, and a synthetic
train → evaluate run. The synthetic run only shows the pipeline learns; it is not a result.

## GPU experiment (to run)

Environment: the existing CityDreamer4D environment for export; the TRELLIS.2 /
native environment (has `transformers`, `lpips`) plus `pip install gsplat` for
training. One idle GPU per job, by UUID; scripts refuse busy GPUs.

```bash
# 1. Teacher scenes: >= 3 train streets, 1 val, 1 test, all from different streets.
python export_teacher.py --name street_a --split train --map-image MAP.png \
  --map-bounds r0,r1,c0,c1 --bounds-frame projection --region r0,r1,c0,c1,z0,z1 \
  --path "r,c;r,c" --out ../rawmap_scenes

# 2. Bounded pilot
python train.py --scenes ../rawmap_scenes --out ../rawmap_runs/pilot_v1 \
  --backbone hf:/path/to/dinov3-vitl16-pretrain-lvd1689m --steps 20000 --max-seconds 21600

# 3. Held-out evaluation with controls (same cameras, fixed seed)
python evaluate.py --checkpoint ../rawmap_runs/pilot_v1/checkpoint.pt \
  --scenes ../rawmap_scenes --split test --out ../rawmap_runs/pilot_v1/eval_test

# 4. Play it: candidate service beside the native one, then the usual bridge
python service.py --checkpoint ../rawmap_runs/pilot_v1/checkpoint.pt --scene ../rawmap_scenes/street_test
```

In the player (`http://127.0.0.1:18936/`), choose **Raw-map candidate** in the
model menu and press R. Native and candidate frames are labelled differently;
the RGB-D navigation preview is neither. Use `--map` for an edited or restyled
map with the same bounds and seed.

## Success criteria for the first claim

On the held-out test street, same cameras and seed:

1. Map dependence: real map beats blank and shuffled by ≥ 3 dB PSNR, and coverage
   IoU vs teacher surfaces ≥ 0.7. Below that, generation is not driven by the map.
2. Styling robustness: a restyled map with the same layout stays within 1 dB of real.
3. Controlled edit: with an edited map and teacher views of the expected world,
   the edited prediction scores better on those views than the unedited map does.
4. Play test: walking, turning at a junction, close approach and sideways motion
   show roads and both building rows where the map has them, without view-dependent popping.

Reported separately and not part of the claim: cold-start seconds (encode +
generate), per-camera render seconds, and visual quality vs the native baseline.
The teacher is blurry at street level, so matching it cannot show better realism.
