# map2world-4D-

Experimental Map-to-3D research code built around TRELLIS.2, including learned
map-conditioning experiments, native street generation, bounded asynchronous
generation, mesh checks, and an interactive persistent-world viewer.

**Reproduction:** [installation, exact dependency commits, weight paths and launch steps](docs/REPRODUCING.md).
The verified environment lock is [here](docs/reproduction-lock.json).
CPU checks: `python tools/reproduce.py audit` and
`python -m unittest discover -s tools -p 'test_reproduce.py' -v`.

## Code map

All project scripts live in `experiments/recursive_map_research_v1/`:

- `trellis2_map_object_module_v1.py`, `trellis2_map_geometry_token_module_v3.py`,
  `trellis2_structure_lora_v8.py`: map/object conditioning and Structure adapters.
- `trellis2_*train*.py`, `trellis2_*generate*.py`: versioned training/inference
  experiments. Historical failed candidates are not endorsements or defaults.
- `trellis2_native_street_v17.py`, `trellis2_long_street_v21*.py`: native
  image-conditioned street generation and fixed export/render tools.
- `v23_*` through `v27_*`: bounded camera-triggered generation, cache, ownership
  checks and viewer services. `v27_live.py` is the latest existing live-service
  entry point; its viewer files are in
  `map_object_world_v1/trellis2_autonomous_v27/`.
- `v28_*`: Structure/Shape support-resolution attribution and fixed saved-mesh
  comparisons; the tested entry change was insufficient, not a successful fix.
- `paths.py`: original server workspace and data paths.

Viewer navigation supports WASD, mouse look, lateral movement and revisits.
Display options include a gradient sky and switchable base-color interpretation.
These are display changes, not improvements to generated geometry.

## Dependencies and reproducibility limits

This is a source archive of existing experiments, not a self-contained packaged
application. It does **not** include weights, checkpoints, training data, reference
images, generated meshes, latents, environments, runtime logs, SSH host keys or
credentials. TRELLIS.2 and its official CUDA dependencies must be installed
separately from their upstream source; no third-party model license is replaced
by this repository. The existing Three.js browser runtime is included with its
MIT license.

The working setup uses a CUDA Linux A800 server, a separately installed
TRELLIS.2 checkout/runtime, and a local Python service using Paramiko. Scripts
also use PyTorch, NumPy, Pillow, trimesh and nvdiffrast; exact compatible versions
belong to the original model environment and are not newly specified here.
Review `trellis2_prepare_v1.py` and the relevant experiment configuration before
reproducing a run. Some scripts apply version-specific compatibility patches.

The archived scripts retain original workspace paths, server addresses, GPU
UUIDs and historical experiment hashes. Update those for your environment and
provide your own verified SSH host-key file; credentials are entered through
`getpass`, not committed. Inputs/configurations and generated assets referred to
by older runs may be absent. Do not blindly execute setup/owner scripts: they can
launch GPU work. Historical source hashes describe original runs and may differ
from later viewer-only changes.

With the original prerequisites and assets in place, the existing local service
entry is:

```sh
cd experiments/recursive_map_research_v1
python -B -u v27_live.py
# Browser: http://127.0.0.1:8772/
```

This command requires the server environment, valid SSH connection and the
experiment's registered inputs; cloning alone is not enough. Do not bypass GPU
ownership checks or reset an experiment's bounded budget to trigger new work.

## Scientific status

Native image-conditioned street generation is not evidence of Map10 control.
Map10 annotations include roof/road proxies from an oblique image, not surveyed
footprints. Failed map-control runs remain failed. Display colors, file export,
cache persistence and successful loading do not establish Map-to-World success.
The repository contains research implementations, not a claim of a solved
real-time, map-aligned city generator.
