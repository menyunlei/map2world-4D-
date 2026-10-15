"""Bounded CPU map/target registration check; prints JSON and writes no data.

No backbone, sampler, latent features, training, network, or remote paths opened.
Remote plan paths are mapped only to their exact local experiments/ suffix.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from trellis2_map_geometry_token_module_v3 import Trellis2MapGeometryTokenConditioner
from trellis2_map_object_module_v1 import native_world_grid

RASTERS = ("footprints", "footprint_known", "roads", "roads_known",
           "open_space", "open_space_known")
JOB = Path(__file__).resolve().parent
ROOT = JOB.parent.parent
BASE = JOB / "map_object_world_v1"
V3 = BASE / "trellis2_map_geometry_tokens_v3"
PARENT = BASE / "trellis2_map_learning_v1/trial_fix1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def local(path):
    path = str(path).replace("\\", "/")
    if "/experiments/" in path:
        return ROOT / "experiments" / path.split("/experiments/", 1)[1]
    return Path(path)


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def pool(array):
    return array.reshape(*array.shape[:-2], 16, 4, 16, 4).mean((-3, -1))


def tensors(arrays):
    return {key: torch.from_numpy(value.copy()).unsqueeze(0)
            for key, value in arrays.items()}


def raster_summary(array):
    y, x = np.indices((64, 64))
    rows, cols = np.where(array > 0)
    return np.array([((x + .5) * array).sum() / array.sum() / 64 - .5,
                     ((y + .5) * array).sum() / array.sum() / 64 - .5,
                     (cols.max() - cols.min() + 1) / 64,
                     (rows.max() - rows.min() + 1) / 64, array.mean()])


def audit_map(path, branch):
    arrays = load(path)
    batch = tensors(arrays)
    context = branch.prepare(batch, native_world_grid())
    valid = arrays["object_valid"]
    confidence = arrays["object_summary"][:, 5] * valid
    expected = pool(arrays["footprints"] * arrays["footprint_known"])
    expected = (expected * confidence[:, None, None]).transpose(2, 1, 0)
    expected = np.broadcast_to(expected[:, :, None, :], (16, 16, 16, len(valid)))
    measured = context.routing_weights.numpy().reshape(16, 16, 16, len(valid))
    reversed_batch = {**batch, **{key: batch[key].flip(-2) for key in RASTERS},
                      "mapYrow": "decreasing"}
    reversed_context = branch.prepare(reversed_batch, native_world_grid())
    zeros = {**batch, **{key: torch.zeros_like(batch[key])
                        for key in ("footprint_known", "roads_known", "open_space_known")}}
    zeros["object_summary"] = batch["object_summary"].clone()
    zeros["object_summary"][..., 5] = 0
    unknown = branch.prepare(zeros, native_world_grid())
    changed = {**zeros, **{key: 1 - zeros[key]
                          for key in ("footprints", "roads", "open_space")}}
    changed_context = branch.prepare(changed, native_world_grid())
    summary_delta = [np.abs(raster_summary(arrays["footprints"][i]) -
                            arrays["object_summary"][i, :5]).tolist()
                     for i in np.flatnonzero(valid)]
    known_xy = np.maximum.reduce([
        (pool(arrays["footprint_known"]) * confidence[:, None, None]).max(0),
        pool(arrays["roads_known"])[0], pool(arrays["open_space_known"])[0]])
    unknown_xy = torch.from_numpy((known_xy == 0).T.reshape(-1))
    area_loss = np.abs(pool(arrays["footprints"]).sum((-2, -1)) * 16 -
                       arrays["footprints"].sum((-2, -1))).max()
    return {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "valid_objects": int(valid.sum()), "finite": all(np.isfinite(v).all() for v in arrays.values()),
        "raster_range": [min(float(arrays[k].min()) for k in RASTERS),
                         max(float(arrays[k].max()) for k in RASTERS)],
        "routing_max_abs_error": float(np.abs(expected - measured).max()),
        "routing_Z_max_range": float(np.ptp(measured, axis=2).max()),
        "row_flip_equivalence_max_error": float((context.token_features -
                                                   reversed_context.token_features).abs().max()),
        "pool_mass_max_abs_error_source_pixels": float(area_loss),
        "support_pixels_per_object": arrays["footprints"].sum((-2, -1)).tolist(),
        "supported_coarse_cells_per_object": (measured[:, :, 0, :] > 0).sum((0, 1)).tolist(),
        "summary_minus_raster_abs_X_Y_extentX_extentY_area": summary_delta,
        "unknown_XY_cells": int(unknown_xy.sum()),
        "unknown_context_max_abs": float(unknown.token_features.abs().max()),
        "unknown_hidden_raster_edit_max_error": float((unknown.token_features -
                                                        changed_context.token_features).abs().max()),
        "known_empty_source_pixels": {key: int(((arrays[key] == 0) &
                                                 (arrays[known] > 0)).sum())
                                      for key, known in zip(RASTERS[::2], RASTERS[1::2])},
    }, batch, context, unknown, unknown_xy


def registration(plan_path):
    plan = read(plan_path)
    fields = {"map_input": "map_sha256", "registered_map": "registered_map_sha256",
              "source_manifest": "manifest_sha256", "target_mesh": "mesh_sha256"}
    available = {key: 0 for key in fields}
    bad_hashes, mesh_errors = [], []
    for record in plan["records"]:
        for key, hash_key in fields.items():
            path = local(record[key])
            if not path.is_file():
                continue
            available[key] += 1
            if hashlib.sha256(path.read_bytes()).hexdigest() != record[hash_key]:
                bad_hashes.append([record["scene_id"], key])
            if key == "target_mesh":
                mesh = load(path)
                expected = (mesh["vertices_world_m"] - np.array([0, 0, 14])) / 32
                mesh_errors.append(float(np.abs(expected - mesh["vertices_native"]).max()))
    inherited = read(PARENT / "PLAN.json")["records"]
    return {"path": str(plan_path), "records": len(plan["records"]),
            "splits": {split: sum(r["split"] == split for r in plan["records"])
                       for split in ("adapter_TRAIN", "adapter_HELDOUT")},
            "records_identical_to_parent": plan["records"] == inherited,
            "exact_local_payload_availability": available, "hash_mismatches": bad_hashes,
            "mesh_native_vs_world_max_abs": max(mesh_errors) if mesh_errors else None,
            "native_shape_target_npz_available": sum((V3 / "targets" /
                                                       (r["scene_id"] + ".npz")).is_file()
                                                      for r in plan["records"])}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=V3 / "PLAN.json")
    parser.add_argument("--map", type=Path, default=PARENT / "real_input/map.npz")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    branch = Trellis2MapGeometryTokenConditioner().cpu().eval()
    report = {"registration": registration(args.plan), "maps": [], "trained_head_support": []}
    primary, batch, context, unknown, unknown_xy = audit_map(args.map, branch)
    report["maps"].append(primary)
    for name in ("removed_last_building_map.npz", "shuffled_map.npz"):
        path = args.map.parent / name
        if path.is_file():
            report["maps"].append(audit_map(path, branch)[0])
    metadata = args.map.parent / "MAP_INPUT.json"
    if metadata.is_file():
        report["source_map_row_convention"] = read(metadata)["coordinate_transform"]["mapYrow"]
        from trellis2_map_real_input_v1 import build_arrays
        saved, rebuilt = load(args.map), build_arrays()
        report["source_polygon_rebuild_max_abs_by_field"] = {
            key: float(np.abs(saved[key].astype(float) - rebuilt[key].astype(float)).max())
            for key in saved}
    for arm in ("original", "separated"):
        for seed in (953101, 953102):
            path = V3 / arm / f"seed{seed}/trained_map_condition.npz"
            if not path.is_file():
                continue
            branch.load_state_dict({key: torch.from_numpy(value) for key, value in load(path).items()})
            actual = branch.prepare(batch, native_world_grid())
            tokens = branch.map_tokens(actual)[0]
            zero_tokens = branch.map_tokens(unknown)[0]
            pooled = actual.token_features.reshape(1, 16, 16, 16, 128).mean(3).reshape(1, 256, 128)
            report["trained_head_support"].append({
                "arm": arm, "seed": seed, "path": str(path),
                "unknown_context_max_abs": float(actual.token_features.reshape(1, 16, 16, 16, 128)
                                                   .mean(3).reshape(256, 128)[unknown_xy].abs().max()),
                "unknown_XY_tokens_nonzero": int((tokens[unknown_xy].abs().amax(1) > 0).sum()),
                "unknown_token_RMS": float(tokens[unknown_xy].square().mean().sqrt()),
                "unknown_token_max_abs": float(tokens[unknown_xy].abs().max()),
                "known_token_RMS": float(tokens[~unknown_xy].square().mean().sqrt()),
                "fully_unknown_map_token_RMS": float(zero_tokens.square().mean().sqrt()),
                "unknown_token_equals_head_zero_max_error": float((tokens[unknown_xy] - zero_tokens[unknown_xy]).abs().max()),
                "Z_pool_before_head_max_error": float((branch.map_condition_head(pooled)[0] - tokens).abs().max()),
                "negative_CFG_max_abs": float(branch(actual, negative_cfg=True).features.abs().max()),
            })
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
