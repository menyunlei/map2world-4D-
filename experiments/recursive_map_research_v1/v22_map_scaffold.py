"""Export the existing Map10 annotations as a bounded, CPU-only viewer scaffold.

Run: python -B experiments/recursive_map_research_v1/v22_map_scaffold.py
Check existing outputs without writing: add --check.

All polygons are existing manual roof/road proxies, not AI geometry. This
module never imports paths.JOB, torch, a model, a viewer, or a remote client.
Only the three named v22 artifacts below are written. No photo is retraced.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


JOB = Path(__file__).resolve().parent
WORLD = JOB / "map_object_world_v1"
OUTPUT = WORLD / "trellis2_streaming_v22"
METADATA = WORLD / "trellis2_map10_block_v11/inputs/MAP_INPUT.json"
RASTER_METADATA = WORLD / "trellis2_map_learning_v1/trial_fix1/real_input/MAP_INPUT.json"
RASTER = RASTER_METADATA.with_name("map.npz")
REFERENCE = METADATA.with_name("source_crop.png")
COMPILER = JOB / "trellis2_map_real_input_v1.py"
SPAN = 100.0
HEIGHT = 4.0
CHUNKS_PER_AXIS = 4
LABEL = "Map-derived visualization scaffold; not AI-generated geometry"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    """Portable paths relative to experiments/recursive_map_research_v1."""
    return Path(path).resolve().relative_to(JOB).as_posix()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def source_record(path, expected=None):
    actual = sha256(path)
    if expected is not None:
        require(actual == expected, f"Source hash mismatch: {path}")
    return {"path": relative(path), "sha256": actual,
            "expected_sha256": expected, "expected_hash_verified": expected is not None}


def load_inputs():
    metadata, raster_metadata = read_json(METADATA), read_json(RASTER_METADATA)
    # These historical manifests differ only in deployment paths and archive
    # bytes. Never claim that the local NPZ has the unavailable v11 NPZ's hash.
    semantic_keys = (set(metadata) | set(raster_metadata)) - {"source", "artifacts"}
    require(all(metadata.get(k) == raster_metadata.get(k) for k in semantic_keys),
            "Local raster and v11 annotations differ semantically")
    sources = {
        "v11_metadata": source_record(METADATA),
        "raster_metadata": source_record(RASTER_METADATA),
        "raster_archive": source_record(RASTER, raster_metadata["artifacts"]["map.npz"]["sha256"]),
        "source_image": source_record(JOB / "map10_source.jpg", metadata["source_sha256"]),
        "annotation_compiler": source_record(COMPILER, metadata["annotation_script_sha256"]),
        "reference_crop": source_record(REFERENCE, metadata["artifacts"]["source_crop.png"]["sha256"]),
        "scaffold_exporter": source_record(Path(__file__)),
    }
    # Execute only the hash-verified, existing CPU annotation compiler's pure
    # array builder, in memory. Its export/main routines are never called.
    spec = importlib.util.spec_from_file_location("map10_fixed_cpu_compiler", COMPILER)
    compiler = importlib.util.module_from_spec(spec)
    # Source compilation avoids creating __pycache__ outside our write scope.
    exec(compile(COMPILER.read_bytes(), str(COMPILER), "exec"), compiler.__dict__)
    rebuilt = compiler.build_arrays()
    with np.load(RASTER, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    compiler.validate_arrays(arrays)
    require(set(arrays) == set(rebuilt), "Compiled raster field mismatch")
    require(all(np.array_equal(arrays[k], rebuilt[k]) for k in arrays),
            "Existing raster does not exactly reproduce with the fixed CPU compiler")
    require(metadata["object_ids"] == [item["id"] for item in compiler.OBJECTS],
            "Persistent object IDs differ from the original compiler")
    for items, fixed in ((metadata["objects"], compiler.OBJECTS), (metadata["roads"], compiler.ROADS)):
        require(len(items) == len(fixed), "Annotation count changed")
        for item, original in zip(items, fixed):
            require(all(item[k] == value for k, value in original.items()),
                    f"Fixed annotation changed: {item['id']}")
    require(metadata["zero_confidence_occlusions"] == list(compiler.OCCLUSIONS), "Occlusions changed")
    require(metadata["coordinate_transform"]["mapYrow"] == "increasing", "Unexpected raster orientation")
    require(not metadata["limitations"]["heights_known"] and arrays["object_valid"].all(),
            "Expected four valid roof proxies with unknown heights")
    require(np.array_equal(arrays["object_summary"], [x["summary"] for x in metadata["objects"]]),
            "Object summaries differ between metadata and raster")
    verification = {
        "semantic_metadata_equality": True,
        "fixed_compiler_array_equality": {k: True for k in sorted(arrays)},
        "v11_recorded_npz_sha256": metadata["artifacts"]["map.npz"]["sha256"],
        "local_npz_sha256": sources["raster_archive"]["sha256"],
        "local_npz_matches_v11_recorded_hash": sources["raster_archive"]["sha256"] == metadata["artifacts"]["map.npz"]["sha256"],
        "raster_selection": "Existing local archive verified against its own manifest and fixed compiler; v11 inputs/map.npz is not used.",
    }
    return metadata, arrays, sources, verification


def world_xz(normalized_xy):
    return [[float(x) * SPAN, -float(y) * SPAN] for x, y in normalized_xy]


def bounds(polygon):
    points = np.asarray(polygon)
    return {"minX": float(points[:, 0].min()), "maxX": float(points[:, 0].max()),
            "minZ": float(points[:, 1].min()), "maxZ": float(points[:, 1].max())}


def centroid(polygon):
    points = np.asarray(polygon, dtype=np.float64)
    following = np.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    require(abs(cross.sum()) > 1e-10, "Degenerate source polygon")
    return (((points + following) * cross[:, None]).sum(0) / (3 * cross.sum())).tolist()


def contains(polygon, point):
    x, z = point
    inside = False
    for (ax, az), (bx, bz) in zip(polygon, polygon[1:] + polygon[:1]):
        if (az > z) != (bz > z) and x < ax + (z - az) * (bx - ax) / (bz - az):
            inside = not inside
    return inside


def chunk_id(x, z):
    require(-SPAN / 2 <= x <= SPAN / 2 and -SPAN / 2 <= z <= SPAN / 2, "Point outside bounded map")
    ix, iz = [min(CHUNKS_PER_AXIS - 1, math.floor((v + SPAN / 2) / (SPAN / CHUNKS_PER_AXIS))) for v in (x, z)]
    return f"map10:{ix}:{iz}"


def intersects(a, b):
    return a["minX"] <= b["maxX"] and a["maxX"] >= b["minX"] and a["minZ"] <= b["maxZ"] and a["maxZ"] >= b["minZ"]


def make_feature(item, kind, index, transform):
    source_pixels = np.asarray(item["outer_source_px"], dtype=np.float64)
    normalized = np.c_[source_pixels, np.ones(len(source_pixels))] @ transform.T
    require(np.allclose(normalized[:, :2], item["outer_normalized_xy"], rtol=0, atol=1e-12),
            f"Source pixel/normalized coordinate mismatch: {item['id']}")
    polygon = world_xz(item["outer_normalized_xy"])
    feature = {
        "id": item["id"], "label": item.get("label", item["id"]), "polygon": polygon,
        "holes": [], "center": centroid(polygon), "bounds": bounds(polygon),
        "source": {"kind": "existing_manual_roof_proxy" if kind == "buildings" else "existing_manual_road_patch",
                   "metadata": relative(METADATA), "record_index": index,
                   "record_collection": "objects" if kind == "buildings" else "roads"},
        "geometry_label": LABEL, "ai_geometry": False,
        "source_polygon_px": item["outer_source_px"], "source_polygon_normalized_xy": item["outer_normalized_xy"],
        "visible_interior_confidence": item["visible_interior_confidence"], "uncertainty": item["uncertainty"],
    }
    require(not item["holes_source_px"], "Unexpected source holes: export must preserve them explicitly")
    if kind == "buildings":
        require(item["height_known"] is False, "Observed height unexpectedly present")
        feature.update(height=HEIGHT, height_known=False, observed_height=None,
                       height_source="uniform visualization placeholder; not measured or inferred",
                       base_y=0.0, building_identity_verified=item["building_identity_verified"])
    else:
        feature.update(base_y=0.0, connected_network_known=False)
    feature["chunk_id"] = chunk_id(*feature["center"])
    return feature


def build_scaffold():
    metadata, arrays, sources, verification = load_inputs()
    source_to_normalized = np.asarray(metadata["coordinate_transform"]["source_pixel_to_normalized_xy_homogeneous"])
    normalized_to_world = np.diag([SPAN, -SPAN, 1.0])
    source_to_world = normalized_to_world @ source_to_normalized
    buildings = [make_feature(item, "buildings", i, source_to_normalized) for i, item in enumerate(metadata["objects"])]
    roads = [make_feature(item, "roads", i, source_to_normalized) for i, item in enumerate(metadata["roads"])]
    chunks = []
    size = SPAN / CHUNKS_PER_AXIS
    for iz in range(CHUNKS_PER_AXIS):
        for ix in range(CHUNKS_PER_AXIS):
            x, z = -SPAN / 2 + ix * size, -SPAN / 2 + iz * size
            chunk = {"id": f"map10:{ix}:{iz}", "index": [ix, iz],
                     "bounds": {"minX": x, "maxX": x + size, "minZ": z, "maxZ": z + size},
                     "building_ids": [], "road_ids": [], "road_cell_ids": []}
            chunks.append(chunk)
    for features, key in ((buildings, "building_ids"), (roads, "road_ids")):
        for feature in features:
            feature["chunk_ids"] = [chunk["id"] for chunk in chunks if intersects(feature["bounds"], chunk["bounds"])]
            for chunk in chunks:
                if chunk["id"] in feature["chunk_ids"]:
                    chunk[key].append(feature["id"])
    shape = arrays["roads"].shape[1:]
    require(list(shape) == metadata["coordinate_transform"]["raster_shape_hw"] == [64, 64], "Unexpected grid size")
    cell_size = SPAN / shape[0]
    cells, candidates = [], []
    chunk_lookup = {chunk["id"]: chunk for chunk in chunks}
    for row, col in np.argwhere(arrays["roads"][0] > 0):
        x, z = ((float(col) + .5) / shape[1] - .5) * SPAN, (.5 - (float(row) + .5) / shape[0]) * SPAN
        cell = {"id": f"map10:road_cell:{row}:{col}", "row": int(row), "col": int(col),
                "center": [x, z], "coverage": float(arrays["roads"][0, row, col]),
                "known": float(arrays["roads_known"][0, row, col]), "chunk_id": chunk_id(x, z)}
        cells.append(cell)
        chunk_lookup[cell["chunk_id"]]["road_cell_ids"].append(cell["id"])
        if (cell["known"] > 0 and not arrays["footprints"][:, row, col].any()
                and any(contains(road["polygon"], [x, z]) for road in roads)
                and not any(contains(b["polygon"], [x, z]) for b in buildings)):
            candidates.append(cell)
    require(candidates, "No source-supported road cell available for display spawn")
    selected = min(candidates, key=lambda cell: (sum(v * v for v in cell["center"]), cell["row"], cell["col"]))
    x, z = selected["center"]
    eye_y = SPAN * .017
    target = min(buildings, key=lambda b: sum((a - c) ** 2 for a, c in zip(b["center"], [x, z])))["center"]
    occlusions = []
    for item in metadata["zero_confidence_occlusions"]:
        points = np.asarray(item["outer_source_px"])
        xy = (np.c_[points, np.ones(len(points))] @ source_to_normalized.T)[:, :2]
        occlusions.append({"id": item["id"], "polygon": world_xz(xy), "known": 0.0, "reason": item["reason"]})
    scaffold = {
        "schema": "trellis2_map10_scaffold_v22", "map_id": "map10", "label": LABEL,
        "ai_geometry": False, "training": False, "native_asset_registration": None,
        "native_asset_relationship": "Independent map visualization layer; no registration to or replacement of the v21 native asset is claimed.",
        "coordinate_system": {
            "up_axis": "Y", "polygon_axes": ["x", "z"], "units": "normalized display units; not meters",
            "georeferenced": False, "metric": False, "world_span": SPAN,
            "world_xz_from_source_normalized_xy": "x=100*X; z=-100*Y; y=display height",
            "axis_meaning": "+x is source image right; +z is source image down; neither is a geographic direction",
            "ground_y": 0.0, "ground_semantics": "display plane only; no terrain elevation inferred",
            "normalized_xy_to_world_xz_homogeneous": normalized_to_world.tolist(),
            "world_xz_to_normalized_xy_homogeneous": np.linalg.inv(normalized_to_world).tolist(),
            "source_pixels_to_world_xz_homogeneous": source_to_world.tolist(),
            "world_xz_to_source_pixels_homogeneous": np.linalg.inv(source_to_world).tolist(),
            "crop_xyxy_source_px": metadata["crop_xyxy_source_px"],
            "source_size_wh": metadata["source_size_wh"], "source_coordinate_transform": metadata["coordinate_transform"],
        },
        "bounds": {"minX": -SPAN / 2, "maxX": SPAN / 2, "minZ": -SPAN / 2, "maxZ": SPAN / 2},
        "height_policy": {"known": False, "placeholder_height": HEIGHT, "units": "normalized display units",
                          "meaning": "All four roofs use the same arbitrary extrusion height for visualization only."},
        "buildings": buildings, "roads": roads, "zero_confidence_occlusions": occlusions,
        "road_grid": {
            "shape": list(shape), "encoding": "sparse cells; original float32 values preserved as JSON numbers",
            "cell_size": cell_size, "origin_corner_xz": [-SPAN / 2, SPAN / 2],
            "column_step_xz": [cell_size, 0.0], "row_step_xz": [0.0, -cell_size],
            "center_formula": "origin + (col+0.5)*column_step + (row+0.5)*row_step",
            "default_coverage": 0.0, "default_known": 0.0, "default_semantics": "unknown; never confirmed empty or traversable",
            "coverage_field": "roads[0,row,col]", "known_field": "roads_known[0,row,col]",
            "source_sha256": sources["raster_archive"]["sha256"], "cells": cells,
        },
        "spawn": {"position": [x, eye_y, z], "look_at": [target[0], eye_y, target[1]],
                  "chunk_id": selected["chunk_id"], "road_cell_id": selected["id"],
                  "selection": "Closest-to-map-center positive-known road cell inside a traced road polygon and outside every roof proxy",
                  "camera_height_semantics": "arbitrary display camera offset; not measured human height",
                  "navigability_verified": False},
        "chunking": {"shape": [CHUNKS_PER_AXIS, CHUNKS_PER_AXIS], "size": size,
                     "id_format": "map10:{x_index}:{z_index}", "coordinate_space": "global world xz; no per-chunk transform",
                     "membership": "Conservative polygon AABB overlap; chunk_id owns centroid, chunk_ids list all overlap chunks. Deduplicate geometry by persistent feature id.",
                     "point_membership": "Half-open cells, positive outer edge included in last chunk",
                     "extent": "Finite existing crop only; empty chunks remain unknown"},
        "chunks": chunks, "source_hashes": sources, "source_verification": verification,
        "reference": {"path": "map10_reference.png", "sha256": sources["reference_crop"]["sha256"],
                      "meaning": "Byte-identical existing source_crop.png; no new interpretation"},
        "limitations": {
            **{k: v for k, v in metadata["limitations"].items() if k != "statement"},
            "statement": "Four existing visible roof proxies and four disconnected road patches only. Placeholder extrusions are map-derived visualization, not AI geometry or reconstructed buildings.",
            "upstream_statement": metadata["limitations"]["statement"],
            "unknown_context": metadata["mask_semantics"], "confidence_semantics": metadata["confidence_semantics"],
            "roads_complete": False, "terrain_known": False, "scaffold_native_alignment_known": False,
        },
    }
    validate_scaffold(scaffold, arrays)
    return scaffold


def validate_scaffold(scaffold, arrays):
    """Consistency checks on the actual exported contract, without model work."""
    require(len(scaffold["buildings"]) == len(scaffold["roads"]) == 4, "Unexpected feature count")
    features = scaffold["buildings"] + scaffold["roads"]
    require(len({f["id"] for f in features}) == len(features), "Duplicate persistent IDs")
    inverse = np.asarray(scaffold["coordinate_system"]["world_xz_to_source_pixels_homogeneous"])
    chunk_lookup = {chunk["id"]: chunk for chunk in scaffold["chunks"]}
    for feature in features:
        p = np.asarray(feature["polygon"])
        require(np.isfinite(p).all() and (abs(p) <= SPAN / 2).all(), "Polygon outside bounded crop")
        restored = np.c_[p, np.ones(len(p))] @ inverse.T
        require(np.allclose(restored[:, :2], feature["source_polygon_px"], rtol=0, atol=1e-9), "Pixel/world transform round-trip failed")
        require(feature["chunk_id"] in feature["chunk_ids"], "Primary chunk absent from overlap list")
        key = "building_ids" if "height" in feature else "road_ids"
        require(all(feature["id"] in chunk_lookup[c][key] for c in feature["chunk_ids"]), "Chunk index mismatch")
    for building in scaffold["buildings"]:
        require(building["height_known"] is False and building["observed_height"] is None and building["height"] == HEIGHT,
                "Unknown height was represented as observed geometry")
    coverage, known = np.zeros_like(arrays["roads"]), np.zeros_like(arrays["roads_known"])
    for cell in scaffold["road_grid"]["cells"]:
        coverage[0, cell["row"], cell["col"]] = cell["coverage"]
        known[0, cell["row"], cell["col"]] = cell["known"]
        require(cell["chunk_id"] == chunk_id(*cell["center"]), "Road cell chunk mismatch")
        require(cell["id"] in chunk_lookup[cell["chunk_id"]]["road_cell_ids"], "Road cell missing from chunk index")
    require(np.array_equal(coverage, arrays["roads"]) and np.array_equal(known, arrays["roads_known"]),
            "Sparse JSON grid does not preserve the existing raster")
    position = scaffold["spawn"]["position"]
    require(scaffold["spawn"]["chunk_id"] == chunk_id(position[0], position[2]), "Spawn chunk mismatch")
    require(any(contains(r["polygon"], [position[0], position[2]]) for r in scaffold["roads"]), "Spawn outside road proxies")
    require(not any(contains(b["polygon"], [position[0], position[2]]) for b in scaffold["buildings"]), "Spawn inside roof proxy")
    require(json.loads(encode(scaffold)) == scaffold, "JSON round-trip mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Rebuild in memory and verify all existing output bytes and source hashes")
    args = parser.parse_args()
    scaffold = build_scaffold()
    scaffold_bytes = encode(scaffold)
    provenance = {
        "schema": "trellis2_map10_scaffold_provenance_v22", "label": LABEL,
        "source_path_base": "experiments/recursive_map_research_v1", "sources": scaffold["source_hashes"],
        "verification": scaffold["source_verification"], "transforms": scaffold["coordinate_system"],
        "limitations": scaffold["limitations"], "height_policy": scaffold["height_policy"],
        "scaffold": {"path": "map10_scaffold.json", "sha256": hashlib.sha256(scaffold_bytes).hexdigest()},
        "reference": scaffold["reference"],
        "checks": ["All declared source hashes verified", "Eight arrays exactly reproduce with existing fixed CPU compiler",
                   "Four persistent roof IDs and all four road polygons preserved", "Source/world transforms round-trip",
                   "Sparse road coverage/known masks round-trip exactly", "Chunk indices and source-supported spawn checked",
                   "JSON round-trip is lossless; heights and untraced context remain unknown"],
        "scope": {"cpu_only": True, "training": False, "gpu": False, "remote_jobs": False,
                  "photo_reinterpretation": False, "viewer_or_server_modified": False,
                  "native_asset_modified": False, "ai_geometry": False},
    }
    payloads = {"map10_scaffold.json": scaffold_bytes,
                "MAP10_PROVENANCE.json": encode(provenance), "map10_reference.png": REFERENCE.read_bytes()}
    if args.check:
        for name, payload in payloads.items():
            require((OUTPUT / name).read_bytes() == payload, f"Output differs from deterministic compilation: {name}")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            target = OUTPUT / name
            if not target.exists() or target.read_bytes() != payload:
                target.write_bytes(payload)
    print(json.dumps({"status": "verified" if args.check else "compiled", "output": str(OUTPUT),
                      "buildings": len(scaffold["buildings"]), "roads": len(scaffold["roads"]),
                      "road_cells": len(scaffold["road_grid"]["cells"]), "chunks": len(scaffold["chunks"]),
                      "scaffold_sha256": provenance["scaffold"]["sha256"], "files": list(payloads)}))


if __name__ == "__main__":
    main()
