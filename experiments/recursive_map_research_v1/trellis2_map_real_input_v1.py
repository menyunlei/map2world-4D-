"""CPU-only, fixed source-image roof/road proxy input sidecar.

No models, targets, training, SSH, or generated scenes are used. The input
is an oblique screenshot, not surveyed footprints or georeferenced data.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


SOURCE_SHA256 = "066bb11bfc99ebccae10505f235167cb0f9c9dcca06d24343d1328a29842cc7b"
SOURCE_SIZE = (1080, 2280)
CROP_XYXY = (72, 910, 982, 1820)
OUTPUT_RELATIVE = Path("map_object_world_v1/trellis2_map_learning_v1/trial_fix1/real_input")
MAP_SIZE = 64
FIELDS = ("footprints", "footprint_known", "roads", "roads_known",
          "open_space", "open_space_known", "object_summary", "object_valid")
RASTER_FIELDS = FIELDS[:6]

# Fixed manual traces in the original 1080x2280 source frame, not crop pixels.
# Object grouping describes visible roof structures, not verified properties.
# No holes are asserted: skylights/UI marks are not evidence of footprint holes.
OBJECTS = (
    {"id": "roof_proxy_01", "label": "gray roof with three skylights",
     "outer_source_px": [[229, 1031], [280, 1015], [319, 1117], [268, 1136]],
     "holes_source_px": [], "visible_interior_confidence": .65,
     "uncertainty": "Roof edge and neighboring attachment/property boundary are uncertain; skylights retained as roof support."},
    {"id": "roof_proxy_02", "label": "lower gray gable roof",
     "outer_source_px": [[463, 1112], [541, 1087], [581, 1156], [487, 1185]],
     "holes_source_px": [], "visible_interior_confidence": .55,
     "uncertainty": "Attached terrace identity/extent is unverified; yellow UI cross occludes the interior and has zero confidence."},
    {"id": "roof_proxy_03", "label": "small domed roof structure",
     "outer_source_px": [[310, 1507], [362, 1487], [409, 1587], [358, 1617]],
     "holes_source_px": [], "visible_interior_confidence": .55,
     "uncertainty": "Domed roof and surrounding flat roof are treated as one visible roof structure; tree-fringe/boundary uncertainty remains."},
    {"id": "roof_proxy_04", "label": "large red hall roof",
     "outer_source_px": [[454, 1319], [606, 1260], [754, 1572], [585, 1635]],
     "holes_source_px": [], "visible_interior_confidence": .70,
     "uncertainty": "Primary visible gable roof only; attached pale/green roof structures are unannotated. Roof overhang, perspective, tree fringe, and full building footprint are unresolved; dark skylight is not a verified hole."},
)
ROADS = (
    {"id": "road_visible_01", "outer_source_px": [[531, 930], [616, 915], [669, 1047], [597, 1078]],
     "holes_source_px": [], "visible_interior_confidence": .75,
     "uncertainty": "Visible upper asphalt lane patch, including shadow; not a complete road width or network."},
    {"id": "road_visible_02", "outer_source_px": [[714, 1228], [757, 1213], [788, 1273], [743, 1292]],
     "holes_source_px": [], "visible_interior_confidence": .80,
     "uncertainty": "Selected visible central asphalt patch between vehicles; adjacent parked/occluded areas are unknown."},
    {"id": "road_visible_03", "outer_source_px": [[792, 1448], [827, 1435], [862, 1512], [828, 1529]],
     "holes_source_px": [], "visible_interior_confidence": .80,
     "uncertainty": "Selected visible lane patch; not extrapolated through vehicle or tree occlusions."},
    {"id": "road_visible_04", "outer_source_px": [[873, 1608], [902, 1597], [937, 1676], [908, 1688]],
     "holes_source_px": [], "visible_interior_confidence": .75,
     "uncertainty": "Selected lower asphalt patch; curb/traffic occlusions and untraced continuation remain unknown."},
)
OCCLUSIONS = (
    {"id": "yellow_UI_cross_horizontal", "outer_source_px": [[499, 1133], [568, 1133], [568, 1145], [499, 1145]],
     "reason": "Screenshot UI obscures roof evidence; confidence is zero, not a roof hole."},
    {"id": "yellow_UI_cross_vertical", "outer_source_px": [[531, 1103], [546, 1103], [546, 1174], [531, 1174]],
     "reason": "Screenshot UI obscures roof evidence; confidence is zero, not a roof hole."},
)
COLORS = ((49, 190, 255), (235, 99, 245), (89, 225, 151), (255, 179, 63))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_crop(source):
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Source SHA256 differs from the fixed manually inspected screenshot")
    with Image.open(source) as image:
        if image.size != SOURCE_SIZE:
            raise ValueError("Source dimensions differ from the annotated source frame")
        return image.convert("RGB").crop(CROP_XYXY)


def normalized_xy(points):
    points = np.asarray(points, dtype=np.float64)
    left, top, right, bottom = CROP_XYXY
    return np.column_stack(((points[:, 0] - left) / (right - left) - .5,
                           .5 - (points[:, 1] - top) / (bottom - top)))


def polygon_mask(points):
    """Even-odd fill at source pixel centers; no scene/3D reconstruction."""
    left, top, right, bottom = CROP_XYXY
    x, y = np.meshgrid(np.arange(left, right) + .5, np.arange(top, bottom) + .5)
    inside = np.zeros(x.shape, dtype=bool)
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3 or ((points[:, 0] < left) | (points[:, 0] > right)
                           | (points[:, 1] < top) | (points[:, 1] > bottom)).any():
        raise ValueError("Every fixed polygon must stay within the source crop")
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        if start[1] == end[1]:
            continue
        crosses = (start[1] > y) != (end[1] > y)
        crossing_x = start[0] + (y - start[1]) * (end[0] - start[0]) / (end[1] - start[1])
        inside ^= crosses & (x < crossing_x)
    return inside


def eroded_interior(mask):
    # Three source pixels of uncertain boundary, never known exterior negatives.
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(
        ImageFilter.MinFilter(7))) > 0


def area_pool_native(mask):
    """Area-average source support, then flip source +Y row0 to native -Y row0."""
    image = Image.fromarray(np.asarray(mask, dtype=np.float32))
    pooled = np.asarray(image.resize((MAP_SIZE, MAP_SIZE), Image.Resampling.BOX), dtype=np.float32)
    return np.ascontiguousarray(np.clip(pooled[::-1], 0., 1.))


def polygon_summary(points, support, known):
    points = np.asarray(points, dtype=np.float64)
    following = np.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    twice_area = cross.sum()
    if abs(twice_area) < 1.:
        raise ValueError("A roof polygon must have nonzero image area")
    center = ((points + following) * cross[:, None]).sum(0) / (3. * twice_area)
    center = normalized_xy([center])[0]
    side = CROP_XYXY[2] - CROP_XYXY[0]
    extent = np.ptp(points, axis=0) / side
    area = abs(twice_area) / (2. * side * side)
    confidence = known.sum(dtype=np.float64) / support.sum(dtype=np.float64)
    return np.array([*center, *extent, area, confidence, 0., 1.], dtype=np.float32)


def build_arrays():
    side = CROP_XYXY[2] - CROP_XYXY[0]
    occluded = np.zeros((side, side), dtype=bool)
    for item in OCCLUSIONS:
        occluded |= polygon_mask(item["outer_source_px"])
    supports, confidences, summaries = [], [], []
    for item in OBJECTS:
        mask = polygon_mask(item["outer_source_px"])
        support = area_pool_native(mask)
        known = area_pool_native(eroded_interior(mask) & ~occluded) * item["visible_interior_confidence"]
        supports.append(support)
        confidences.append(known)
        summaries.append(polygon_summary(item["outer_source_px"], support, known))
    road = np.zeros((side, side), dtype=np.float32)
    road_known = np.zeros_like(road)
    for item in ROADS:
        mask = polygon_mask(item["outer_source_px"])
        road = np.maximum(road, mask)
        confidence = (eroded_interior(mask) & ~occluded) * item["visible_interior_confidence"]
        road_known = np.maximum(road_known, confidence)
    arrays = {
        "footprints": np.stack(supports), "footprint_known": np.stack(confidences),
        "roads": area_pool_native(road)[None], "roads_known": area_pool_native(road_known)[None],
        "open_space": np.zeros((1, MAP_SIZE, MAP_SIZE), dtype=np.float32),
        "open_space_known": np.zeros((1, MAP_SIZE, MAP_SIZE), dtype=np.float32),
        "object_summary": np.stack(summaries), "object_valid": np.ones(len(OBJECTS), dtype=bool),
    }
    validate_arrays(arrays)
    return arrays


def validate_arrays(arrays):
    if set(arrays) != set(FIELDS):
        raise ValueError("NPZ must contain exactly the eight conditioner input fields")
    objects = len(OBJECTS)
    for key in FIELDS:
        expected = ((objects, MAP_SIZE, MAP_SIZE) if key in FIELDS[:2]
                    else (1, MAP_SIZE, MAP_SIZE) if key in RASTER_FIELDS
                    else (objects, 8) if key == "object_summary" else (objects,))
        value = arrays[key]
        dtype = np.dtype(bool) if key == "object_valid" else np.dtype(np.float32)
        if value.shape != expected or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"Invalid shape/dtype/finite contract: {key}")
        if key in RASTER_FIELDS and ((value < 0) | (value > 1)).any():
            raise ValueError(f"Raster outside [0, 1]: {key}")
    for support, known in (("footprints", "footprint_known"), ("roads", "roads_known"),
                           ("open_space", "open_space_known")):
        if (arrays[known] > arrays[support] + 1e-6).any():
            raise ValueError("Known masks may mark only traced positive visible support")
    summary = arrays["object_summary"]
    if ((summary[:, :2] < -.5) | (summary[:, :2] > .5)).any():
        raise ValueError("Summary XY must be normalized image coordinates")
    if ((summary[:, 2:6] < 0) | (summary[:, 2:6] > 1)).any():
        raise ValueError("Summary extents/area/confidence outside [0, 1]")
    if (summary[:, 6] != 0).any() or (summary[:, 7] != 1).any():
        raise ValueError("No known heights; every slot has classBuilding=1")
    if arrays["open_space"].any() or arrays["open_space_known"].any():
        raise ValueError("No verified open-space annotation is asserted")


def annotation_overlay(crop):
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    left, top = CROP_XYXY[:2]
    try:
        font = ImageFont.load_default(size=17)
    except TypeError:  # Older Pillow on the server can still export the sidecar.
        font = ImageFont.load_default()
    for index, item in enumerate(OBJECTS):
        points = [(x - left, y - top) for x, y in item["outer_source_px"]]
        color = COLORS[index]
        draw.polygon(points, fill=(*color, 48))
        draw.line(points + points[:1], fill=(*color, 255), width=3)
        x, y = np.asarray(points).mean(0)
        label = f"B{index + 1}: roof proxy"
        draw.rectangle((int(x) - 8, int(y) - 11, int(x) + 140, int(y) + 12), fill=(0, 0, 0, 200))
        draw.text((int(x), int(y) - 8), label, fill=(*color, 255), font=font)
    for index, item in enumerate(ROADS):
        points = [(x - left, y - top) for x, y in item["outer_source_px"]]
        draw.polygon(points, fill=(255, 242, 97, 60), outline=(255, 242, 97, 240), width=2)
        x, y = np.asarray(points).mean(0)
        draw.text((int(x) - 8, int(y) - 8), f"R{index + 1}", fill=(255, 255, 180, 255), font=font,
                  stroke_width=1, stroke_fill=(0, 0, 0, 255))
    for item in OCCLUSIONS:
        points = [(x - left, y - top) for x, y in item["outer_source_px"]]
        draw.polygon(points, fill=(255, 40, 70, 85), outline=(255, 40, 70, 240), width=2)
    draw.rectangle((5, 5, 905, 51), fill=(0, 0, 0, 210))
    draw.text((12, 8), "SOURCE ONLY: 4 roof proxies / selected asphalt patches / red = UI unknown",
              fill=(255, 255, 255, 255), font=font)
    draw.text((12, 29), "Oblique screenshot: NO measured footprint, geographic accuracy, or height ground truth",
              fill=(255, 220, 155, 255), font=font)
    return Image.alpha_composite(crop.convert("RGBA"), overlay).convert("RGB")


def ensure_image(path, image):
    if path.exists():
        with Image.open(path) as existing:
            if not np.array_equal(np.asarray(existing.convert("RGB")), np.asarray(image)):
                raise ValueError(f"Existing {path.name} differs; refusing to overwrite")
    else:
        image.save(path)


def export_input(source, output, crop):
    archive_names = ("map.npz", "removed_last_building_map.npz", "shuffled_map.npz")
    if any((output / name).exists() for name in (*archive_names, "MAP_INPUT.json")):
        raise FileExistsError("Real-input archives/manifest already exist; refusing to replace a frozen input")
    arrays = build_arrays()
    removed = {key: value.copy() for key, value in arrays.items()}
    removed["footprints"][-1] = 0
    removed["footprint_known"][-1] = 0
    removed["object_summary"][-1] = 0
    removed["object_summary"][-1, 7] = 1
    removed["object_valid"][-1] = False
    columns = (np.arange(MAP_SIZE) + 16) % MAP_SIZE
    shuffled = {key: value.copy() for key, value in arrays.items()}
    for key in RASTER_FIELDS:
        shuffled[key] = np.ascontiguousarray(arrays[key][..., columns])
    for variant in (removed, shuffled):
        validate_arrays(variant)
    ensure_image(output / "annotation_overlay.png", annotation_overlay(crop))
    for name, variant in zip(archive_names, (arrays, removed, shuffled)):
        np.savez_compressed(output / name, **variant)
        with np.load(output / name, allow_pickle=False) as saved:
            loaded = {key: saved[key] for key in saved.files}
        validate_arrays(loaded)
        if any(not np.array_equal(loaded[key], variant[key]) for key in FIELDS):
            raise ValueError("Archive round-trip changed input values")
    left, top, right, bottom = CROP_XYXY
    side = right - left
    object_records = []
    for index, item in enumerate(OBJECTS):
        object_records.append({**item, "outer_normalized_xy": normalized_xy(item["outer_source_px"]).tolist(),
                               "summary": arrays["object_summary"][index].tolist(), "height_known": False,
                               "building_identity_verified": False, "complete_selected_roof_proxy_inside_crop": True})
    records = {
        "schema": "trellis2_map_real_input_v1", "source": str(source.resolve()),
        "server_source_default": "paths.JOB/map10_source.jpg",
        "user_original_source": "E:/D/map_build/map/10.jpg", "source_sha256": SOURCE_SHA256,
        "source_size_wh": list(SOURCE_SIZE), "source_kind": "oblique map-app screenshot",
        "crop_xyxy_source_px": list(CROP_XYXY), "crop_bounds": "left/top inclusive; right/bottom exclusive",
        "square_padding_ltrb_px": [0, 0, 0, 0], "crop_size_wh": [side, side],
        "annotation_basis": "Manual inspection of original 10.jpg and this single fixed source crop; polygons fixed before raster/control export; no generated 3D output or evaluation feedback used.",
        "annotation_script_sha256": sha256(Path(__file__)),
        "limitations": {
            "surveyed_footprints": False, "georeferenced": False, "measured_footprint_accuracy_supported": False,
            "geographic_accuracy_supported": False, "heights_known": False, "real_3d_ground_truth": False,
            "metric_scale_or_ground_plane_inferred": False, "building_property_identity_verified": False,
            "statement": "Visible roof/road proxies cannot substantiate measured-footprint or geographic accuracy. Perspective, roof overhangs, occlusions and partial context prevent those claims. No procedural 3D geometry is constructed.",
        },
        "coordinate_transform": {
            "axes": "+X right; +Y up in image; Z not annotated", "normalized_xy_range": [-.5, .5],
            "source_pixels_to_normalized_xy": "X=(u-left)/side-.5; Y=.5-(v-top)/side",
            "source_pixel_to_normalized_xy_homogeneous": [[1 / side, 0, -left / side - .5],
                                                         [0, -1 / side, top / side + .5], [0, 0, 1]],
            "normalized_xy_to_source_pixel_homogeneous": [[side, 0, left + side / 2],
                                                         [0, -side, top + side / 2], [0, 0, 1]],
            "raster_shape_hw": [MAP_SIZE, MAP_SIZE], "row0": "-Y; bottom of original source crop",
            "col0": "-X; left of original source crop", "mapYrow": "increasing",
            "source_crop_row0": "+Y; top of original source crop; flipped once during export",
            "native_dense_token_axis_order": "X then Y then Z; Z fastest; raster indexes row=Y, col=X",
            "raster_pixel_centers_to_source": "u=left+(col+.5)*side/64; v=top+side-(row+.5)*side/64",
            "raster_pixel_centers_to_normalized": "X=(col+.5)/64-.5; Y=(row+.5)/64-.5",
            "meaning": "Image affine coordinates only, not surveyed/geographic or metric world registration.",
        },
        "objects": object_records,
        "roads": [{**item, "outer_normalized_xy": normalized_xy(item["outer_source_px"]).tolist()} for item in ROADS],
        "zero_confidence_occlusions": list(OCCLUSIONS),
        "holes": "None visually verified; no holes asserted. UI occlusions are confidence masks, not holes.",
        "mask_semantics": "Only traced positive visible support can be known. Three source pixels of each boundary and explicit UI occlusions are unknown. All untraced context, negative/background absence, foliage, occluded pavement and open-space regions remain unknown; no all-known/empty artificial regions.",
        "confidence_semantics": "Conservative manual visibility scores, not calibrated probabilities or geographic/footprint measurement confidence.",
        "rasterization": "Even-odd fill at original pixel centers; 3-pixel interior erosion for known masks; BOX area pooling to 64, followed by vertical flip. Fractional support retained.",
        "object_summary_order": ["centroidX", "centroidY", "extentX", "extentY", "area", "knownXYconfidence", "heightKnown", "classBuilding"],
        "summary_derivation": "Centroid/XY extent/area from traced 2D roof polygon only; confidence=pooled_known_sum/pooled_support_sum; heightKnown=0; classBuilding=1. No boxes/heights from generated or real 3D targets.",
        "npz_fields": {key: {"shape": list(arrays[key].shape), "dtype": str(arrays[key].dtype)} for key in FIELDS},
        "batch_axis": "omitted; main adds B", "object_ids": [item["id"] for item in OBJECTS],
        "object_id_semantics": "External metadata only; not numeric learned embeddings.",
        "variants": {
            "map.npz": {"kind": "fixed manually traced source proxy conditions", "valid_objects": 4},
            "removed_last_building_map.npz": {"kind": "source-derived object removal control", "valid_objects": 3,
                "removed_id": OBJECTS[-1]["id"], "slot_policy": "Keep 4 slots; last support/known/summary zero except classBuilding=1; object_valid=False; all other conditions unchanged."},
            "shuffled_map.npz": {"kind": "deliberately corrupted raster control; not an authentic scene annotation",
                "column_permutation_new_col_to_old": columns.tolist(), "applies_to": list(RASTER_FIELDS),
                "summaries_validity_ids": "held fixed", "reason": "Spatial raster corruption changes locality without an equivariant aligned object reorder."},
        },
        "artifacts": {name: {"sha256": sha256(output / name)} for name in
                      ("source_crop.png", "annotation_overlay.png", *archive_names)},
        "local_checks": "Exact eight-field shapes/dtypes, finite/range masks, known<=positive support, no height claims/open-space knowledge, and NPZ round-trip equality checked on CPU; no native/model/data-quality validation claimed.",
    }
    with (output / "MAP_INPUT.json").open("x", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output.resolve()), "crop_xyxy": list(CROP_XYXY),
                      "source_sha256": SOURCE_SHA256, "objects": len(OBJECTS), "fields": list(FIELDS)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, help="Local JOB override; default is existing paths.JOB")
    parser.add_argument("--source", type=Path, help="Default: JOB/map10_source.jpg")
    parser.add_argument("--crop-only", action="store_true", help="Prepare source evidence before tracing")
    parser.add_argument("--overlay-only", action="store_true", help="Inspect source/polygon overlay before exporting inputs")
    args = parser.parse_args()
    if args.job is None:
        from paths import JOB
        job = JOB
    else:
        job = args.job
    source = args.source if args.source is not None else job / "map10_source.jpg"
    output = job / OUTPUT_RELATIVE
    crop = source_crop(source)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "source_crop.png"
    ensure_image(target, crop)
    if args.crop_only:
        print(target.resolve())
    elif args.overlay_only:
        ensure_image(output / "annotation_overlay.png", annotation_overlay(crop))
        print((output / "annotation_overlay.png").resolve())
    else:
        export_input(source, output, crop)


if __name__ == "__main__":
    main()
