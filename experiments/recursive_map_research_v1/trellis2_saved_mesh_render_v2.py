"""Render a saved TRELLIS.2 GLB, without repairing or exporting its meshes.

Launch with ordinary Python (starts a fresh Blender process)::

    python trellis2_saved_mesh_render_v2.py --asset world.glb --output views \
        --blender /path/to/blender-4.5.9/blender

Alternatively use ``blender -b --factory-startup --python SCRIPT -- ...``.
``--bounds`` / ``--sharedbounds`` accepts a JSON file or inline JSON containing
min/max vectors, two corner vectors, or center/radius. Reuse the SAME bounds for
both seeds; use their union AABB when neither individual AABB contains both.

Outputs: textured/*.png, neutral/*.png, contact.jpg, bounds.json,
frames/{textured,neutral}/frame_000.png ... frame_047.png, render_manifest.json,
and lateral-{textured,neutral}.mp4 when ffmpeg with libx264 is available.
Video encoding is optional; the manifest also records the encoding commands.
Only cameras, lighting, and a temporary neutral material override are added.
No asset transforms, normals, topology, modifiers, or mesh materials are edited.
"""

import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


SIZE = 768
SAMPLES = 16
FPS = 24
FRAME_COUNT = 48
FOV = math.radians(45.0)
MODES = ("textured", "neutral")
VIEW_NAMES = (
    "front", "back", "left", "right", "top", "oblique", "closeup",
    "lateral-left", "lateral-right",
)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset", required=True, help="The saved .glb to import")
    parser.add_argument("--output", required=True, help="Directory for rendered views")
    parser.add_argument("--bounds", "--sharedbounds", dest="bounds",
                        help='JSON file or inline JSON, e.g. {"min":[-1,-1,-1],"max":[1,1,1]}')
    parser.add_argument("--blender", help="Blender 4.5.9 executable; alternatively BLENDER_BIN")
    parser.add_argument("--ffmpeg", help="Optional ffmpeg executable; alternatively FFMPEG_BIN")
    parser.add_argument("--skip-video", action="store_true", help="Leave MP4 encoding to the caller")
    parser.add_argument("--_blender-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.asset = Path(args.asset).expanduser().resolve()
    args.output = Path(args.output).expanduser().resolve()
    if not args.asset.is_file() or args.asset.suffix.lower() != ".glb":
        parser.error("--asset must name an existing saved .glb file")
    return args


def executable(candidate):
    if not candidate:
        return None
    path = Path(candidate).expanduser()
    if path.is_file():
        return str(path.resolve())
    return shutil.which(str(candidate))


def find_blender(requested):
    if requested:
        found = executable(requested)
        if not found:
            raise RuntimeError("--blender executable does not exist: " + requested)
        return found
    for candidate in (os.environ.get("BLENDER_BIN"), os.environ.get("BLENDER_PATH"), "blender"):
        found = executable(candidate)
        if found:
            return found
    if os.name == "nt":
        roots = [Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Blender Foundation"]
        if os.environ.get("LOCALAPPDATA"):
            roots.append(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Blender Foundation")
        for root in roots:
            for pattern in ("*4.5.9*/blender.exe", "*4.5*/blender.exe"):
                for path in sorted(root.glob(pattern)):
                    if path.is_file():
                        return str(path.resolve())
    raise RuntimeError("Blender 4.5.9 was not found; pass --blender or set BLENDER_BIN")


def vector3(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(label + " must contain three finite numbers")
    result = [float(component) for component in value]
    if not all(math.isfinite(component) for component in result):
        raise ValueError(label + " must contain three finite numbers")
    return result


def camera_bounds(spec):
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        spec = {"min": spec[0], "max": spec[1]}
    if not isinstance(spec, dict):
        raise ValueError("Bounds must be min/max, two corners, or center/radius JSON")
    if "camera_bounds" in spec:
        return camera_bounds(spec["camera_bounds"])
    if "bounds" in spec:
        return camera_bounds(spec["bounds"])
    result = {}
    if "min" in spec or "max" in spec:
        low, high = vector3(spec["min"], "min"), vector3(spec["max"], "max")
        if any(low[i] > high[i] for i in range(3)):
            raise ValueError("Every min coordinate must be <= the corresponding max")
        result.update({"min": low, "max": high})
    if "center" in spec or "radius" in spec:
        center = vector3(spec["center"], "center")
        radius = float(spec["radius"])
    elif result:
        center = [(low[i] + high[i]) * 0.5 for i in range(3)]
        radius = math.sqrt(sum((high[i] - low[i]) ** 2 for i in range(3))) * 0.5
    else:
        raise ValueError("Bounds need min/max or center/radius")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Bounds radius must be finite and positive")
    result.update({"center": center, "radius": radius})
    return result


def read_bounds(argument, measured):
    if not argument:
        return camera_bounds(measured)
    value = argument.strip()
    if value.startswith(("{", "[")):
        return camera_bounds(json.loads(value))
    return camera_bounds(json.loads(Path(value).expanduser().read_text(encoding="utf-8-sig")))


def mesh_bounds(meshes):
    # Read real vertices in world coordinates, including nested node transforms.
    low, high = [math.inf] * 3, [-math.inf] * 3
    vertex_count = 0
    for obj in meshes:
        transform = obj.matrix_world.copy()
        for vertex in obj.data.vertices:
            point = transform @ vertex.co
            if not all(math.isfinite(float(component)) for component in point):
                raise ValueError("Asset contains a non-finite world-space vertex")
            for axis in range(3):
                low[axis] = min(low[axis], float(point[axis]))
                high[axis] = max(high[axis], float(point[axis]))
            vertex_count += 1
    if vertex_count == 0:
        raise ValueError("The GLB contains no mesh vertices")
    return {"min": low, "max": high}, vertex_count


def asset_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_poses(center, radius, Vector):
    distance = radius / math.sin(FOV * 0.5) * 1.15
    origin = Vector(center)

    def at(direction, scale=1.0):
        eye = origin + Vector(direction).normalized() * distance * scale
        return {"eye": list(eye), "look": list(origin)}

    poses = {
        "front": at((0, -1, 0)), "back": at((0, 1, 0)),
        "left": at((-1, 0, 0)), "right": at((1, 0, 0)),
        "top": at((0, 0, 1)), "oblique": at((1, -1, 0.75)),
        "closeup": at((1, -1, 0.75), 0.62),
    }
    base_eye = origin + Vector((0, -1, 0.35)).normalized() * distance
    motion = []
    for index in range(FRAME_COUNT):
        shift = (-0.6 + 1.2 * index / (FRAME_COUNT - 1)) * radius
        eye = base_eye + Vector((shift, 0, 0))
        motion.append({"frame": index, "time_seconds": index / FPS,
                       "eye": list(eye), "look": list(origin)})
    poses["lateral-left"] = {key: motion[0][key] for key in ("eye", "look")}
    poses["lateral-right"] = {key: motion[-1][key] for key in ("eye", "look")}
    return poses, motion, distance


def set_pose(camera, pose, Vector):
    camera.location = Vector(pose["eye"])
    camera.rotation_euler = (Vector(pose["look"]) - camera.location).to_track_quat("-Z", "Y").to_euler()


# Tiny embedded font keeps contact-sheet creation independent of Pillow/ffmpeg.
FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "N": ("10001", "11001", "11001", "10101", "10011", "10011", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    " ": ("00000",) * 7,
}


def contact_sheet(bpy, output):
    tile, footer, columns, rows = 384, 32, 3, 6
    width, height = columns * tile, rows * (tile + footer)
    pixels = array("f", (0.12, 0.12, 0.12, 1.0)) * (width * height)
    ink = array("f", (0.92, 0.92, 0.92, 1.0))
    # Alternate matching textured/neutral rows so each pair shares a column.
    for group in range(3):
        for mode_index, mode in enumerate(MODES):
            row = group * 2 + mode_index
            bottom = height - (row + 1) * (tile + footer)
            for column, name in enumerate(VIEW_NAMES[group * 3:group * 3 + 3]):
                source = bpy.data.images.load(str(output / mode / (name + ".png")), check_existing=False)
                try:
                    # Copy encoded PNG colors unchanged; never apply a second view transform.
                    source.colorspace_settings.name = "Non-Color"
                    source.scale(tile, tile)
                    source_pixels = array("f", [0.0]) * (tile * tile * 4)
                    source.pixels.foreach_get(source_pixels)
                    for y in range(tile):
                        start = ((bottom + footer + y) * width + column * tile) * 4
                        pixels[start:start + tile * 4] = source_pixels[y * tile * 4:(y + 1) * tile * 4]
                finally:
                    bpy.data.images.remove(source)
                for char_index, char in enumerate((mode + " " + name).upper()):
                    for glyph_y, line in enumerate(FONT[char]):
                        for glyph_x, bit in enumerate(line):
                            if bit == "1":
                                for dy in range(2):
                                    for dx in range(2):
                                        x = column * tile + 10 + char_index * 12 + glyph_x * 2 + dx
                                        y = bottom + 9 + (6 - glyph_y) * 2 + dy
                                        start = (y * width + x) * 4
                                        pixels[start:start + 4] = ink
    sheet = bpy.data.images.new("Saved mesh contact", width=width, height=height, alpha=False)
    try:
        sheet.colorspace_settings.name = "Non-Color"
        sheet.pixels.foreach_set(pixels)
        sheet.filepath_raw = str(output / "contact.jpg")
        sheet.file_format = "JPEG"
        sheet.save()
    finally:
        bpy.data.images.remove(sheet)


def encode_videos(args):
    ffmpeg = executable(args.ffmpeg or os.environ.get("FFMPEG_BIN") or "ffmpeg")
    records = {}
    for mode in MODES:
        command = [ffmpeg or "ffmpeg", "-nostdin", "-y", "-framerate", str(FPS),
                   "-start_number", "0", "-i", str(args.output / "frames" / mode / "frame_%03d.png"),
                   "-frames:v", str(FRAME_COUNT), "-c:v", "libx264", "-preset", "medium",
                   "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an",
                   "-movflags", "+faststart", str(args.output / ("lateral-" + mode + ".mp4"))]
        status = "deferred"
        if ffmpeg and not args.skip_video:
            try:
                result = subprocess.run(command, check=False)
                status = "complete" if result.returncode == 0 else "failed"
            except OSError as error:
                status = "failed"
                print("VIDEO_ENCODING_WARNING:", error, flush=True)
        records[mode] = {"status": status, "command": command,
                         "duration_seconds": FRAME_COUNT / FPS}
        if status != "complete":
            print("VIDEO_ENCODING_" + status.upper(), json.dumps(command), flush=True)
    return records


def render(args):
    import bpy
    from mathutils import Vector

    if tuple(bpy.app.version[:3]) != (4, 5, 9):
        raise RuntimeError("This fixed renderer requires Blender 4.5.9; found " + bpy.app.version_string)
    source_hash = asset_digest(args.asset)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.asset))
    scene = bpy.context.scene
    scene.frame_set(1)  # Keep imported animation at its saved starting frame for every render.
    bpy.context.view_layer.update()
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    measured, vertices = mesh_bounds(meshes)
    bounds = read_bounds(args.bounds, measured)
    center, radius = Vector(bounds["center"]), bounds["radius"]
    poses, motion, distance = camera_poses(center, radius, Vector)
    for obj in scene.objects:
        if obj.type == "LIGHT":
            obj.hide_render = True  # Imported light rigs must not change comparison lighting.

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.seed = 0
    scene.cycles.use_animated_seed = False
    scene.cycles.max_bounces = 8
    scene.render.resolution_x = scene.render.resolution_y = SIZE
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.use_file_extension = True
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.frame_start, scene.frame_end = 1, FRAME_COUNT
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    world = bpy.data.worlds.new("Fixed neutral world")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.18, 0.18, 0.18, 1)
    background.inputs["Strength"].default_value = 0.55

    light_records = []
    for name, offset, power, size in (
        ("Key", (-3, -4, 5), 450, 3),
        ("Fill", (4, -1, 3), 180, 4),
        ("Rim", (1, 4, 5), 260, 3),
    ):
        data = bpy.data.lights.new("Fixed " + name, type="AREA")
        data.shape, data.energy, data.size = "DISK", power * radius ** 2, size * radius
        light = bpy.data.objects.new("Fixed " + name, data)
        scene.collection.objects.link(light)
        light.location = center + Vector(offset) * radius
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
        light_records.append({"name": name, "location": list(light.location),
                              "energy": data.energy, "size": data.size})

    camera_data = bpy.data.cameras.new("Fixed comparison camera")
    camera = bpy.data.objects.new("Fixed comparison camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36
    camera_data.lens = 36 / (2 * math.tan(FOV * 0.5))
    camera_data.clip_start = max(radius * 0.0001, 0.000001)
    camera_data.clip_end = max(distance + radius * 20, 1)
    neutral = bpy.data.materials.new("Neutral geometry override")
    neutral.use_nodes = True
    shader = neutral.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1)
    shader.inputs["Metallic"].default_value = 0
    shader.inputs["Roughness"].default_value = 0.8
    view_layer = bpy.context.view_layer

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "bounds.json").write_text(json.dumps(bounds, indent=2) + "\n", encoding="utf-8")
    still_records = []
    try:
        for mode in MODES:
            view_layer.material_override = neutral if mode == "neutral" else None
            still_dir = args.output / mode
            frame_dir = args.output / "frames" / mode
            still_dir.mkdir(parents=True, exist_ok=True)
            frame_dir.mkdir(parents=True, exist_ok=True)
            for name in VIEW_NAMES:
                set_pose(camera, poses[name], Vector)
                scene.render.filepath = str(still_dir / (name + ".png"))
                print("RENDER_STILL", mode, name, flush=True)
                bpy.ops.render.render(write_still=True)
                still_records.append({"mode": mode, "view": name,
                                      "file": mode + "/" + name + ".png", **poses[name]})
            for pose in motion:
                frame = pose["frame"]
                frame_path = frame_dir / ("frame_%03d.png" % frame)
                if frame in (0, FRAME_COUNT - 1):
                    # Exact lateral endpoint renders already exist with the same pose.
                    name = "lateral-left" if frame == 0 else "lateral-right"
                    shutil.copyfile(still_dir / (name + ".png"), frame_path)
                else:
                    set_pose(camera, pose, Vector)
                    scene.render.filepath = str(frame_path)
                    print("RENDER_LATERAL", mode, frame, flush=True)
                    bpy.ops.render.render(write_still=True)
    finally:
        view_layer.material_override = None
    contact_sheet(bpy, args.output)
    video_records = encode_videos(args)
    manifest = {
        "asset": str(args.asset), "asset_sha256": source_hash,
        "blender_version": bpy.app.version_string,
        "geometry_modified": False, "asset_frame": 1,
        "coordinate_system": "Blender world coordinates; front=-Y, left=-X, up=+Z",
        "asset_bounds": measured, "camera_bounds": bounds,
        "bounds_source": args.bounds or "asset AABB",
        "mesh_objects": len(meshes), "vertices_in_instances": vertices,
        "polygons_in_instances": sum(len(obj.data.polygons) for obj in meshes),
        "engine": "Cycles", "device": "CPU", "samples": SAMPLES,
        "resolution": [SIZE, SIZE], "fov_degrees": math.degrees(FOV),
        "camera_distance": distance, "lights": light_records,
        "world": {"color": [0.18, 0.18, 0.18], "strength": 0.55},
        "neutral": "View-layer material override only; original mesh data is retained",
        "stills": still_records,
        "contact": {"file": "contact.jpg", "columns": 3, "rows": 6,
                    "row_modes": ["textured", "neutral"] * 3},
        "motion": {"fps": FPS, "frame_count": FRAME_COUNT,
                   "duration_seconds": FRAME_COUNT / FPS, "poses": motion,
                   "frame_pattern": "frames/{mode}/frame_%03d.png"},
        "videos": video_records,
    }
    (args.output / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("SAVED_MESH_VIEWS_READY", str(args.output), flush=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parse_args(argv)
    if args._blender_worker or "_bpy" in sys.modules:
        render(args)
        return 0
    command = [find_blender(args.blender), "--background", "--factory-startup",
               "--disable-autoexec", "--python-exit-code", "1",
               "--python", str(Path(__file__).resolve()), "--", *argv, "--_blender-worker"]
    print("START_BLENDER_PROCESS", json.dumps(command), flush=True)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
