"""Additive v24 road-boundary wrapper; v23 backend and frontend stay on disk unchanged."""
from pathlib import Path
from io import BytesIO
from html import escape
import json
from urllib.parse import urlsplit

JOB = Path(__file__).resolve().parent
V23_FRONTEND = JOB / 'map_object_world_v1/trellis2_online_v23'
CHUNK_POSITION = [0, 0, -100.0154161373455]
# Fixed v21 display affine, composed with native Z-up -> glTF Y-up axes.
# Row-major homogeneous matrix acting on native [x, y, z, 1].
PARENT_WORLD_AFFINE = [
    [100.0154161373455, 0, 0, 0],
    [0, 0, 100.0154161373455, 7.354263551565667],
    [0, -100.0154161373455, 0, 0],
    [0, 0, 0, 1],
]
# The existing parent GLB already contains this column-major root matrix.
PARENT_GLTF_ROOT_MATRIX = [
    100.0154161373455, 0, 0, 0,
    0, 100.0154161373455, 0, 0,
    0, 0, 100.0154161373455, 0,
    0, 7.354263551565667, 0, 1,
]
INHERITED_OBSERVED_ROAD_CONDITIONS = {
    'label': 'Inherited observed-road conditions from the generated v21 parent',
    'source': 'map_object_world_v1/trellis2_long_street_v21/STREET_ROUTE.json',
    'selection_basis': 'Actual generated ORTHO_TOP, not a target map',
    'centerline_native_xy': [[-0.007, -0.45], [-0.007, 0.45]],
    'ground_native_z': -0.06219705939292908,
    'forward_native_xyz': [0, 1, 0],
    'forward_world_xyz': [0, 0, -1],
    'scope': 'Parent-road observations used as road-boundary conditions; not Map10 alignment evidence',
}


def condition_metadata():
    """Include the predeclared placement/condition labels in state and durable request."""
    return {
        'parent_chunk_id': 'native-v21-000',
        'parent_world_affine': PARENT_WORLD_AFFINE,
        'parent_world_affine_convention': 'row-major native XYZ to viewer XYZ',
        'parent_glTF_root_matrix': PARENT_GLTF_ROOT_MATRIX,
        'inherited_observed_road_conditions': INHERITED_OBSERVED_ROAD_CONDITIONS,
        'mechanism': 'Small-fraction noise-time boundary latent clamp in Structure flow; same fixed parent affine at export',
        'fixed_chunk_translation_GLTF': list(CHUNK_POSITION),
        'fixed_before_inference': True,
        'output_extent_fit': False,
        'map_aligned': False,
        'map10_alignment_claim': False,
        'seamless_guarantee': False,
    }


def configure_state(value):
    value.update(condition_metadata())
    value['chunk']['id'] = 'road-boundary-0--1'
    value['chunk']['position'] = list(CHUNK_POSITION)


def frontend(name):
    body = (V23_FRONTEND / name).read_text(encoding='utf8')
    if name == 'app.js':
        body = body.replace("const API_ORIGIN='http://127.0.0.1:8768';",
                            "const API_ORIGIN='http://127.0.0.1:8769';")
        body = body.replace('online-0--1', 'road-boundary-0--1')
        body = body.replace('[0,0,-100]', '[0,0,-100.0154161373455]')
        return body.replace('v=23', 'v=24').replace('v23_browser_metrics', 'v24_browser_metrics')
    body = body.replace('v23 · TRELLIS.2 Online', 'v24 · TRELLIS.2 Road Boundary Online')
    body = body.replace('V23 / CAMERA → ONLINE GENERATION', 'V24 / CAMERA → ROAD-BOUNDARY GENERATION')
    body = body.replace('<h1>Walk toward the next street segment</h1>', '<h1>v24 · Road-boundary-conditioned continuation</h1>')
    body = body.replace('127.0.0.1:8768', '127.0.0.1:8769')
    body = body.replace('Native world · Original v21 street + one online generation.',
                        'Native world · Inherited observations of the generated v21 road, with one road-boundary-conditioned generation.')
    body = body.replace('v21 is image-conditioned;',
                        'Conditions are inherited from observations of the generated v21 road; the fixed parent-world affine and chunk position are declared before inference; no Map10 alignment is claimed;')
    details = ('<details><summary>Fixed parent-world affine / inherited observed-road conditions</summary>'
               '<p>The original v21 GLB already contains the parent transform; it is documented here and not applied again. '
               'The road centerline, ground and direction come from observations of the generated parent street. The position is fixed at '
               '[0, 0, -100.0154161373455], with no guarantee of Map10 alignment or seamless joins.</p><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'
               + escape(json.dumps(condition_metadata(), ensure_ascii=False, indent=2))
               + '</pre></details>')
    return body.replace('</aside>', details + '</aside>')


def configure_backend(namespace):
    configure_state(namespace['state'])
    base = namespace['Handler']

    class BoundaryHandler(base):
        def send_head(self):
            path = urlsplit(self.path).path
            name = {'/': 'index.html', '/index.html': 'index.html', '/app.js': 'app.js'}.get(path)
            if name:
                content = frontend(name).encode('utf8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8' if name.endswith('.html')
                                 else 'text/javascript; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                return BytesIO(content)
            return super().send_head()

        def translate_path(self, path):
            name = {'/style.css': 'style.css', '/mesh_worker.js': 'mesh_worker.js'}.get(urlsplit(path).path)
            if name:
                return str(V23_FRONTEND / name)
            return super().translate_path(path)

    namespace['Handler'] = BoundaryHandler


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('v23 source changed; expected one occurrence of ' + repr(old))
    return source.replace(old, new, 1)


# Execute the preserved backend in this module's namespace so its functions,
# SSH connection, lock, durable queue and main loop share exactly one state.
_source = (JOB / 'v23_server.py').read_text(encoding='utf8')
_source = _source.replace('trellis2_online_v23', 'trellis2_boundary_online_v24')
_source = _source.replace('online-0--1', 'road-boundary-0--1')
_source = replace_once(_source, 'v23_native_job.py --detach', 'v24_native_job.py --detach')
_source = replace_once(_source, "for name in ['COMPLETE.json',",
                       "for name in ['CODEC_PREFLIGHT.json','BOUNDARY_TRACE.json','STRUCTURE_CHECK.json',"
                       "'FIXED_WORLD_EXPORT.json','GENERATED_STRUCTURE.npz','ROAD_OBSERVATION.npz',"
                       "'FINAL_HEIGHTFIELD.npz','FORWARD_VISIBILITY.npz','COMPLETE.json',")
_source = replace_once(_source, "'position':[0,0,-100]", "'position':[0,0,-100.0154161373455]")
_source = replace_once(_source, "request={**state['camera'],", "request={**condition_metadata(),'position_fixed':list(CHUNK_POSITION),**state['camera'],")
_source = replace_once(_source, 'default=8768', 'default=8769')
_source = replace_once(_source, 'V23_READY', 'V24_READY')
_source = replace_once(_source, "state=json.loads((OUT/'STATE.json').read_text(encoding='utf8'))",
                       "state=json.loads((OUT/'STATE.json').read_text(encoding='utf8'));configure_state(state)")
_source = replace_once(_source, "if __name__=='__main__':",
                       "configure_backend(globals())\nif __name__=='__main__':")
exec(compile(_source, str(JOB / 'v23_server.py'), 'exec'), globals())
