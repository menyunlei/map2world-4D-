"""Cameras and Gaussian rendering for the raw-map world branch.

World frame: the CityDreamer4D volume frame, ordered (row, col, z) in native
voxel units (4 units per metre). A CityDreamer pose {'cam_position': {x, y, z}}
maps to world (y, x, z), exactly as scripts.inference builds cam_origin.

The camera follows the native caster: focal = H/2 / tan(vfov), ray for pixel
(i, j) proportional to up*(cy - i - o) + side*s*(j + o - cx) + forward*focal.
The pixel offset o and side sign s are measured against the native caster's own
ray directions by export_teacher.py and stored with every view; they are not
assumed.
"""
import math

import torch


def pose_to_world(pose):
    """CityDreamer pose dict -> (origin [3], look_at [3]) in (row, col, z)."""
    p, t = pose['cam_position'], pose['cam_look_at']
    return (torch.tensor([p['y'], p['x'], p['z']], dtype=torch.float32),
            torch.tensor([t['y'], t['x'], t['z']], dtype=torch.float32))


def world_to_pose(origin, look_at):
    o, t = [float(v) for v in origin], [float(v) for v in look_at]
    return {'cam_position': {'x': o[1], 'y': o[0], 'z': o[2]},
            'cam_look_at': {'x': t[1], 'y': t[0], 'z': t[2]}}


def camera_axes(origin, look_at, side_sign=1.):
    forward = look_at - origin
    forward = forward / forward.norm()
    side = torch.linalg.cross(forward, torch.tensor([0., 0., 1.], dtype=forward.dtype))
    side = side_sign * side / side.norm()
    up = torch.linalg.cross(side_sign * side, forward)
    return forward, side, up


class Camera:
    """Pinhole camera in the world frame, OpenCV axes (x right, y down, z forward)."""

    def __init__(self, origin, look_at, height, width, focal, pixel_offset=0., side_sign=1.):
        self.origin = torch.as_tensor(origin, dtype=torch.float32)
        self.look_at = torch.as_tensor(look_at, dtype=torch.float32)
        self.height, self.width, self.focal = int(height), int(width), float(focal)
        self.pixel_offset, self.side_sign = float(pixel_offset), float(side_sign)
        forward, side, up = camera_axes(self.origin, self.look_at, self.side_sign)
        # Rows of R are the camera axes expressed in world coordinates.
        self.R = torch.stack([side, -up, forward])
        self.t = -self.R @ self.origin

    @classmethod
    def from_view(cls, view, scale=1.):
        origin, look_at = pose_to_world(view['pose'])
        return cls(origin, look_at, round(view['height'] * scale), round(view['width'] * scale),
                   view['focal'] * scale, view.get('pixel_offset', 0.), view.get('side_sign', 1.))

    @property
    def K(self):
        # Continuous pixel coordinate of pixel j's centre is j + 0.5.
        cx = self.width / 2 + 0.5 - self.pixel_offset
        cy = self.height / 2 + 0.5 - self.pixel_offset
        return torch.tensor([[self.focal, 0., cx], [0., self.focal, cy], [0., 0., 1.]])

    @property
    def viewmat(self):
        m = torch.eye(4)
        m[:3, :3], m[:3, 3] = self.R, self.t
        return m

    def ray_directions(self):
        """Unit world directions [H, W, 3], same formula as the native caster."""
        i = torch.arange(self.height, dtype=torch.float32)[:, None]
        j = torch.arange(self.width, dtype=torch.float32)[None, :]
        forward, side, up = camera_axes(self.origin, self.look_at, self.side_sign)
        o = self.pixel_offset
        d = (up * (self.height / 2 - i - o)[..., None] + side * (j + o - self.width / 2)[..., None]
             + forward * self.focal)
        return d / d.norm(dim=-1, keepdim=True)


def match_native_rays(native_dirs, origin, look_at, height, width, focal):
    """Pick (pixel_offset, side_sign) reproducing the native caster's ray directions.

    native_dirs: [H, W, 3] from get_voxel_intersection_perspective (raydirs[0, :, :, 0]).
    Returns (pixel_offset, side_sign, minimum cosine). Callers require > 0.99999.
    """
    native = native_dirs.float().cpu()
    native = native / native.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    best = (None, None, -2.)
    for offset in (0., .5):
        for sign in (1., -1.):
            ours = Camera(origin, look_at, height, width, focal, offset, sign).ray_directions()
            cos = (ours * native).sum(-1).min().item()
            if cos > best[2]:
                best = (offset, sign, cos)
    return best


# ------------------------------------------------------------------ rendering
def quat_to_rotmat(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def render_reference(g, camera, near=1., pixel_chunk=4096):
    """Differentiable dense EWA splatting in plain torch. O(pixels x Gaussians):
    for CPU tests and tiny debugging scenes only. Same outputs as render_gsplat.
    g: dict with means [N,3], quats [N,4] (wxyz), scales [N,3], opacities [N], colors [N,3].
    Returns rgb [H,W,3], alpha [H,W], depth [H,W] (expected z, 0 where alpha = 0).
    """
    dev = g['means'].device
    R, t = camera.R.to(dev), camera.t.to(dev)
    K = camera.K.to(dev)
    cam = g['means'] @ R.T + t
    keep = cam[:, 2] > near
    cam = cam[keep]
    z = cam[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mean2d = torch.stack([fx * cam[:, 0] / z + cx, fy * cam[:, 1] / z + cy], -1)
    Rg = quat_to_rotmat(g['quats'][keep])
    S = torch.diag_embed(g['scales'][keep])
    cov = R @ (Rg @ S @ S.transpose(1, 2) @ Rg.transpose(1, 2)) @ R.T
    J = torch.zeros((cam.shape[0], 2, 3), device=dev)
    J[:, 0, 0], J[:, 0, 2] = fx / z, -fx * cam[:, 0] / z ** 2
    J[:, 1, 1], J[:, 1, 2] = fy / z, -fy * cam[:, 1] / z ** 2
    cov2d = J @ cov @ J.transpose(1, 2) + 0.3 * torch.eye(2, device=dev)
    conic = torch.linalg.inv(cov2d)
    order = torch.argsort(z)
    mean2d, conic, z = mean2d[order], conic[order], z[order]
    opacity = g['opacities'][keep][order]
    color = g['colors'][keep][order]
    H, W = camera.height, camera.width
    v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32) + .5,
                          torch.arange(W, device=dev, dtype=torch.float32) + .5, indexing='ij')
    pix = torch.stack([u.reshape(-1), v.reshape(-1)], -1)
    rgb, acc, dep = [], [], []
    for s in range(0, pix.shape[0], pixel_chunk):
        d = pix[s:s + pixel_chunk, None, :] - mean2d[None]                        # [P,N,2]
        power = -.5 * (d[..., 0] ** 2 * conic[:, 0, 0] + 2 * d[..., 0] * d[..., 1] * conic[:, 0, 1]
                       + d[..., 1] ** 2 * conic[:, 1, 1])
        alpha = (opacity * torch.exp(power.clamp(max=0))).clamp(max=.99)
        alpha = torch.where(alpha < 1 / 255, torch.zeros_like(alpha), alpha)
        T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], 1), 1)
        w = alpha * T
        rgb.append(w @ color)
        a = w.sum(1)
        acc.append(a)
        dep.append((w @ z) / a.clamp_min(1e-8))
    return (torch.cat(rgb).reshape(H, W, 3), torch.cat(acc).reshape(H, W), torch.cat(dep).reshape(H, W))


def render_gsplat(g, camera, near=1., far=1e5):
    """Tile rasterizer (gsplat >= 1.0) for real scenes on GPU."""
    from gsplat import rasterization
    dev = g['means'].device
    out, alpha, _ = rasterization(g['means'], g['quats'], g['scales'], g['opacities'], g['colors'],
                                  camera.viewmat.to(dev)[None], camera.K.to(dev)[None],
                                  camera.width, camera.height, near_plane=near, far_plane=far,
                                  render_mode='RGB+ED', packed=False)
    return out[0, ..., :3], alpha[0, ..., 0], out[0, ..., 3]


def render(g, camera, backend='auto', **kw):
    if backend == 'auto':
        backend = 'gsplat' if g['means'].is_cuda else 'reference'
    return (render_gsplat if backend == 'gsplat' else render_reference)(g, camera, **kw)


def fov_focal(height, vfov_deg):
    """CityDreamer4D convention: the configured VFOV acts as a half-angle."""
    return height / 2 / math.tan(math.radians(vfov_deg))
