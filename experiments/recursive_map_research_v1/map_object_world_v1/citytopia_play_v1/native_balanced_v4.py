"""Skip building rays that cannot affect any visible output pixel.

Native building RenderCNN has four 3x3, stride-one convolutions and otherwise
pointwise operations, no spatial normalization. Its receptive radius is four
pixels. Keep eight pixels of original context, clipped to original padded tile.
No change to density, samples per ray, model weights or rendered resolution.
"""
import json, os, signal, threading
from http.server import ThreadingHTTPServer
import native_balanced_v3 as device_correct
engine = device_correct.engine


def tile(self, index):
    torch, e, n = self.torch, self.env, self.native
    assert not (self.bldg.cfg.SKY_ENABLED and self.bldg.cfg.SKY_GLOBAL_AVGPOOL)
    sy, sx = (index//4)*135, (index%4)*240
    ey, ex = sy+135, sx+240
    psx, pex, psy, pey = n.get_pad_img_bbox(sx, ex, sy, ey, 540, 960, 8)
    ids = self.rays[0][:, psy:pey, psx:pex]
    depths = self.rays[1][:, psy:pey, psx:pex]
    dirs = self.rays[2][:, psy:pey, psx:pex]
    origin = self.rays[3]
    bgids = self.bg_ids[:, psy:pey, psx:pex]
    bounds = (sx, ex, sy, ey, psx, pex, psy, pey, 8)
    output, _ = self.forward(self.bg, self.bg_features, bgids, depths, dirs, origin)
    road = (bgids[:, None, :, :, 0, 0] == e['inst']['ROAD_CID']).repeat(1, 3, 1, 1).float()
    output = self.blurrer(output)*road + output*(1-road)
    result = n.get_img_without_pad(output, *bounds).clone()
    core = self.rays[0][0, sy:ey, sx:ex, 0, 0]
    buildings = n.get_visible_buildings(core, e['inst']).tolist()
    import numpy as np
    core_cpu = core.cpu().numpy()
    for building in buildings:
        if building not in e['stats'] or building not in e['styles']:
            continue
        roof_id = building+e['inst']['ROOF_OFFSET']
        yy, xx = np.where((core_cpu == building) | (core_cpu == roof_id))
        if not len(xx):
            continue
        # The predicted, visible RGB has exactly its native convolution support.
        x0, x1 = sx+int(xx.min()), sx+int(xx.max())+1
        y0, y1 = sy+int(yy.min()), sy+int(yy.max())+1
        ax, bx = max(psx,x0-8), min(pex,x1+8)
        ay, by = max(psy,y0-8), min(pey,y1+8)
        bid_source = self.rays[0][:, ay:by, ax:bx]
        bid = bid_source.clone()
        facade, roof = bid_source == building, bid_source == roof_id
        bid[~(facade | roof)] = 0
        bid[facade] = e['inst']['FACADE_CID']
        bid[roof] = e['inst']['ROOF_CID']
        bdirs = self.rays[2][:, ay:by, ax:bx].clone()
        bdirs[bid[..., 0, 0] == 0] = 0
        features, stats = self.building_features(building)
        output, _ = self.forward(self.bldg, features, bid, self.rays[1][:, ay:by, ax:bx], bdirs,
                                 origin, stats, e['styles'][building])
        local_core = core[y0-sy:y1-sy,x0-sx:x1-sx]
        mask = ((local_core == building) | (local_core == roof_id))[None,None].float()
        previous = result[:,:,y0-sy:y1-sy,x0-sx:x1-sx]
        result[:,:,y0-sy:y1-sy,x0-sx:x1-sx] = previous*(1-mask)+output[:,:,y0-ay:y1-ay,x0-ax:x1-ax]*mask
    return engine.one.base.rgb(result).copy(), len(buildings)


engine.TileEngine.tile = tile

if __name__ == '__main__':
    def shutdown(*unused):
        engine.pool.release()
        os._exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    server = ThreadingHTTPServer(('127.0.0.1', engine.one.PORT), engine.one.Handler)
    engine.one.state.update(phase='four-GPU visible-surface renderer ready',gpus=[0,1,2,3])
    threading.Thread(target=engine.coordinator,daemon=True).start()
    print(json.dumps(dict(pid=os.getpid(),port=engine.one.PORT,mode='native balanced v4 ROI')),flush=True)
    server.serve_forever()
