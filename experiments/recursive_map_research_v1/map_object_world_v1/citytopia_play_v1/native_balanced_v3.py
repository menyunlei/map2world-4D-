"""Device-correct cached native renderer (DataParallel used to transfer stats)."""
import json, os, signal, threading
from http.server import ThreadingHTTPServer
import native_balanced_v2 as engine

original_forward = engine.TileEngine.forward

def forward(self, model, features, ids, depths, dirs, origin, stats=None, style=None):
    if stats is not None:
        stats = stats.to(origin.device)
    return original_forward(self, model, features, ids, depths, dirs, origin, stats, style)

engine.TileEngine.forward = forward

if __name__ == '__main__':
    def shutdown(*unused):
        engine.pool.release()
        os._exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    server = ThreadingHTTPServer(('127.0.0.1', engine.one.PORT), engine.one.Handler)
    engine.one.state.update(phase='balanced renderer ready; load on demand', gpus=[0, 1, 2, 3])
    threading.Thread(target=engine.coordinator, daemon=True).start()
    print(json.dumps(dict(pid=os.getpid(),port=engine.one.PORT,mode='native balanced v3')),flush=True)
    server.serve_forever()
