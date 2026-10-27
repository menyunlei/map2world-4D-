"""Balanced native renderer with length-delimited keep-alive SSH HTTP transport."""
import json, os, signal, threading
from http.server import ThreadingHTTPServer
import native_balanced_v4 as roi
engine = roi.engine
# http.client closes HTTP/1.0 sockets before body consumption; Paramiko channels
# do not have socket.makefile's independent close lifetime. Keep the channel open
# until the existing bridge has read the Content-Length bytes and closes it.
engine.one.Handler.protocol_version = 'HTTP/1.1'

if __name__ == '__main__':
    def shutdown(*unused):
        engine.pool.release()
        os._exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    server = ThreadingHTTPServer(('127.0.0.1', engine.one.PORT), engine.one.Handler)
    engine.one.state.update(phase='four-GPU balanced native renderer ready',gpus=[0,1,2,3])
    threading.Thread(target=engine.coordinator,daemon=True).start()
    print(json.dumps(dict(pid=os.getpid(),port=engine.one.PORT,mode='native balanced v5 transport')),flush=True)
    server.serve_forever()
