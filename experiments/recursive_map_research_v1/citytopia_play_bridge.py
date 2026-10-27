"""Minimal CityTopia mode of the existing World Lab, with an SSH native-render relay."""
import http.client, json, mimetypes, os, re, threading
from pathlib import Path, PurePosixPath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import paramiko

JOB=Path(__file__).resolve().parent
UI=JOB/'map_object_world_v1/world_workbench_v1/ui'
OUT=JOB/'map_object_world_v1/citytopia_play_v1'
VENDOR=JOB/'matched_backbone_adaptation_v1/asset_delivery_v3/interactive/dist/vendor'
REMOTE='/hpc2hdd/home/xhe989/Documents/2541/project2/DreamX-World-master/experiments/recursive_map_research_v1/map_object_world_v1/citytopia_play_v1'
PORT=18936

def main():
    ssh=paramiko.SSHClient()
    for path in [Path.home()/'.ssh/known_hosts',JOB.parents[1]/'work_known_hosts']:
        if path.is_file():ssh.load_host_keys(str(path))
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    ssh.connect('10.120.18.240',port=6988,username='xhe989-YKWbGTug',password=os.environ.pop('MAP_CITY_SSH_PASSWORD'),allow_agent=False,look_for_keys=False,timeout=25)
    ssh.get_transport().set_keepalive(30)
    def rpc(method,body=None):
        ch=ssh.get_transport().open_channel('direct-tcpip',('127.0.0.1',18935),('127.0.0.1',0),timeout=15)
        conn=http.client.HTTPConnection('127.0.0.1',18935,timeout=20);conn.sock=ch
        try:
            conn.request(method,'/render' if method=='POST' else '/status',body,{'Content-Type':'application/json'})
            res=conn.getresponse();return res.status,res.read()
        finally:conn.close()
    html=(UI/'index.html').read_text(encoding='utf-8')
    html=re.sub(r'<header>.*?</header>','<header><div class="brand"><b>WORLD<span>LAB</span></b><small>EXPERIMENT B · CITYTOPIA / MAP10</small></div><div id="health" class="pill">连接中</div></header>',html,flags=re.S)
    html=re.sub(r'<aside>.*?</aside>','',html,flags=re.S)
    html=re.sub(r'<section class="timeline">.*?</section>','',html,flags=re.S)
    html=html.replace('</head>','''<style>main{display:block!important;max-width:1400px;margin:auto}.selector,.subbar,#map-compare,#onboarding,#building-labels{display:none!important}#viewport{height:auto;aspect-ratio:16/9;min-height:320px;max-height:78vh}#render{width:100%;height:100%}#minimap{display:none}#loading{top:50%;height:max-content}#native-image{position:absolute;inset:0;width:100%;height:100%;object-fit:fill;pointer-events:none}#labels{z-index:5;pointer-events:none}#hint{z-index:6}#move-pad{z-index:7}#crosshair{z-index:6}#camera-bar{flex-wrap:wrap}footer{padding:8px}#case-note{padding:8px;font-size:13px}#map-source{padding:8px}</style></head>''')
    html=html.replace('正在读取已保存的真实 GLB…','加载已存 CityTopia 原生 RGB / 深度预览…')
    html=html.replace('Source assets immutable · 原生 GLB / 研究失败结果 / 混合世界分开记录','CityTopia 原生模型 · Map10 固定条件和风格 · 本轮无训练 / 无逐帧增强 / 非动态4D')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def reply(self,code,data,kind='application/json'):
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-cache');self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            path=urlsplit(self.path).path
            try:
                if path in ['/','/index.html']:return self.reply(200,html.encode(),'text/html; charset=utf-8')
                if path=='/api/status':
                    code,data=rpc('GET');return self.reply(code,data)
                local={'/app.js':OUT/'play.js','/style.css':UI/'style.css','/vendor/three.module.js':VENDOR/'three.module.js'}
                if path in local:return self.reply(200,local[path].read_bytes(),mimetypes.guess_type(path)[0] or 'text/javascript')
                rel=PurePosixPath(path.lstrip('/'))
                if '..' in rel.parts or len(rel.parts)!=2 or rel.parts[0] not in ['preview','renders']:return self.reply(404,b'{}')
                cached=OUT.joinpath(*rel.parts)
                if not cached.is_file():
                    with ssh.open_sftp() as ftp:
                        with ftp.open(REMOTE+'/'+str(rel),'rb') as f:data=f.read()
                    cached.parent.mkdir(exist_ok=True,parents=True);cached.write_bytes(data)
                return self.reply(200,cached.read_bytes(),mimetypes.guess_type(path)[0] or 'application/octet-stream')
            except Exception as e:self.reply(503,json.dumps({'error':str(e)}).encode())
        def do_POST(self):
            if urlsplit(self.path).path!='/api/render':return self.reply(404,b'{}')
            if self.headers.get('Origin') not in [None,f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}']:return self.reply(403,b'{}')
            try:
                n=int(self.headers.get('Content-Length',0));assert 0<n<8192
                code,data=rpc('POST',self.rfile.read(n));self.reply(code,data)
            except Exception as e:self.reply(503,json.dumps({'error':str(e)}).encode())
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    print(f'http://127.0.0.1:{PORT}/',flush=True)
    (OUT/'LOCAL_SERVICE.json').write_text(json.dumps({'pid':os.getpid(),'port':PORT}))
    try:server.serve_forever()
    finally:server.server_close();ssh.close()

if __name__=='__main__':main()
