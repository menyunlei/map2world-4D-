"""One-shot verified-host task SSH. Credentials are supplied in process memory."""
import argparse, hashlib, json, os, sys
from pathlib import Path, PurePosixPath
import paramiko

JOB=Path(__file__).resolve().parent
REMOTE='/hpc2hdd/home/xhe989/Documents/2541/project2/DreamX-World-master/experiments/recursive_map_research_v1'

def main():
    p=argparse.ArgumentParser();p.add_argument('request');a=p.parse_args()
    request=json.loads(Path(a.request).read_text(encoding='utf-8'))
    client=paramiko.SSHClient()
    for path in [Path.home()/'.ssh/known_hosts',JOB.parents[1]/'work_known_hosts']:
        if path.is_file():client.load_host_keys(str(path))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password=os.environ.pop('MAP_CITY_SSH_PASSWORD')
    client.connect('10.120.18.240',port=6988,username='xhe989-YKWbGTug',password=password,
                   timeout=25,allow_agent=False,look_for_keys=False)
    password=None
    try:
        results=[]
        for op in request['operations']:
            if op['op']=='exec':
                _,out,err=client.exec_command(op['command'],timeout=op.get('timeout',60))
                results.append(dict(op='exec',stdout=out.read().decode(errors='replace'),
                                    stderr=err.read().decode(errors='replace'),exit_code=out.channel.recv_exit_status()))
            elif op['op'] in ['upload','download']:
                files=[]
                with client.open_sftp() as ftp:
                    for name in op['files']:
                        rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
                        dest=PurePosixPath(op.get('destination_map',{}).get(name,name)) if op['op']=='download' else rel
                        assert not dest.is_absolute() and '..' not in dest.parts
                        local=JOB.joinpath(*dest.parts);remote=REMOTE+'/'+str(rel)
                        if op['op']=='upload':
                            data=local.read_bytes();sha=hashlib.sha256(data).hexdigest()
                            try:
                                with ftp.open(remote,'rb') as f:old=hashlib.sha256(f.read()).hexdigest()
                                assert old==sha, 'Preserve changed remote files: '+name
                            except FileNotFoundError:
                                current=PurePosixPath(REMOTE)
                                for part in rel.parent.parts:
                                    current/=part
                                    try:ftp.stat(str(current))
                                    except FileNotFoundError:ftp.mkdir(str(current))
                                with ftp.open(remote,'wx') as f:f.write(data)
                        else:
                            local.parent.mkdir(parents=True,exist_ok=True)
                            if local.exists():
                                with ftp.open(remote,'rb') as f:sha=hashlib.sha256(f.read()).hexdigest()
                                assert hashlib.sha256(local.read_bytes()).hexdigest()==sha
                            else:
                                h=hashlib.sha256()
                                with ftp.open(remote,'rb') as src,local.open('xb') as dst:
                                    while True:
                                        b=src.read(2**20)
                                        if not b:break
                                        dst.write(b);h.update(b)
                                sha=h.hexdigest()
                        files.append(dict(file=name,sha256=sha,bytes=local.stat().st_size))
                results.append(dict(op=op['op'],files=files))
            else:raise ValueError(op['op'])
        print(json.dumps(results,ensure_ascii=False))
    finally:client.close()

if __name__=='__main__':main()
