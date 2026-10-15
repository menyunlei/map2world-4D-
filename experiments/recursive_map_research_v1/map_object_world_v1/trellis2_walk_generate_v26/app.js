import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';
const API_ORIGIN=location.origin;
const assetURL=url=>new URL(url,API_ORIGIN).href;
const $=id=>document.getElementById(id),vp=$('viewport'),boot=performance.now(),session=crypto.randomUUID();
const renderer=new THREE.WebGLRenderer({antialias:false,powerPreference:'high-performance'});renderer.setPixelRatio(1);renderer.setClearColor('#26313b');renderer.outputColorSpace=THREE.SRGBColorSpace;
vp.appendChild(renderer.domElement);renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label','三维交互视口');
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(65,1,.03,80),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;
const groups={native:new THREE.Group(),online:new THREE.Group()};Object.values(groups).forEach(g=>scene.add(g));
const worker=new Worker('/mesh_worker.js?v=26'),pending=new Map(),cache=new Map(),keys=new Set(),events=[],samples=[],dir=new THREE.Vector3();
const world='native';
let mode='walk',yaw=0,pitch=0,drag=null,api,last=performance.now(),fpsAt=last,frames=0,miniAt=0;
let nativeLoading=false,onlineLoading=false,cameraRevision=0,reportedRevision=0,cameraSending=false,navigationPose='',userOrbit=false;
let fps=0,lastFrameMs=0,maxFrameMs=0,hiddenFrames=0,renderScale=1,scaleAt=0,inputAt=0,inputFrameMs=0,cameraRttMs=0;
const gl=renderer.getContext(),debugGPU=gl.getExtension('WEBGL_debug_renderer_info');
const metrics={session,bootEpoch:Date.now(),userAgent:navigator.userAgent,renderer:debugGPU?gl.getParameter(debugGPU.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER),loads:{},events,samples,limitations:['rAF submission FPS, not hardware GPU timer','exact triangle draw batches with frustum culling; no geometry simplification','free camera without collision certification','two bounded predictive camera-triggered generations; no training','online chunk has no seam or map-control guarantee']};
function event(type,data={}){events.push({type,t:performance.now()-boot,epoch:Date.now(),world,...data});if(events.length>1000)events.shift();}
function error(e){$('error').hidden=false;$('error').textContent=String(e);event('error',{message:String(e)});}
addEventListener('error',e=>error(e.message));addEventListener('unhandledrejection',e=>error(e.reason));
function resize(){const w=vp.clientWidth,h=vp.clientHeight;renderer.setPixelRatio(renderScale);renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}new ResizeObserver(resize).observe(vp);resize();
function look(){camera.rotation.order='YXZ';camera.rotation.set(pitch,yaw,0);camera.updateMatrixWorld();}
function setWalk(){mode='walk';controls.enabled=false;camera.getWorldDirection(dir);yaw=Math.atan2(-dir.x,-dir.z);pitch=Math.asin(dir.y);look();}
function aim(p,target){camera.position.fromArray(p);camera.lookAt(new THREE.Vector3(...target));setWalk();}
function home(){keys.clear();drag=null;camera.far=Number($('distance').value);camera.updateProjectionMatrix();aim([-.70,2.73,45],[-.70,2.73,-45]);event('home',{position:camera.position.toArray()});}
function top(){mode='orbit';camera.far=800;camera.updateProjectionMatrix();controls.enabled=true;camera.position.set(100,220,100);controls.target.set(0,0,-150);controls.update();keys.clear();event('top');}
function poseKey(){return [...camera.position.toArray(),...camera.quaternion.toArray()].map(n=>n.toFixed(8)).join(',');}
function noteNavigation(){const pose=poseKey();if(pose===navigationPose)return;navigationPose=pose;cameraRevision++;}
function userAction(action){inputAt=performance.now();action();noteNavigation();renderer.domElement.focus();}
controls.addEventListener('start',()=>{userOrbit=true;});
controls.addEventListener('change',()=>{if(userOrbit)noteNavigation();});
controls.addEventListener('end',()=>{if(userOrbit)noteNavigation();userOrbit=false;});
function vis(){groups.native.visible=true;groups.online.visible=true;}
function load(id,url,sha){
 const key=id+':'+sha;
 if(cache.has(key)){event('cache_hit',{id,sha256:sha});return Promise.resolve(cache.get(key));}
 if(pending.has(key))return pending.get(key).promise;
 const start=performance.now();event('load_start',{id,sha256:sha});$('loading').textContent=`后台读取 ${id}，移动无需等待…`;
 let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});
 pending.set(key,{id,promise,resolve,reject,start});worker.postMessage({id:key,url:assetURL(url),sha256:sha});return promise;
}
worker.onmessage=({data:d})=>{const job=pending.get(d.id);if(!job)return;if(!d.ok){pending.delete(d.id);job.reject(Error(d.error));return;}try{
 const root=new THREE.Group();let triangles=0,vertices=0;const geometries=d.meshes.map(prims=>prims.flatMap(p=>{const attrs={};for(const [name,a] of [['position',p.position],['color',p.color],['normal',p.normal]])if(a)attrs[name]=new THREE.BufferAttribute(a.array,a.itemSize,a.normalized);const index=new THREE.BufferAttribute(p.index.array,1);vertices+=p.position.array.length/3;triangles+=p.index.array.length/3;return p.batches.map(b=>{const g=new THREE.BufferGeometry();for(const [name,a] of Object.entries(attrs))g.setAttribute(name,a);g.setIndex(index);g.setDrawRange(b.start,b.count);g.boundingBox=new THREE.Box3(new THREE.Vector3(...b.min),new THREE.Vector3(...b.max));g.boundingSphere=g.boundingBox.getBoundingSphere(new THREE.Sphere());return g;});}));
 function node(i){const n=d.nodes[i],o=new THREE.Group();if(n.matrix){o.matrix.fromArray(n.matrix);o.matrixAutoUpdate=false;}else{if(n.translation)o.position.fromArray(n.translation);if(n.rotation)o.quaternion.fromArray(n.rotation);if(n.scale)o.scale.fromArray(n.scale);}
 if(n.mesh!==undefined)for(const g of geometries[n.mesh])o.add(new THREE.Mesh(g,new THREE.MeshBasicMaterial({vertexColors:!!g.attributes.color,side:THREE.DoubleSide,color:0xffffff})));for(const child of n.children||[])o.add(node(child));return o;}
 d.roots.forEach(i=>root.add(node(i)));const record={id:job.id,root,sha256:d.sha256,triangles,vertices,bytes:d.bytes,loadMs:performance.now()-job.start,workerMs:d.workerMs,loadStart:job.start};cache.set(d.id,record);pending.delete(d.id);metrics.loads[job.id]={...record,root:undefined};event('load_complete',{id:job.id,sha256:d.sha256,ms:record.loadMs,triangles});$('loading').textContent=`${job.id}：${(record.loadMs/1000).toFixed(2)}s 读取解析 / ${(d.bytes/1e6).toFixed(1)}MB / ${(triangles/1e6).toFixed(2)}M 三角形`;
 $('cache').textContent=[...cache.values()].map(v=>`${v.id}: ${v.sha256.slice(0,10)}`).join(' | ');job.resolve(record);
 }catch(e){pending.delete(d.id);job.reject(e);}};
worker.onerror=e=>{for(const job of pending.values())job.reject(Error(e.message||'Mesh worker failed'));pending.clear();error(e.message||'Mesh worker failed');};
async function ensureNative(){
 if(nativeLoading)return;nativeLoading=true;
 try{
  const rec=await load('native-v21-000',api?.v21?.asset_url||'/assets/v21.glb',api?.v21?.sha256||'b372c05fbdd2eba6b5978b15cf7a25fcb1e238aea47a9d93ca6fbe3e40f016fe');
  if(!rec.root.parent)groups.native.add(rec.root);vis();
  $('init').textContent='原始 v21 已载入';event('native_ready',{sha256:rec.sha256});
 }catch(e){error(e);}finally{nativeLoading=false;}
}
async function ensureOnline(){
 if(onlineLoading||!api)return;onlineLoading=true;
 try{
  for(const c of [...(api.cached||[]),...(api.chunks||[])]){
   const final=c.status==='ready'&&c.sha256;
   const sha=final?c.sha256:c.preview_sha256,url=final?c.asset_url:c.preview_url;
   if(!sha||!url)continue;
   const id=c.id+(final?'':'.preview'),key=id+':'+sha;
   if(cache.has(key)&&cache.get(key).root.parent)continue;
   const rec=await load(id,url,sha);
   if(final){
    const draft=[...cache.entries()].find(([k,r])=>r.id===c.id+'.preview');
    if(draft){
     const [key,r]=draft;r.root.removeFromParent();
     r.root.traverse(o=>{if(o.geometry)o.geometry.dispose();if(o.material)o.material.dispose();});
     r.root.clear();cache.delete(key);
     event('draft_replaced',{id:c.id,old_sha256:r.sha256,new_sha256:sha,only_draft_removed:true});
    }
   }
   if(!rec.root.parent){rec.root.position.fromArray(c.position);groups.online.add(rec.root);
    event(final?'chunk_arrival':'geometry_preview_arrival',{id:c.id,sha256:sha,camera:camera.position.toArray(),pose_unchanged:true});}
   $('cache').textContent=[...cache.values()].map(v=>`${v.id}: ${v.sha256.slice(0,10)}`).join(' | ');vis();break;
  }
 }catch(e){error(e);}finally{onlineLoading=false;}
}
const statusLabels={waiting_camera:'等待相机进入触发区域',queued:'已排队',generating:'GPU 生成中',exporting:'导出原始 GLB',transferring:'传输资产',ready:'资产就绪',failed:'本次尝试失败'};
function showState(){
 const chunks=api.chunks||[];
 $('budget').textContent=`预算 ${api.budget?.used??0} / ${api.budget?.max_generations??2}`;
 $('job').textContent=chunks.map(c=>`${c.id} · ${statusLabels[c.status]||c.status}`).join(' / ');
 $('stage').textContent='常驻服务：'+(api.worker?.stage||'连接中')+' · '+(api.worker?.gpu?'A800 '+api.worker.gpu.index:'')+' | '+chunks.map(c=>`${c.id}: ${c.stage||c.status}${c.gpu?' · A800 '+c.gpu.index:''}${c.status==='generating'&&c.gpu_stats?' · '+c.gpu_stats.utilization_percent+'% / '+(c.gpu_stats.memory_MiB/1024).toFixed(1)+'GiB':''}`).join(' | ');
 $('elapsed').textContent=chunks.map(c=>`${c.id}: ${(c.elapsed_seconds||0).toFixed(1)}s${c.preview_seconds?' · 几何到本机 '+c.preview_seconds.toFixed(1)+'s':''}`).join(' | ');
 const w=api.worker||{},idle=w.stage==='resident_ready'?Math.max(0,300-(Date.now()/1000-w.epoch)):null; $('message').textContent=w.stage==='stopped'||w.stage==='failed'?'常驻服务已停止；缓存仍可浏览，但本轮不再生成。':chunks.every(c=>c.status==='ready')?'三块预算已完成，原世界保持缓存。':w.stage==='resident_ready'?`模型已驻留内存；还可生成 ${Math.max(0,3-(api.budget?.used||0))} 块。空闲释放剩余约 ${idle.toFixed(0)} 秒。继续向前走才触发下一块。`:'常驻服务处理中；可以继续移动。预热、排队、GPU推理阶段如实显示。';
 $('trigger').textContent='先进入已缓存前沿，再按W前行；每次只生成一块。上一块已加载且继续前进才触发下一块。三块硬上限，非无限生成。';
 $('online-download').hidden=!chunks.some(c=>c.status==='ready');
 const ready=chunks.find(c=>c.status==='ready');if(ready)$('online-download').href=assetURL(ready.asset_url);
}
async function poll(){
 try{
  const r=await fetch(API_ORIGIN+'/api/state',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);api=await r.json();showState();
  $('connection').textContent='API 已连接 · 状态每0.5秒刷新';
  if(![...cache.values()].some(r=>r.id==='native-v21-000'))void ensureNative();void ensureOnline();
 }catch(e){$('connection').textContent='状态连接中断 · '+e.message+' · 已载入场景可继续浏览';}
 setTimeout(poll,500);
}
async function reportCamera(){
 if(cameraSending||cameraRevision===reportedRevision||document.hidden)return;
 const revision=cameraRevision;camera.getWorldDirection(dir);
 const payload={session,world,position:camera.position.toArray(),forward:dir.toArray(),fps,loaded:[...cache.values()].map(r=>({id:r.id,sha256:r.sha256})),event:'camera'};
 cameraSending=true;const sendAt=performance.now();
 try{
  const r=await fetch(API_ORIGIN+'/api/camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!r.ok)throw Error('HTTP '+r.status);
  const ack=await r.json();cameraRttMs=performance.now()-sendAt;reportedRevision=revision;event('camera_report',{position:payload.position,forward:payload.forward,rtt_ms:cameraRttMs,accepted:ack.accepted});
 }catch(e){event('camera_report_error',{message:String(e)});}finally{cameraSending=false;}
}
setInterval(reportCamera,100);
function mini(){
 const c=$('minimap').getContext('2d'),bounds=[-65,65,-665,65],xy=(x,z)=>[12+(x-bounds[0])/(bounds[1]-bounds[0])*236,32+(z-bounds[2])/(bounds[3]-bounds[2])*232];
 c.fillStyle='#182630';c.fillRect(0,0,260,300);c.lineWidth=1;
 const block=(x,z,pendingBlock,label)=>{
  const a=xy(x-50,z-50),b=xy(x+50,z+50);c.fillStyle=pendingBlock?'#20303b':'#29443f';c.fillRect(...a,b[0]-a[0],b[1]-a[1]);
  c.strokeStyle=pendingBlock?'#f0ba65':'#80dbc0';c.setLineDash(pendingBlock?[5,4]:[]);c.strokeRect(...a,b[0]-a[0],b[1]-a[1]);c.setLineDash([]);
  c.fillStyle=pendingBlock?'#edc48a':'#a5d8cb';c.font='11px system-ui';c.fillText(label,a[0]+5,a[1]+15);
 };
 block(0,0,![...cache.values()].some(r=>r.id==='native-v21-000'),'v21 缓存');
 for(const c of [...(api?.cached||[]),...(api?.chunks||[])]){
 const loaded=[...cache.values()].some(r=>(r.id===c.id||r.id===c.id+'.preview')&&r.root.parent);
 block(c.position[0],c.position[2],!loaded,c.id+(loaded?' 已载入':' '+(c.stage||c.status)));
 }
 camera.getWorldDirection(dir);const p=xy(camera.position.x,camera.position.z),q=xy(camera.position.x+dir.x*8,camera.position.z+dir.z*8);
 c.fillStyle='#ff7649';c.beginPath();c.arc(...p,4,0,7);c.fill();c.strokeStyle='#ff7649';c.lineWidth=2;c.beginPath();c.moveTo(...p);c.lineTo(...q);c.stroke();
 c.font='11px system-ui';c.fillStyle='#e4edf3';c.fillText('位置示意 · 虚线为待载入块',12,17);c.fillText('橙色：真实相机 / 绿色：触发线',12,286);
}
function snapshot(){camera.getWorldDirection(dir);return {session,epoch:Date.now(),world,mode,position:camera.position.toArray(),forward:dir.toArray(),quaternion:camera.quaternion.toArray(),fps,maxFrameMs,renderScale,inputFrameMs,cameraRttMs,residentWorker:api?.worker,visible:!document.hidden,loaded:[...cache.values()].map(r=>({id:r.id,sha256:r.sha256,triangles:r.triangles})),pending:[...pending.values()].map(r=>r.id),renderSize:[renderer.domElement.width,renderer.domElement.height],chunkStatus:api?.chunks?.map(c=>({id:c.id,status:c.status,stage:c.stage})),budget:api?.budget,generation_started_epoch:api?.generation_started_epoch,generation_elapsed_seconds:api?.generation_elapsed_seconds};}
function telemetry(){const s=snapshot();samples.push(s);if(samples.length>600)samples.shift();fetch(API_ORIGIN+'/api/telemetry',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({snapshot:s,renderer:metrics.renderer,loads:metrics.loads,events:events.slice(-40)}),keepalive:false}).then(r=>{if(!r.ok)event('telemetry_http_error',{status:r.status});return r.json();}).catch(e=>event('telemetry_error',{message:String(e)}));}setInterval(telemetry,10000);
function frame(now){requestAnimationFrame(frame);const raw=now-last;last=now;if(document.hidden){hiddenFrames++;return;}const dt=Math.min(raw/1000,.05);maxFrameMs=Math.max(maxFrameMs,raw);lastFrameMs=raw;
 if(mode==='walk'){const f=Number(keys.has('w'))-Number(keys.has('s')),r=Number(keys.has('d'))-Number(keys.has('a')),speed=(keys.has('shift')?8:2)*dt;camera.position.x+=(-Math.sin(yaw)*f+Math.cos(yaw)*r)*speed;camera.position.z+=(-Math.cos(yaw)*f-Math.sin(yaw)*r)*speed;camera.position.y+=(Number(keys.has('e'))-Number(keys.has('q')))*speed;yaw+=(Number(keys.has('arrowleft'))-Number(keys.has('arrowright')))*dt;pitch=THREE.MathUtils.clamp(pitch+(Number(keys.has('arrowup'))-Number(keys.has('arrowdown')))*dt,-1.45,1.45);look();if(f||r||keys.has('e')||keys.has('q')||keys.has('arrowleft')||keys.has('arrowright')||keys.has('arrowup')||keys.has('arrowdown'))noteNavigation();}else controls.update();renderer.render(scene,camera);if(inputAt){inputFrameMs=performance.now()-inputAt;event('input_render_submission',{ms:inputFrameMs});inputAt=0;}
 
 for(const r of cache.values()){const id=r.id;if(!r.firstVisibleMs&&r.root.parent?.visible){r.firstVisibleMs=performance.now()-r.loadStart;metrics.loads[id].firstVisibleMs=r.firstVisibleMs;event('first_asset_render',{id,ms:r.firstVisibleMs});}}
 frames++;if(now-fpsAt>=1000){fps=frames*1000/(now-fpsAt);$('fps').textContent=`${fps.toFixed(1)} FPS · ${(renderer.info.render.triangles/1e6).toFixed(2)}M drawn tris`;$('pos').textContent=`X${camera.position.x.toFixed(1)} Y${camera.position.y.toFixed(1)} Z${camera.position.z.toFixed(1)}`;frames=0;fpsAt=now;$('latency').textContent=`输入→rAF提交 ${inputFrameMs.toFixed(1)}ms · 相机RTT ${cameraRttMs.toFixed(0)}ms · 渲染倍率 ${renderScale.toFixed(2)}`;if(now-scaleAt>2500&&[...cache.values()].length){const next=fps<27?Math.max(.45,renderScale-.1):fps>40?Math.min(1,renderScale+.05):renderScale;if(next!==renderScale){renderScale=next;resize();event('render_scale',{scale:renderScale,geometry_unchanged:true});}scaleAt=now;}}if(now-miniAt>100){mini();miniAt=now;}}
document.addEventListener('visibilitychange',()=>{last=performance.now();fpsAt=last;frames=0;keys.clear();event('visibility',{hidden:document.hidden});});
addEventListener('keydown',e=>{if(['SELECT','INPUT','TEXTAREA'].includes(document.activeElement.tagName))return;const k=e.key.toLowerCase();if(['w','a','s','d','q','e','shift','arrowup','arrowdown','arrowleft','arrowright'].includes(k)){keys.add(k);inputAt=performance.now();e.preventDefault();if(!e.repeat)event('key_down',{key:k,position:camera.position.toArray()});}});addEventListener('keyup',e=>{keys.delete(e.key.toLowerCase());event('key_up',{key:e.key,position:camera.position.toArray()});});addEventListener('blur',()=>keys.clear());
renderer.domElement.addEventListener('pointerdown',e=>{renderer.domElement.focus();if(mode==='walk'){drag={x:e.clientX,y:e.clientY};renderer.domElement.setPointerCapture(e.pointerId);}});renderer.domElement.addEventListener('pointermove',e=>{if(!drag||mode!=='walk')return;inputAt=performance.now();yaw-=(e.clientX-drag.x)*.004;pitch=THREE.MathUtils.clamp(pitch-(e.clientY-drag.y)*.004,-1.45,1.45);drag={x:e.clientX,y:e.clientY};look();noteNavigation();});renderer.domElement.addEventListener('pointerup',()=>{drag=null;event('look',{quaternion:camera.quaternion.toArray()});});renderer.domElement.addEventListener('pointercancel',()=>drag=null);
$('distance').onchange=()=>{camera.far=Number($('distance').value);camera.updateProjectionMatrix();event('draw_distance',{value:camera.far,geometry_unchanged:true});};
$('home').onclick=()=>userAction(home);
$('top').onclick=()=>userAction(top);
$('walk').onclick=()=>userAction(()=>{setWalk();keys.clear();event('walk');});
$('stop').onclick=()=>{keys.clear();drag=null;userOrbit=false;event('stop',{position:camera.position.toArray()});renderer.domElement.focus();};
$('back').onclick=()=>userAction(()=>{setWalk();yaw+=Math.PI;look();event('turn_back');});
$('step').onclick=()=>userAction(()=>{setWalk();camera.getWorldDirection(dir);camera.position.addScaledVector(dir,10);event('step',{distance:10,position:camera.position.toArray()});});
$('side').onclick=()=>userAction(()=>{setWalk();camera.position.x+=Math.cos(yaw)*3;camera.position.z-=Math.sin(yaw)*3;event('side',{position:camera.position.toArray()});});
$('metrics').onclick=()=>{telemetry();const blob=new Blob([JSON.stringify({...metrics,current:snapshot()},null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='v26_browser_metrics.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
$('hardware').textContent=/Basic Render|SwiftShader|llvmpipe/i.test(metrics.renderer)?'当前为软件渲染：大资产帧率受限，不代表 A800 生成速度。':'当前 WebGL 渲染器：'+metrics.renderer;
$('frontier').onclick=()=>userAction(()=>{keys.clear();aim([-.7,2.73,-160],[-.7,2.73,-260]);event('cached_frontier',{teleport_within_visited_cache:true});});
home();navigationPose=poseKey();vis();requestAnimationFrame(frame);void ensureNative();void poll();
