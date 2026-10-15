import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';

const $=id=>document.getElementById(id),vp=$('viewport'),boot=performance.now(),session=crypto.randomUUID();
const renderer=new THREE.WebGLRenderer({antialias:false,powerPreference:'high-performance'});
// Display-only clear sky; never recolor the generated asset to hide errors.
renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.setClearColor('#a9d4ee');
vp.appendChild(renderer.domElement);renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label','Map10 可导航三维视口');
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(65,1,.03,800),controls=new OrbitControls(camera,renderer.domElement);
// Native preview exports contain display base-color bytes (see v17 export).
// Decode those bytes once before the renderer's sRGB output, without changing buffers.
const displayDecode={value:1};
function assetMaterial(hasColor){
 const material=new THREE.MeshBasicMaterial({vertexColors:hasColor,side:THREE.DoubleSide,color:0xffffff});
 material.onBeforeCompile=shader=>{shader.uniforms.nativeDisplayDecode=displayDecode;shader.vertexShader='uniform float nativeDisplayDecode;\n'+shader.vertexShader;
  shader.vertexShader=shader.vertexShader.replace('#include <color_vertex>',`#include <color_vertex>
  #if defined(USE_COLOR) || defined(USE_COLOR_ALPHA)
    if(nativeDisplayDecode > 0.5){
      vec3 c=max(vColor.rgb,vec3(0.0));
      vColor.rgb=mix(pow((c+0.055)/1.055,vec3(2.4)),c/12.92,lessThanEqual(c,vec3(0.04045)));
    }
  #endif`);
 };
 material.customProgramCacheKey=()=> 'native-display-basecolor-v1';return material;
}
const sky=new THREE.Mesh(new THREE.SphereGeometry(350,16,12),new THREE.ShaderMaterial({
 side:THREE.BackSide,depthWrite:false,depthTest:false,
 uniforms:{zenith:{value:new THREE.Color('#74b9e5')},horizon:{value:new THREE.Color('#e3edf1')},nadir:{value:new THREE.Color('#c8c4b9')}},
 vertexShader:'varying vec3 skyDirection; void main(){skyDirection=position; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
 fragmentShader:`uniform vec3 zenith; uniform vec3 horizon; uniform vec3 nadir; varying vec3 skyDirection;
 void main(){float h=normalize(skyDirection).y; vec3 color=h>=0.0?mix(horizon,zenith,smoothstep(0.0,0.85,h)):mix(horizon,nadir,smoothstep(0.0,0.4,-h));gl_FragColor=vec4(color,1.0);
 #include <colorspace_fragment>
 }`
}));sky.renderOrder=-1000;sky.frustumCulled=false;scene.add(sky);
controls.enableDamping=false;controls.enabled=false;
const groups={base:new THREE.Group(),placeholders:new THREE.Group(),accepted:new THREE.Group(),quarantine:new THREE.Group(),outlines:new THREE.Group()};
Object.values(groups).forEach(g=>scene.add(g));
const worker=new Worker('./mesh_worker.js'),pending=new Map(),cache=new Map(),events=[],keys=new Set(),dir=new THREE.Vector3();
const gl=renderer.getContext(),gpu=gl.getExtension('WEBGL_debug_renderer_info');
const rendererName=String(gpu?gl.getParameter(gpu.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER));
const software=/Basic Render|SwiftShader|llvmpipe|software|softpipe|lavapipe|WARP/i.test(rendererName);
let world='map',mode='walk',yaw=0,pitch=0,drag=null,map,api,acceptedHash=null,acceptedLoading=null,candidate=null,viewRevision=0;
let renderScale=software?.7:1,last=performance.now(),fpsAt=last,frames=0,fps=0,miniAt=0,scaleAt=0,maxFrameMs=0;
let firstMapMs=null,initialNavigationMs=null,firstNavigationMs=null,navigationRevision=0,revisitSameHash=0,cameraSending=false,telemetrySending=false,cameraPosts=0,cameraRttMs=null;
let lastPose=null,mapPose=null,scaffoldLoading=false;
const generationMovement={distance:0,rotationRadians:0,frames:0};
const metrics={session,bootEpoch:Date.now(),renderer:rendererName,software,loads:{},events,limitations:['rAF submission FPS; not GPU timing','All original triangles; no geometry simplification','Map height placeholders; free camera without collision constraints']};
const statusNames={waiting_camera:'等待地图内移动',queued:'排队中',generating:'生成中',exporting:'导出中',transferring:'传输中',evaluating:'验收中',validating:'验收中',accepted:'已验收',rejected:'未通过',quarantine:'隔离',quarantined:'隔离',failed:'失败',ready:'就绪',running:'运行中',completed:'已结束',all_failed:'全部失败'};
const label=s=>statusNames[String(s).toLowerCase()]||s||'—';
const brief=v=>v==null?'—':typeof v==='string'?v:JSON.stringify(v);
const shaOf=a=>typeof a?.sha256==='string'&&/^[a-f0-9]{64}$/i.test(a.sha256)?a.sha256.toLowerCase():null;
function event(type,data={}){events.push({type,t:performance.now()-boot,epoch:Date.now(),world,...data});if(events.length>600)events.shift();}
function error(e){$('error').hidden=false;$('error').textContent=String(e);event('error',{message:String(e)});}
addEventListener('error',e=>error(e.message));addEventListener('unhandledrejection',e=>error(e.reason));
function resize(){const w=Math.max(1,vp.clientWidth),h=Math.max(1,vp.clientHeight);renderer.setPixelRatio(renderScale);renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();$('resolution').textContent=`${renderScale<.999?'低分辨率显示':'原尺寸显示'} · 倍率 ${renderScale.toFixed(2)} · 实际 ${renderer.domElement.width}×${renderer.domElement.height} · 几何完整`;$('resolution').dataset.low=String(renderScale<.999);}
new ResizeObserver(resize).observe(vp);resize();
function look(){camera.rotation.order='YXZ';camera.rotation.set(pitch,yaw,0);camera.updateMatrixWorld();}
function setWalk(){mode='walk';controls.enabled=false;camera.getWorldDirection(dir);yaw=Math.atan2(-dir.x,-dir.z);pitch=Math.asin(THREE.MathUtils.clamp(dir.y,-1,1));look();}
function aim(p,target){camera.position.fromArray(p);camera.lookAt(new THREE.Vector3(...target));setWalk();}
function home(){keys.clear();drag=null;aim(map?.spawn?.position||[35,1.7,35],map?.spawn?.look_at||[0,1.7,0]);}
function top(){keys.clear();drag=null;mode='orbit';controls.enabled=true;camera.position.set(0,135,1);controls.target.set(0,0,0);controls.update();}
function pose(){return {p:camera.position.clone(),q:camera.quaternion.clone()};}
function generating(){return [api?.status,api?.cycle?.status,...(api?.cycle?.attempts||[]).map(a=>a.status)].some(s=>/^(queued|generating|exporting|transferring|evaluating|validating|running)$/i.test(s||''));}
function observeMovement(user=false){const now=pose();if(lastPose){const distance=lastPose.p.distanceTo(now.p),angle=lastPose.q.angleTo(now.q);if(distance>1e-8||angle>1e-7){if(user){navigationRevision++;if(firstNavigationMs===null){firstNavigationMs=performance.now()-boot;event('first_navigation',{ms:firstNavigationMs});}if(generating()){generationMovement.distance+=distance;generationMovement.rotationRadians+=angle;generationMovement.frames++;}}}}lastPose=now;}
function action(fn){fn();observeMovement(true);renderer.domElement.focus();event('navigation',{position:camera.position.toArray(),navigationRevision});}
controls.addEventListener('change',()=>observeMovement(true));
function vis(){const quarantine=world==='quarantine';groups.base.visible=!quarantine;groups.placeholders.visible=!quarantine&&!acceptedHash;groups.accepted.visible=!quarantine&&!!acceptedHash;groups.quarantine.visible=quarantine;groups.outlines.visible=quarantine&&$('outlines').checked;
 $('world-label').dataset.quarantine=String(quarantine);$('world-label').textContent=quarantine?`隔离检查 · ${candidate?.id||'候选'} · 未作为 ACCEPTED 安装`:acceptedHash?'MAP10 · 服务器 ACCEPTED 资产':'MAP10 · 全部建筑仍为占位';
 $('note').textContent=quarantine?'当前是独立候选检查层，未安装到地图。橙色为原地图轮廓；资产保持原始世界坐标。':acceptedHash?'服务器已验收资产在原始世界坐标显示。':'Map10 道路与建筑轮廓脚手架；全部灰色建筑为占位，高度未知。';
 $('legend').textContent=quarantine?'隔离候选 · 未验收安装 · 橙色线为地图占位轮廓':acceptedHash?'ACCEPTED 资产 · 原始世界坐标 · 橙色箭头为真实相机':'灰色建筑：地图占位 · 橙色箭头：真实相机';}
function returnMap(){viewRevision++;if(world!=='map'){world='map';candidate=null;groups.quarantine.clear();if(mapPose){camera.position.copy(mapPose.p);camera.quaternion.copy(mapPose.q);controls.target.copy(mapPose.target);if(mapPose.mode==='walk')setWalk();else{mode='orbit';controls.enabled=true;controls.update();}}else home();observeMovement(false);}keys.clear();vis();event('return_map',{sha256:acceptedHash});if(acceptedHash)revisit(acceptedHash,'return_map');}
function polygon(poly,y,color,height=0){const shape=new THREE.Shape(poly.map(p=>new THREE.Vector2(p[0],-p[1])));const geometry=height?new THREE.ExtrudeGeometry(shape,{depth:height,bevelEnabled:false,steps:1}):new THREE.ShapeGeometry(shape);geometry.rotateX(-Math.PI/2);const m=new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({color,side:THREE.DoubleSide}));m.position.y=y;return m;}
async function scaffold(){if(scaffoldLoading||map)return;scaffoldLoading=true;try{const r=await fetch('/map10_scaffold.json');if(!r.ok)throw Error(`脚手架 HTTP ${r.status}`);map=await r.json();
 const span=map.coordinate_system?.world_span||100,ground=new THREE.Mesh(new THREE.PlaneGeometry(span,span),new THREE.MeshBasicMaterial({color:0xb3afa5,side:THREE.DoubleSide}));ground.rotation.x=-Math.PI/2;ground.userData.source='map_placeholder_display_only';groups.base.add(ground);
 for(const road of map.roads||[])groups.base.add(polygon(road.polygon,.02,0x52565a));
 for(const b of map.buildings||[]){const m=polygon(b.polygon,0,0xa8b5bc,b.height);m.userData={id:b.id,source:'map_placeholder'};groups.placeholders.add(m);const edges=new THREE.EdgesGeometry(m.geometry),outline=new THREE.LineSegments(edges,new THREE.LineBasicMaterial({color:0xffb367,depthTest:false}));outline.renderOrder=10;groups.outlines.add(outline);m.add(new THREE.LineSegments(edges,new THREE.LineBasicMaterial({color:0x415762})));}
 const grid=new THREE.GridHelper(span,4,0x90918c,0xa5a69f);grid.position.y=.03;grid.visible=$('display-grid').checked;grid.name='map-display-grid';groups.base.add(grid);
 for(const b of map.buildings||[]){const li=document.createElement('li');li.textContent=`${b.id} · 地图轮廓 / 占位高度 ${b.height}`;$('objects').appendChild(li);}
 if(navigationRevision===0&&world==='map')home();observeMovement(false);firstMapMs=performance.now()-boot;event('scaffold_ready',{ms:firstMapMs,objects:map.buildings.length});vis();
 }catch(e){error(e);setTimeout(scaffold,3000);}finally{scaffoldLoading=false;}}
function revisit(sha,reason){revisitSameHash++;event('revisit_same_hash',{sha256:sha,reason,count:revisitSameHash});}
function load(a){const sha=shaOf(a);if(!sha||!a.asset_url)return Promise.reject(Error('资产缺少有效 SHA256 或 URL'));
 if(cache.has(sha)){revisit(sha,'cache_hit');return Promise.resolve(cache.get(sha));}if(pending.has(sha))return pending.get(sha).promise;
 let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});const job={id:a.id,promise,resolve,reject,start:performance.now()};pending.set(sha,job);
 job.timeout=setTimeout(()=>{if(pending.get(sha)===job){pending.delete(sha);reject(Error('资产加载超时，地图仍可浏览'));}},90000);
 $('loading').textContent=`后台读取 ${a.id} · SHA ${sha.slice(0,12)}，可继续移动`;event('load_start',{id:a.id,sha256:sha});
 worker.postMessage({id:sha,url:new URL(a.asset_url,location.origin).href,sha256:sha});return promise;}
worker.onmessage=({data:d})=>{const job=pending.get(d.id);if(!job)return;clearTimeout(job.timeout);pending.delete(d.id);if(!d.ok){job.reject(Error(d.error));return;}try{
 if(d.sha256!==d.id||!d.all_faces_preserved)throw Error('SHA / exact geometry 验证失败');
 const root=new THREE.Group();let triangles=0,vertices=0;
 const meshes=d.meshes.map(prims=>prims.flatMap(p=>{const attrs={};for(const [name,a] of [['position',p.position],['color',p.color],['normal',p.normal]])if(a)attrs[name]=new THREE.BufferAttribute(a.array,a.itemSize,a.normalized);const index=new THREE.BufferAttribute(p.index.array,1);vertices+=p.position.array.length/3;triangles+=p.index.array.length/3;
 return p.batches.map(b=>{const g=new THREE.BufferGeometry();for(const [name,a] of Object.entries(attrs))g.setAttribute(name,a);g.setIndex(index);g.setDrawRange(b.start,b.count);g.boundingBox=new THREE.Box3(new THREE.Vector3(...b.min),new THREE.Vector3(...b.max));g.boundingSphere=g.boundingBox.getBoundingSphere(new THREE.Sphere());return g;});}));
 function node(i){const n=d.nodes[i],o=new THREE.Group();if(n.matrix){o.matrix.fromArray(n.matrix);o.matrixAutoUpdate=false;}else{if(n.translation)o.position.fromArray(n.translation);if(n.rotation)o.quaternion.fromArray(n.rotation);if(n.scale)o.scale.fromArray(n.scale);}if(n.mesh!==undefined)for(const g of meshes[n.mesh])o.add(new THREE.Mesh(g,assetMaterial(!!g.attributes.color)));for(const child of n.children||[])o.add(node(child));return o;}
 d.roots.forEach(i=>root.add(node(i)));const record={id:job.id,sha256:d.sha256,root,triangles,vertices,bytes:d.bytes,loadMs:performance.now()-job.start,workerMs:d.workerMs};cache.set(d.sha256,record);metrics.loads[d.sha256]={...record,root:undefined};event('load_complete',{id:job.id,sha256:d.sha256,triangles,ms:record.loadMs});$('loading').textContent=`${job.id} · ${(record.loadMs/1000).toFixed(2)}s · ${(triangles/1e6).toFixed(2)}M 三角形 · SHA 已校验`;$('cache').textContent=[...cache.keys()].map(s=>s.slice(0,16)).join(' / ');job.resolve(record);
 }catch(e){job.reject(e);}};
worker.onerror=e=>{for(const job of pending.values()){clearTimeout(job.timeout);job.reject(Error(e.message||'Mesh worker failed'));}pending.clear();error(e.message||'Mesh worker failed');};
async function ensureAccepted(){const a=api?.cycle?.accepted,sha=shaOf(a);if(!a||!sha||!a.asset_url){if(acceptedHash){groups.accepted.clear();acceptedHash=null;event('accepted_withdrawn');}vis();return;}
 if(acceptedHash&&acceptedHash!==sha){groups.accepted.clear();acceptedHash=null;vis();}
 if(acceptedHash===sha||acceptedLoading===sha)return;acceptedLoading=sha;
 try{const rec=await load(a);if(shaOf(api?.cycle?.accepted)!==sha)return;groups.accepted.clear();groups.accepted.add(rec.root.clone(true));acceptedHash=sha;event('accepted_installed',{id:a.id,sha256:sha,camera:camera.position.toArray(),world_frame_unchanged:true});vis();showState();}
 catch(e){error(e);}finally{if(acceptedLoading===sha)acceptedLoading=null;}}
async function inspect(index){const a=api?.cycle?.attempts?.find(a=>a.id===`attempt-${index}`);if(!shaOf(a)||!a.asset_url)return;
 const revision=++viewRevision;keys.clear();event('candidate_requested',{id:a.id,sha256:shaOf(a)});
 try{const rec=await load(a);if(revision!==viewRevision)return;if(world==='map')mapPose={...pose(),mode,target:controls.target.clone()};world='quarantine';candidate={id:a.id,sha256:rec.sha256};groups.quarantine.clear();groups.quarantine.add(rec.root.clone(true));keys.clear();vis();event('candidate_inspection',{...candidate,installed:false});}
 catch(e){error(e);}}
function reportField(report,key,depth=0){if(!report||typeof report!=='object'||depth>6)return undefined;if(Object.hasOwn(report,key))return report[key];for(const value of Object.values(report)){const found=reportField(value,key,depth+1);if(found!==undefined)return found;}return undefined;}
function reportSummary(report){if(!report)return '等待独立报告';const value=k=>brief(reportField(report,k));return `决策 ${value('decision')} · 理由 ${brief(reportField(report,'reasons')??reportField(report,'reason'))} · road_surface_near_display_ground_fraction=${value('road_surface_near_display_ground_fraction')} · road_air_column_obstruction_fraction=${value('road_air_column_obstruction_fraction')}`;}
function showState(){const c=api?.cycle||{},attempts=c.attempts||[],budget=api?.budget||{};
 $('job').textContent=`${c.id||'map10-0'} · ${label(c.status)} · 服务 ${label(api?.status)}`;
 $('worker').textContent='Worker：'+brief(api?.worker?.stage||api?.worker?.status||api?.worker);
 $('budget').textContent=`预算 ${budget.used??budget.generations_used??'—'} / ${budget.max_generations??budget.limit??budget.max??'—'}`;
 $('message').textContent=brief(c.message||api?.message||'可继续浏览，等待服务器状态');$('feedback').textContent='针对性反馈：'+brief(c.feedback);
 for(let i=0;i<2;i++){const a=attempts.find(a=>a.id===`attempt-${i}`);$(`candidate-${i}`).disabled=!shaOf(a)||!a?.asset_url;$(`attempt-${i}`).textContent=`attempt-${i}：${label(a?.status)}${Number.isFinite(a?.seconds)?` · ${a.seconds.toFixed(1)}s`:''}${shaOf(a)?' · '+shaOf(a).slice(0,12):''} · ${reportSummary(a?.report)}`;}
 $('acceptance').textContent=acceptedHash?`ACCEPTED 已安装 · SHA ${acceptedHash.slice(0,16)}`:c.accepted?'服务器已标记 ACCEPTED，等待资产校验与载入；当前仍为占位。':'尚无 ACCEPTED：地图建筑全部保留占位。';
 $('reports').textContent=JSON.stringify(attempts.map(a=>({id:a.id,status:a.status,report:a.report,seconds:a.seconds})),null,2);$('counters').textContent=`尝试 ${attempts.length} / 缓存 ${cache.size} / 重访同 SHA ${revisitSameHash} / 相机上报 ${cameraPosts}`;}
async function poll(){try{const r=await fetch('/api/state',{cache:'no-store',signal:AbortSignal.timeout(5000)});if(!r.ok)throw Error(`HTTP ${r.status}`);api=await r.json();showState();$('connection').textContent='API 已连接 · 状态 1s / 相机 200ms / 遥测 3s';void ensureAccepted();}catch(e){$('connection').textContent='状态连接中断：'+e.message+'；已有地图仍可导航';}finally{setTimeout(poll,1000);}}
function cameraPayload(){camera.getWorldDirection(dir);return {session,epoch:Date.now(),world,mode,position:camera.position.toArray(),forward:dir.toArray(),quaternion:camera.quaternion.toArray(),event:'camera',navigation_revision:navigationRevision,user_moved:navigationRevision>0,fps,loaded:[...cache.values()].map(r=>({id:r.id,sha256:r.sha256,installed:r.sha256===acceptedHash,layer:r.sha256===acceptedHash?'accepted':'quarantine-cache'})),accepted_sha256:acceptedHash,candidate_sha256:candidate?.sha256||null};}
async function reportCamera(){if(cameraSending||!map)return;cameraSending=true;const payload=cameraPayload(),start=performance.now();try{const r=await fetch('/api/camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:AbortSignal.timeout(4000)});if(!r.ok)throw Error(`HTTP ${r.status}`);cameraPosts++;cameraRttMs=performance.now()-start;}catch(e){event('camera_report_error',{message:String(e)});}finally{cameraSending=false;}}
function snapshot(){return {...cameraPayload(),renderer:rendererName,renderScale,renderSize:[renderer.domElement.width,renderer.domElement.height],viewportSize:[vp.clientWidth,vp.clientHeight],firstMapMs,initialNavigationMs,firstNavigationMs,maxFrameMs,cameraRttMs,cameraPosts,cameraMovementDuringGeneration:{...generationMovement},revisitSameHash,visible:!document.hidden,pending:[...pending.keys()],cycle:{id:api?.cycle?.id,status:api?.cycle?.status},budget:api?.budget};}
async function telemetry(){if(telemetrySending)return;telemetrySending=true;try{const r=await fetch('/api/telemetry',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({snapshot:snapshot(),renderer:rendererName,loads:metrics.loads,events:events.slice(-60)}),signal:AbortSignal.timeout(5000)});if(!r.ok)throw Error(`HTTP ${r.status}`);}catch(e){event('telemetry_error',{message:String(e)});}finally{telemetrySending=false;}}
function mini(){const c=$('minimap').getContext('2d'),span=map?.coordinate_system?.world_span||100,xy=(x,z)=>[20+(x/span+.5)*260,35+(z/span+.5)*260];c.fillStyle='#172731';c.fillRect(0,0,300,320);c.lineWidth=1;
 const draw=(poly,fill)=>{c.beginPath();poly.forEach((p,i)=>i?c.lineTo(...xy(...p)):c.moveTo(...xy(...p)));c.closePath();if(fill){c.fillStyle=fill;c.fill();}c.stroke();};
 c.strokeStyle='#8adbb499';c.setLineDash([4,4]);c.strokeRect(20,35,260,260);c.setLineDash([]);
 for(const r of map?.roads||[]){c.strokeStyle='#718995';draw(r.polygon,'#3c515d');}c.font='10px system-ui';
 for(const b of map?.buildings||[]){c.strokeStyle='#ffbd77';draw(b.polygon,world==='map'&&!acceptedHash?'#9dadb5':null);const p=b.polygon.reduce((a,p)=>[a[0]+p[0]/b.polygon.length,a[1]+p[1]/b.polygon.length],[0,0]);c.fillStyle='#fff3de';c.fillText(b.id,...xy(...p));}
 camera.getWorldDirection(dir);const p=xy(camera.position.x,camera.position.z),q=xy(camera.position.x+dir.x*7,camera.position.z+dir.z*7);c.fillStyle='#ff7951';c.beginPath();c.arc(...p,4,0,Math.PI*2);c.fill();c.strokeStyle='#ff7951';c.lineWidth=3;c.beginPath();c.moveTo(...p);c.lineTo(...q);c.stroke();c.font='11px system-ui';c.fillStyle='#d8e8ee';c.fillText(world==='map'?'Map10 对象轮廓 / 范围内移动由服务器判断':'隔离候选 / 地图轮廓参考',12,18);c.fillText(`真实相机 X${camera.position.x.toFixed(1)} Z${camera.position.z.toFixed(1)}`,12,312);}
function frame(now){requestAnimationFrame(frame);const raw=now-last;last=now;if(document.hidden)return;const dt=Math.min(raw/1000,.05);maxFrameMs=Math.max(maxFrameMs,raw);
 if(mode==='walk'){const f=Number(keys.has('w'))-Number(keys.has('s')),r=Number(keys.has('d'))-Number(keys.has('a')),speed=(keys.has('shift')?12:4)*dt/Math.max(1,Math.hypot(f,r));camera.position.x+=(-Math.sin(yaw)*f+Math.cos(yaw)*r)*speed;camera.position.z+=(-Math.cos(yaw)*f-Math.sin(yaw)*r)*speed;camera.position.y+=(Number(keys.has('e'))-Number(keys.has('q')))*(keys.has('shift')?12:4)*dt;yaw+=(Number(keys.has('arrowleft'))-Number(keys.has('arrowright')))*dt;pitch=THREE.MathUtils.clamp(pitch+(Number(keys.has('arrowup'))-Number(keys.has('arrowdown')))*dt,-1.45,1.45);look();observeMovement(true);}else controls.update();sky.position.copy(camera.position);renderer.render(scene,camera);
 if(map&&initialNavigationMs===null){initialNavigationMs=performance.now()-boot;$('init').textContent=`地图可导航 ${Math.round(initialNavigationMs)}ms`;event('initial_navigation_ready',{ms:initialNavigationMs});}
 frames++;if(now-fpsAt>=1000){fps=frames*1000/(now-fpsAt);frames=0;fpsAt=now;$('fps').textContent=`${fps.toFixed(1)} FPS · ${(renderer.info.render.triangles/1e6).toFixed(2)}M drawn tris`;if(now-scaleAt>=2500){const next=fps<27?Math.max(.45,renderScale-.1):fps>40?Math.min(1,renderScale+.05):renderScale;if(Math.abs(next-renderScale)>1e-6){renderScale=next;resize();event('render_scale',{scale:renderScale,renderSize:[renderer.domElement.width,renderer.domElement.height],geometry_unchanged:true});}scaleAt=now;}}
 if(now-miniAt>100){miniAt=now;mini();camera.getWorldDirection(dir);$('pos').textContent=`${world==='map'?'地图':'隔离'} · X ${camera.position.x.toFixed(2)} Y ${camera.position.y.toFixed(2)} Z ${camera.position.z.toFixed(2)} · 方向 ${dir.toArray().map(v=>v.toFixed(2)).join(', ')}`;}}
const movementKeys=new Set(['w','a','s','d','q','e','shift','arrowup','arrowdown','arrowleft','arrowright']);
addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement?.tagName))return;const k=e.key.toLowerCase();if(movementKeys.has(k)){e.preventDefault();if(mode!=='walk')setWalk();keys.add(k);}});
addEventListener('keyup',e=>keys.delete(e.key.toLowerCase()));addEventListener('blur',()=>{keys.clear();drag=null;});
document.addEventListener('visibilitychange',()=>{keys.clear();drag=null;last=performance.now();fpsAt=last;frames=0;event('visibility',{hidden:document.hidden});});
renderer.domElement.addEventListener('pointerdown',e=>{renderer.domElement.focus();if(mode==='walk'){drag={x:e.clientX,y:e.clientY};renderer.domElement.setPointerCapture(e.pointerId);}});
renderer.domElement.addEventListener('pointermove',e=>{if(!drag||mode!=='walk')return;yaw-=(e.clientX-drag.x)*.004;pitch=THREE.MathUtils.clamp(pitch-(e.clientY-drag.y)*.004,-1.45,1.45);drag={x:e.clientX,y:e.clientY};look();observeMovement(true);});
for(const name of ['pointerup','pointercancel','lostpointercapture'])renderer.domElement.addEventListener(name,()=>{drag=null;});
$('map').onclick=()=>action(returnMap);$('home').onclick=()=>action(home);$('top').onclick=()=>action(top);$('walk').onclick=()=>action(setWalk);$('turn').onclick=()=>action(()=>{setWalk();yaw+=Math.PI;look();});$('side').onclick=()=>action(()=>{setWalk();camera.position.x+=Math.cos(yaw)*3;camera.position.z-=Math.sin(yaw)*3;});$('step').onclick=()=>action(()=>{setWalk();camera.position.x-=Math.sin(yaw)*5;camera.position.z-=Math.cos(yaw)*5;});$('stop').onclick=()=>{keys.clear();drag=null;};
$('candidate-0').onclick=()=>void inspect(0);$('candidate-1').onclick=()=>void inspect(1);$('outlines').onchange=vis;
$('display-color').onchange=e=>{displayDecode.value=e.target.value==='native-srgb'?1:0;event('display_color',{mode:e.target.value,geometry_unchanged:true});};
$('display-sky').onchange=e=>{sky.visible=e.target.checked;renderer.setClearColor('#c3ccd2');event('display_sky',{gradient:sky.visible});};
$('display-grid').onchange=e=>{const grid=groups.base.getObjectByName('map-display-grid');if(grid)grid.visible=e.target.checked;};
$('metrics').onclick=()=>{void telemetry();const blob=new Blob([JSON.stringify({...metrics,current:snapshot()},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='v27_browser_metrics.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('hardware').textContent=`${software?'软件 GPU，自适应降低显示分辨率':'WebGL 渲染器'}：${rendererName}`;
home();observeMovement(false);vis();requestAnimationFrame(frame);void scaffold();void poll();setInterval(reportCamera,200);setInterval(telemetry,3000);
