// CityTopia mode: reuse World Lab canvas, navigation controls and styles.
import * as T from 'three';
const $=id=>document.getElementById(id),canvas=$('render'),viewport=$('viewport');
const renderer=new T.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.outputColorSpace=T.SRGBColorSpace;
renderer.setClearColor('#9bacb2');renderer.toneMapping=T.NoToneMapping;
const scene=new T.Scene(),camera=new T.PerspectiveCamera(60,16/9,.04,600);
let yaw=0,pitch=0,mode='walk',drag=null,last=performance.now(),dirty=true,manifest,currentJob=null,lastResult=null,selected=[],selectionTime=0,autoDue=0;
let statusTimer=null,statusBusy=false,submittedAt=0,displayedAt=0,nextNativeAt=0,moveFrames=0,fpsSince=performance.now(),previewFPS=0;
const keys=new Set(),cache=new Map(),loading=new Map(),loader=new T.TextureLoader();
const image=new Image();image.id='native-image';image.hidden=true;viewport.insertBefore(image,$('labels'));
$('crosshair').hidden=false;$('move-pad').hidden=false;
$('labels').innerHTML='<span id="view-label">Preparing</span>';
$('map-source').textContent='CityDreamer4D · CityTopia weights · Existing Map10 conditions and fixed building styles. Street defects remain visible. This scene uses neither Bistro nor Sat3DGen.';
$('case-note').textContent='Fast navigation projects saved native RGB with coarse depth into 3D; holes and seams may appear. Use it to choose a camera. Press R for A800 native neural rendering at that camera (960×540, no sharpening/SR). The navigation preview does not establish final generation quality.';
$('camera-bar').innerHTML='<button id="walk" class="primary">▶ Walk WASD</button><button id="fly">Free flight Q/E</button><button id="native" class="primary">R · Render current native view</button><button id="preview">Return to fast preview</button><button id="last-result" disabled>Latest native result</button><select id="route-view" aria-label="Fixed native cameras"></select><button id="back">180° Turn around</button><button id="save-view">Save camera JSON</button><button id="shot">Screenshot PNG</button>';
$('camera-bar').insertAdjacentHTML('beforeend','<label><input id="auto-native" type="checkbox" checked>Movement triggers four GPUs · Latest camera first</label>');
$('hint').textContent='WASD to move · Drag to turn · Shift to accelerate · R for native render · Esc to stop';

function toWorld(p){return new T.Vector3((p.y-1536)/4,(p.z-14)/4,-(p.x-1536)/4);}
function toNative(v){return {x:1536-4*v.z,y:1536+4*v.x,z:14+4*v.y};}
function pose(){return {cam_position:toNative(camera.position),cam_look_at:toNative(camera.position.clone().add(camera.getWorldDirection(new T.Vector3()).multiplyScalar(10)))};}
function setPose(p){camera.position.copy(toWorld(p.cam_position));const d=toWorld(p.cam_look_at).sub(camera.position).normalize();yaw=Math.atan2(-d.x,-d.z);pitch=Math.asin(d.y);rotate();dirty=true;}
function rotate(){camera.rotation.order='YXZ';camera.rotation.set(pitch,yaw,0);dirty=true;}
function preview(){image.hidden=true;$('view-label').textContent=`Fast RGB-D navigation · ${previewFPS} FPS · Not the native neural frame rate`;dirty=true;}
function showNative(url,p,label){setPose(p);const expected=JSON.stringify(pose());image.onload=()=>{if(expected!==JSON.stringify(pose()))return;image.hidden=false;displayedAt=performance.now();$('view-label').textContent=label;canvas.dataset.nativeDisplayedAt=displayedAt;};image.src=url;}
function changed(t){if(!autoDue)autoDue=t+120;sessionStorage.setItem('citytopia-live-pose',JSON.stringify(pose()));}
function fail(e){$('error').hidden=false;$('error').textContent=String(e);}
async function fetchJSON(url,body){const r=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const d=await r.json();if(!r.ok)throw Error(d.error||r.status);return d;}
function download(data,name,type){const url=URL.createObjectURL(new Blob([data],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}
function savePose(){const data={model:'CityTopia Map10',pose:pose(),mode:image.hidden?'RGB-D preview':'native RGB',image:image.hidden?null:image.src,time:new Date().toISOString()};localStorage.setItem('citytopia-last-view',JSON.stringify(data));download(JSON.stringify(data,null,2),'citytopia-camera.json','application/json');}
function screenshot(){const c=document.createElement('canvas');c.width=960;c.height=540;const ctx=c.getContext('2d');ctx.drawImage(image.hidden?canvas:image,0,0,960,540);const a=document.createElement('a');a.download=`citytopia-${image.hidden?'preview':'native'}-${Date.now()}.png`;a.href=c.toDataURL('image/png');a.click();}
async function mesh(c){
 if(cache.has(c.index))return cache.get(c.index);if(loading.has(c.index))return loading.get(c.index);
 const promise=(async()=>{const response=await fetch(c.mesh);if(!response.ok)throw Error('Preview loading failed');const data=await response.arrayBuffer(),g=new T.BufferGeometry(),n=c.vertices;
  g.setAttribute('position',new T.BufferAttribute(new Float32Array(data,0,n*3),3));g.setAttribute('uv',new T.BufferAttribute(new Float32Array(data,n*12,n*2),2));g.setIndex(new T.BufferAttribute(new Uint32Array(data,n*20,c.indices),1));
  const texture=await loader.loadAsync(c.rgb);texture.colorSpace=T.SRGBColorSpace;texture.anisotropy=Math.min(4,renderer.capabilities.getMaxAnisotropy());
  const m=new T.Mesh(g,new T.MeshBasicMaterial({map:texture,side:T.DoubleSide}));m.visible=false;scene.add(m);cache.set(c.index,m);dirty=true;return m;})();
 loading.set(c.index,promise);try{return await promise;}finally{loading.delete(c.index);}
}
function selectPreview(){
 if(!manifest)return;const dir=camera.getWorldDirection(new T.Vector3());
 const ranked=manifest.cameras.map(c=>{const p=toWorld(c.pose.cam_position),d=toWorld(c.pose.cam_look_at).sub(p).normalize();return {c,score:p.distanceTo(camera.position)+12*(1-d.dot(dir))};}).sort((a,b)=>a.score-b.score);
 selected=ranked.slice(0,3).map(v=>v.c.index);
 for(const [id,m] of cache)m.visible=selected.includes(id);
 ranked.slice(0,5).forEach(({c},rank)=>mesh(c).then(m=>{m.visible=selected.includes(c.index);m.material.polygonOffset=true;m.material.polygonOffsetFactor=rank;m.material.polygonOffsetUnits=rank;dirty=true;}).catch(fail));
 if(cache.size>12){for(const [id,m] of cache)if(!ranked.slice(0,8).some(v=>v.c.index===id)){scene.remove(m);m.geometry.dispose();m.material.map.dispose();m.material.dispose();cache.delete(id);}}
}
async function requestNative(){
 try{if(currentJob||performance.now()<nextNativeAt)return;autoDue=0;currentJob='sending';submittedAt=performance.now();const p=pose(),job=await fetchJSON('/api/render',{pose:p});currentJob=job.id;$('native').disabled=true;$('error').hidden=true;$('health').textContent='A800 GPU0–3 dynamic tiles / Native rendering; navigation remains available';clearTimeout(statusTimer);status();}
 catch(e){currentJob=null;$('native').disabled=false;autoDue=0;nextNativeAt=performance.now()+3000;fail(e);}
}
async function status(){
 if(statusBusy)return;statusBusy=true;clearTimeout(statusTimer);
 try{const s=await fetchJSON('/api/status');const j=currentJob&&s.jobs[currentJob];
  const gpu=s.state.gpus?.join('/')??s.state.gpu??'?';
  $('health').textContent=j?`GPU ${gpu} · ${j.status} · ${j.tiles_done?.length??0}/${j.tiles_total??4} dynamic tiles · Navigation remains available`:`GPU ${gpu} · ${s.state.phase}`;
  if(j?.status==='done'){currentJob=null;$('native').disabled=false;lastResult=j;$('last-result').disabled=false;$('last-result').textContent=`View native result (${j.seconds.toFixed(1)}s)`;
   const now=pose(),same=['cam_position','cam_look_at'].every(k=>['x','y','z'].every(a=>Math.abs(now[k][a]-j.pose[k][a])<.02));
   canvas.dataset.nativeSeconds=j.seconds;canvas.dataset.nativeJob=j.id;
   if(same){autoDue=0;showNative(j.url,j.pose,`Four-GPU native 960×540 · ${(j.render_seconds??j.seconds).toFixed(2)}s compute · No image enhancement`);}
  }else if(j?.status==='error'){currentJob=null;$('native').disabled=false;nextNativeAt=performance.now()+3000;fail(j.error);}
  else if(currentJob&&currentJob!=='sending'&&!j){currentJob=null;$('native').disabled=false;autoDue=performance.now()+500;}
 }catch(e){$('health').textContent='Native service unavailable; loaded previews remain navigable';}
 finally{statusBusy=false;statusTimer=setTimeout(status,currentJob?150:3000);}
}
$('walk').onclick=()=>{mode='walk';canvas.focus();};$('fly').onclick=()=>{mode='fly';canvas.focus();};
$('native').onclick=requestNative;$('preview').onclick=preview;$('save-view').onclick=savePose;$('shot').onclick=screenshot;
$('last-result').onclick=()=>{if(lastResult)showNative(lastResult.url,lastResult.pose,`A800 native result · ${lastResult.seconds.toFixed(1)}s`);};
$('back').onclick=()=>{preview();yaw+=Math.PI;rotate();selectPreview();changed(performance.now());};
$('route-view').onchange=()=>{const c=manifest.cameras.find(c=>c.index===+$('route-view').value);if(c)showNative(c.rgb,c.pose,'Saved CityTopia native render · No new inference');};
canvas.onpointerdown=e=>{canvas.focus();drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!drag)return;preview();yaw-=(e.clientX-drag.x)*.003;pitch=T.MathUtils.clamp(pitch-(e.clientY-drag.y)*.003,-1.3,1.3);drag={x:e.clientX,y:e.clientY};rotate();changed(performance.now());};
canvas.onpointerup=()=>drag=null;canvas.onpointercancel=()=>drag=null;
addEventListener('keydown',e=>{if(e.target.closest('input,textarea,select'))return;if(e.code==='KeyR'){e.preventDefault();requestNative();return;}if(e.code==='Escape'){keys.clear();drag=null;return;}if(['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ArrowLeft','ArrowRight','ShiftLeft'].includes(e.code)){keys.add(e.code);e.preventDefault();}});
addEventListener('keyup',e=>keys.delete(e.code));addEventListener('blur',()=>{keys.clear();drag=null;});
document.querySelectorAll('[data-key]').forEach(b=>{b.onpointerdown=e=>{keys.add(b.dataset.key);b.setPointerCapture(e.pointerId);};b.onpointerup=()=>keys.delete(b.dataset.key);b.onpointercancel=()=>keys.clear();});
function frame(t){requestAnimationFrame(frame);const dt=Math.min(.05,(t-last)/1000);last=t;
 if(keys.size){preview();const speed=dt*(keys.has('ShiftLeft')?8:2.5),v=new T.Vector3(Number(keys.has('KeyD'))-Number(keys.has('KeyA')),0,Number(keys.has('KeyS'))-Number(keys.has('KeyW')));if(v.lengthSq())v.normalize().applyAxisAngle(new T.Vector3(0,1,0),yaw).multiplyScalar(speed);
  if(mode==='fly')v.y=(Number(keys.has('KeyE'))-Number(keys.has('KeyQ')))*speed;
  camera.position.add(v);camera.position.x=T.MathUtils.clamp(camera.position.x,-220,220);camera.position.z=T.MathUtils.clamp(camera.position.z,-220,220);camera.position.y=T.MathUtils.clamp(camera.position.y,.3,100);
  yaw+=(Number(keys.has('ArrowLeft'))-Number(keys.has('ArrowRight')))*dt;rotate();changed(t);}
 if(autoDue&&t>autoDue&&!currentJob&&$('auto-native').checked&&!document.hidden)requestNative();
 if(t-selectionTime>250&&dirty){selectPreview();selectionTime=t;}
 if(dirty){renderer.render(scene,camera);moveFrames++;canvas.dataset.pose=JSON.stringify(pose());dirty=false;}
 if(t-fpsSince>1000){previewFPS=Math.round(moveFrames*1000/(t-fpsSince));canvas.dataset.previewFps=previewFPS;moveFrames=0;fpsSince=t;}
}
new ResizeObserver(()=>{const w=viewport.clientWidth,h=viewport.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();dirty=true;}).observe(viewport);
async function boot(){manifest=await fetchJSON('/preview/manifest.json');for(const c of manifest.cameras){const o=document.createElement('option');o.value=c.index;o.textContent=`${c.index} · ${c.tag}`;$('route-view').append(o);}const c=manifest.cameras.find(c=>c.index===manifest.default);$('route-view').value=c.index;setPose(c.pose);await mesh(c);selectPreview();$('loading').hidden=true;const saved=sessionStorage.getItem('citytopia-live-pose');if(saved){try{setPose(JSON.parse(saved));preview();selectPreview();}catch{showNative(c.rgb,c.pose,'Saved CityTopia native render');}}else showNative(c.rgb,c.pose,'Saved CityTopia native render · Movement triggers four GPUs');canvas.focus();status();}
boot().catch(fail);requestAnimationFrame(frame);
