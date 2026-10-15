// Minimal loader for the verified native trimesh GLBs. No topology modification.
self.onmessage=async({data:{id,url,sha256}})=>{const start=performance.now();try{
 const r=await fetch(url);if(!r.ok)throw Error(`HTTP ${r.status}`);const buffer=await r.arrayBuffer();
 const bytes=buffer.byteLength,dv=new DataView(buffer);if(dv.getUint32(0,true)!==0x46546c67||dv.getUint32(4,true)!==2)throw Error('Not GLB2');
 const digest=[...new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))].map(x=>x.toString(16).padStart(2,'0')).join('');if(sha256&&digest!==sha256)throw Error('SHA256 mismatch');
 let doc,bin;for(let p=12;p<bytes;){const n=dv.getUint32(p,true),type=dv.getUint32(p+4,true);if(type===0x4e4f534a)doc=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,p+8,n)));if(type===0x004e4942)bin=p+8;p+=8+n;}
 if(!doc||!bin||doc.extensionsRequired?.length)throw Error('Unsupported GLB layout');
 const C={5121:Uint8Array,5123:Uint16Array,5125:Uint32Array,5126:Float32Array},N={SCALAR:1,VEC2:2,VEC3:3,VEC4:4};
 function acc(i){const a=doc.accessors[i],v=doc.bufferViews[a.bufferView],T=C[a.componentType],item=N[a.type];if(!T||!item||a.sparse||v.byteStride||v.buffer!==0)throw Error('Unsupported accessor');return {array:new T(buffer,bin+(v.byteOffset||0)+(a.byteOffset||0),a.count*item),itemSize:item,normalized:!!a.normalized};}
 const transfer=[buffer];
 // Spatial draw batches only: every original triangle appears exactly once.
 // Preserve all original vertex positions/colors, no LOD/decimation/repair.
 function partition(p){const pos=p.position.array,idx=p.index.array,n=idx.length/3,counts=new Uint32Array(64),ids=new Uint8Array(n),bounds=Array.from({length:64},()=>[Infinity,Infinity,Infinity,-Infinity,-Infinity,-Infinity]);
  let loX=Infinity,hiX=-Infinity,loZ=Infinity,hiZ=-Infinity;for(let i=0;i<pos.length;i+=3){loX=Math.min(loX,pos[i]);hiX=Math.max(hiX,pos[i]);loZ=Math.min(loZ,pos[i+2]);hiZ=Math.max(hiZ,pos[i+2]);}
  for(let t=0;t<n;t++){const a=idx[t*3]*3,b=idx[t*3+1]*3,c=idx[t*3+2]*3,x=(pos[a]+pos[b]+pos[c])/3,z=(pos[a+2]+pos[b+2]+pos[c+2])/3;const bx=Math.max(0,Math.min(7,Math.floor((x-loX)/Math.max(1e-9,hiX-loX)*8))),bz=Math.max(0,Math.min(7,Math.floor((z-loZ)/Math.max(1e-9,hiZ-loZ)*8))),k=bz*8+bx;ids[t]=k;counts[k]+=3;const bd=bounds[k];for(const v of [a,b,c])for(let d=0;d<3;d++){bd[d]=Math.min(bd[d],pos[v+d]);bd[d+3]=Math.max(bd[d+3],pos[v+d]);}}
  const starts=new Uint32Array(64),cursor=new Uint32Array(64);for(let k=1;k<64;k++)starts[k]=starts[k-1]+counts[k-1];cursor.set(starts);const ordered=new idx.constructor(idx.length);
  for(let t=0;t<n;t++){const k=ids[t],j=cursor[k];ordered[j]=idx[t*3];ordered[j+1]=idx[t*3+1];ordered[j+2]=idx[t*3+2];cursor[k]+=3;}
  p.index={...p.index,array:ordered};p.batches=Array.from({length:64},(_,k)=>({start:starts[k],count:counts[k],min:bounds[k].slice(0,3),max:bounds[k].slice(3)})).filter(b=>b.count);transfer.push(ordered.buffer);return p;
 }
 const meshes=doc.meshes.map(m=>m.primitives.map(p=>{if((p.mode??4)!==4)throw Error('Not triangles');return partition({position:acc(p.attributes.POSITION),color:p.attributes.COLOR_0===undefined?null:acc(p.attributes.COLOR_0),normal:p.attributes.NORMAL===undefined?null:acc(p.attributes.NORMAL),index:acc(p.indices)});}));
 self.postMessage({id,ok:true,meshes,nodes:doc.nodes,roots:doc.scenes[doc.scene||0].nodes,bytes,sha256:digest,workerMs:performance.now()-start,all_faces_preserved:true},transfer);
 }catch(e){self.postMessage({id,ok:false,error:String(e)})}};
