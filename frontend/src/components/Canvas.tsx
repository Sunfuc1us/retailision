import {useEffect,useRef,useState} from 'react';
import type {CandidateInstance} from '../types/vision';

interface Props {
  src:string;
  instances:CandidateInstance[];
  matched:boolean;
  mode:string;
  opacity:number;
  selectedId:number|null;
  onSelect:(id:number)=>void;
}

export default function Canvas({src,instances,matched,mode,opacity,selectedId,onSelect}:Props){
  const cache=useRef(new Map<string,Promise<HTMLImageElement>>());
  const ref=useRef<HTMLCanvasElement>(null);
  const viewport=useRef<HTMLDivElement>(null);
  const drag=useRef<{x:number;y:number;panX:number;panY:number}|null>(null);
  const moved=useRef(false);
  const[view,setView]=useState({zoom:1,x:0,y:0});
  const[area,setArea]=useState({width:1,height:1});
  const[natural,setNatural]=useState({width:1,height:1});

  useEffect(()=>{cache.current.clear()},[src,instances]);

  useEffect(()=>{
    setView({zoom:1,x:0,y:0});
    const element=viewport.current;if(!element)return;
    const observer=new ResizeObserver(()=>{
      setArea({width:Math.max(1,element.clientWidth-32),height:Math.max(1,element.clientHeight-32)});
    });
    observer.observe(element);
    function wheel(event:WheelEvent){
      event.preventDefault();
      const bounds=element!.getBoundingClientRect();
      const x=event.clientX-bounds.left-bounds.width/2,y=event.clientY-bounds.top-bounds.height/2;
      const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?bounds.height:1);
      setView(previous=>{
        const zoom=Math.max(.25,Math.min(8,previous.zoom*Math.exp(-delta*.002)));
        const ratio=zoom/previous.zoom;
        return {zoom,x:x-(x-previous.x)*ratio,y:y-(y-previous.y)*ratio};
      });
    }
    element.addEventListener('wheel',wheel,{passive:false});
    return()=>{observer.disconnect();element.removeEventListener('wheel',wheel)};
  },[src]);

  useEffect(()=>{
    let cancelled=false;
    async function draw(){
      const canvas=ref.current;if(!canvas||!src)return;
      function load(url:string){
        let pending=cache.current.get(url);
        if(!pending){pending=(async()=>{const image=new Image();image.src=url;await image.decode();return image})();cache.current.set(url,pending)}
        return pending;
      }
      const img=await load(src);if(cancelled)return;
      canvas.width=img.width;canvas.height=img.height;
      setNatural({width:img.width,height:img.height});
      const ctx=canvas.getContext('2d')!;
      ctx.clearRect(0,0,img.width,img.height);
      if(mode!=='Mask')ctx.drawImage(img,0,0);
      else{ctx.fillStyle='#111';ctx.fillRect(0,0,img.width,img.height)}
      if(mode==='Original')return;
      for(const item of instances){
        if(!item.mask_path)continue;
        const mask=await load(item.mask_preview_path??item.mask_path);if(cancelled)return;
        const temp=document.createElement('canvas');temp.width=mask.width;temp.height=mask.height;
        const tc=temp.getContext('2d')!;tc.drawImage(mask,0,0);
        const pixels=tc.getImageData(0,0,mask.width,mask.height);
        const [r,g,b]=mode==='Mask'?[255,255,255]:matched?[62,137,114]:[184,107,40];
        for(let i=0;i<pixels.data.length;i+=4){
          const alpha=pixels.data[i];pixels.data[i]=r;pixels.data[i+1]=g;pixels.data[i+2]=b;pixels.data[i+3]=alpha;
        }
        tc.putImageData(pixels,0,0);ctx.globalAlpha=mode==='Mask'?1:opacity;ctx.drawImage(temp,...(item.mask_origin??[0,0]) as [number,number]);ctx.globalAlpha=1;
      }
    }
    draw().catch(console.error);
    return()=>{cancelled=true};
  },[src,instances,matched,mode,opacity]);

  const fit=Math.min(area.width/natural.width,area.height/natural.height);
  const size={width:natural.width*fit,height:natural.height*fit};
  if(!src)return <div className="empty">选择商品图片</div>;
  return <div ref={viewport} className="image-viewport" tabIndex={0} aria-label="图片视图，滚轮缩放，拖动平移，Home 复位"
    onPointerDown={event=>{
      if(event.button!==0||(event.target as HTMLElement).closest('.view-reset'))return;
      drag.current={x:event.clientX,y:event.clientY,panX:view.x,panY:view.y};moved.current=false;
    }}
    onPointerMove={event=>{
      const start=drag.current;if(!start)return;
      const x=event.clientX-start.x,y=event.clientY-start.y;
      if(!moved.current&&Math.hypot(x,y)<4)return;
      moved.current=true;event.currentTarget.setPointerCapture(event.pointerId);
      setView(previous=>({...previous,x:start.panX+x,y:start.panY+y}));
    }}
    onPointerUp={()=>{drag.current=null}}
    onPointerCancel={()=>{drag.current=null}}
    onClickCapture={event=>{if(moved.current){event.preventDefault();event.stopPropagation();moved.current=false}}}
    onKeyDown={event=>{
      if(event.target!==event.currentTarget)return;
      if(event.key==='Home'){event.preventDefault();setView({zoom:1,x:0,y:0})}
      else if(['+','=','-','ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){
        event.preventDefault();setView(previous=>({...previous,
          zoom:Math.max(.25,Math.min(8,previous.zoom*(event.key==='+'||event.key==='='?1.2:event.key==='-'?1/1.2:1))),
          x:previous.x+(event.key==='ArrowLeft'?40:event.key==='ArrowRight'?-40:0),
          y:previous.y+(event.key==='ArrowUp'?40:event.key==='ArrowDown'?-40:0)}));
      }
    }}>
    <div className="image-stage" style={{width:size.width,height:size.height,transform:`translate(-50%, -50%) translate(${view.x}px, ${view.y}px) scale(${view.zoom})`}}>
    <canvas ref={ref} aria-label="商品图片与实例 Mask"/>
    {mode!=='Original'&&<div className={`instance-layer${matched?'':' candidates'}`} style={{width:size.width,height:size.height}}>
      {instances.map(item=>{
        const [x,y,x2,y2]=item.bbox;
        const left=x/natural.width*size.width,top=y/natural.height*size.height;
        const score=item.detection_score?.toFixed(3)??'—';
        return <button key={item.instance_id} type="button"
          className={`instance-box${selectedId===item.instance_id?' selected':''}${selectedId!==null&&selectedId!==item.instance_id?' muted':''}`}
          aria-label={`图中实例 ${item.instance_id}，商品检测分数 ${score}`} aria-pressed={selectedId===item.instance_id}
          title={`实例 ${item.instance_id} · 商品检测分数 ${score}，不代表 SKU 身份`}
          style={{left,top,width:(x2-x)/natural.width*size.width,height:(y2-y)/natural.height*size.height}}
          onClick={()=>onSelect(item.instance_id)}>
          <span className="instance-label" style={{top:top<25?0:-25,left:Math.min(0,size.width-left-135)}}>#{item.instance_id} · 检测 {score}</span>
        </button>;
      })}
    </div>}
    </div>
    <button className="view-reset" onClick={()=>setView({zoom:1,x:0,y:0})} aria-label="复位图片视图">适应窗口</button>
  </div>;
}
