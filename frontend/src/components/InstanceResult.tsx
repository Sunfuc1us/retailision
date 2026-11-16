import type {CandidateInstance} from '../types/vision';

export default function InstanceResult({instance,selected,candidate=false,onSelect}:{
  instance:CandidateInstance;selected:boolean;candidate?:boolean;onSelect:(id:number)=>void;
}){
  const x=instance;
  const rejection:Record<string,string>={below_similarity:'SKU 相似度不足',below_margin:'SKU 候选差值不足',insufficient_candidates:'SKU 候选不足',empty_mask_in_bbox:'Mask 为空'};
  return <article className={`instance-result${selected?' selected':''}`} id={`instance-${x.instance_id}`}>
    <button className="instance-select" aria-label={`选择实例 ${x.instance_id}`} aria-pressed={selected} onClick={()=>onSelect(x.instance_id)}>
      <strong>实例 {x.instance_id}</strong>
      <span>商品检测分数 {x.detection_score?.toFixed(3)??'—'}</span>
    </button>
    <p title="余弦相似度，不是 SKU 识别正确率">SKU 相似度 {x.identity_score?.toFixed(3)??'—'}</p>
    {candidate&&<>
      {x.accepted===false&&<p className="hint">未确认 SKU{x.reason?` · ${rejection[x.reason]??x.reason}`:''}</p>}
      {x.mask_pixels!==undefined&&<p>Mask {x.mask_pixels} px</p>}
      {x.top1_sku&&<p>Top-1 {x.top1_sku}{x.margin!==undefined&&x.margin!==null?` · Margin ${x.margin.toFixed(3)}`:''}</p>}
      {x.top_k&&x.top_k.length>1&&<p className="hint">{x.top_k.slice(0,3).map(k=>`${k.sku_id} ${k.score.toFixed(3)}`).join(' / ')}</p>}
    </>}
    <p>Box {x.bbox.map(n=>Math.round(n)).join(', ')}</p>
    {x.mask_path&&<a href={x.mask_path} download>{candidate?'导出候选 Mask':'导出 Mask'}</a>}
  </article>;
}
