import {useEffect,useMemo,useState} from 'react';
import {Upload,Play,Download} from 'lucide-react';
import Canvas from '../components/Canvas';
import InstanceResult from '../components/InstanceResult';
import {predict,request} from '../services/api';
import type {SKU,Prediction,InferenceModel,InferenceModels} from '../types/vision';

const backendLabels:Record<string,string>={pytorch:'PyTorch',onnxruntime:'ONNX Runtime GPU',tensorrt:'TensorRT'};

export default function Workbench(){
  const[file,setFile]=useState<File|null>(null),[src,setSrc]=useState('');
  const[target,setTarget]=useState(''),[catalog,setCatalog]=useState<SKU[]>([]);
  const[models,setModels]=useState<InferenceModel[]>([]),[modelId,setModelId]=useState('');
  const[backend,setBackend]=useState('pytorch'),[loadingCatalog,setLoadingCatalog]=useState(false);
  const[analysis,setResult]=useState<Prediction|null>(null),[error,setError]=useState(''),[running,setRunning]=useState(false);
  const[mode,setMode]=useState('Overlay'),[opacity,setOpacity]=useState(.8),[threshold,setThreshold]=useState(.8);
  const[resolved,setResolved]=useState<{query:string;items:SKU[]}>({query:'',items:[]});
  useEffect(()=>{
    if(!modelId||!target.trim())return;
    let active=true;
    request<SKU[]>('/api/catalog?model_id='+encodeURIComponent(modelId)+'&q='+encodeURIComponent(target))
      .then(items=>{if(active)setResolved({query:target,items})}).catch(()=>{if(active)setResolved({query:target,items:[]})});
    return()=>{active=false};
  },[modelId,target]);
  const[elapsed,setElapsed]=useState<number|null>(null);
  const result=useMemo(()=>{
    if(!analysis?.candidates)return analysis;
    const found=resolved.query===target?resolved.items:[];
    if(found.length!==1)return {...analysis,target_found:false,sku_id:undefined,identity_score:undefined,instances:[],reason:found.length?'ambiguous_target':'unknown_target'};
    const matches=analysis.candidates.filter(x=>x.sku_id===found[0].sku_id&&x.accepted!==false);
    return {...analysis,sku_id:found[0].sku_id,identity_score:matches.length?Math.max(...matches.map(x=>x.identity_score??0)):undefined,target_found:matches.length>0,instances:matches as Prediction['instances'],reason:matches.length?undefined:'no_confident_match'};
  },[analysis,target,resolved]);
  const[selectedInstance,setSelectedInstance]=useState<number|null>(null);
  const[showCandidates,setShowCandidates]=useState(false);
  const candidates=useMemo(()=>(result?.candidates??[]).filter(x=>Array.isArray(x.bbox)),[result]);
  const instances=useMemo(()=>result?.instances?.length?result.instances:showCandidates?candidates:[],[result,showCandidates,candidates]);
  const selected=models.find(x=>x.id===modelId);
  const backends=selected?.backends??[];
  const resultMessage=result?.target_found?'找到目标':result?.reason==='no_confident_match'
    ? result.candidates?.length===0?'未检测到商品实例'
      :'未匹配目标 SKU'
    :result?.reason==='ambiguous_target'?'匹配到多个商品，请从输入框候选中选择具体商品'
    :result?.reason==='unknown_target'?'未找到该商品，请检查商品名称或 SKU'
    :result?.reason;

  useEffect(()=>{setSelectedInstance(null);setShowCandidates(false)},[result]);
  function selectInstance(id:number,fromImage=false){
    setSelectedInstance(previous=>previous===id?null:id);
    if(mode==='Original')setMode('Overlay');
    if(fromImage){const row=document.getElementById(`instance-${id}`);const details=row?.closest('details');if(details)details.open=true;row?.scrollIntoView({block:'nearest'})}
  }

  function chooseModel(model:InferenceModel|undefined){
    if(model?.id===modelId)return;
    setModelId(model?.id??'');setTarget('');setCatalog([]);setResult(null);setError('');
    setLoadingCatalog(Boolean(model));
    setBackend(previous=>model?.backends.includes(previous)?previous:model?.backends[0]??'');
  }
  useEffect(()=>{
    let active=true;
    request<InferenceModels>('/api/inference/models').then(value=>{
      if(!active)return;
      setModels(value.models);
      chooseModel(value.models.find(x=>x.id===value.default_model)??value.models.find(x=>x.id==='retail_6sku_v002'&&x.backends.length)??value.models.find(x=>x.backends.length));
    }).catch(e=>{if(active)setError(e.message)});
    return()=>{active=false};
  },[]);
  useEffect(()=>{
    if(!modelId)return;
    let active=true;
    setLoadingCatalog(true);
    request<SKU[]>('/api/catalog?model_id='+encodeURIComponent(modelId))
      .then(value=>{if(active)setCatalog(value)})
      .catch(e=>{if(active)setError(e.message)})
      .finally(()=>{if(active)setLoadingCatalog(false)});
    return()=>{active=false};
  },[modelId]);
  useEffect(()=>{
    if(!file)return;
    const url=URL.createObjectURL(file);setSrc(url);
    return()=>URL.revokeObjectURL(url);
  },[file]);

  async function run(){
    if(!file||!selected)return;
    const started=performance.now();
    setRunning(true);setError('');setResult(null);
    try{setResult(await predict(file,target,threshold,backend,modelId))}
    catch(e){setError((e as Error).message)}
    finally{setRunning(false);setElapsed(performance.now()-started)}
  }
  function exportResult(){
    const a=document.createElement('a');
    a.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
    a.download='retailvision-result.json';a.click();URL.revokeObjectURL(a.href);
  }

  return <div className="app">
    <header><strong>RetailVision</strong><span>Vision Workbench</span><span className="local">本地工作站</span></header>
    <div className="workspace">
      <aside>
        <h2>输入</h2>
        <label className="upload"><Upload size={18}/>选择图片
          <input className="sr-only" aria-label="上传商品图片" type="file" accept="image/*" disabled={running}
            onChange={e=>{setFile(e.target.files?.[0]??null);setResult(null);setError('')}}/>
        </label>
        <p className="filename">{file?.name??'尚未选择图片'}</p>
        <label htmlFor="target">目标商品</label>
        <input id="target" placeholder="商品名称或 SKU ID" list="skus" value={target} disabled={running||loadingCatalog}
          onChange={e=>{setTarget(e.target.value)}}/>
        <datalist id="skus">{catalog.map(x=><option key={x.sku_id}
          value={[`${x.brand??''}${x.product??''}`,x.flavor,x.specification].filter(Boolean).join(' ')}
          label={x.sku_id}/>)}</datalist>
        <details className="settings"><summary>运行设置</summary>
        <h2>模型</h2>
        <label>分割权重
          <select aria-label="分割模型" value={modelId} disabled={running||!models.length}
            onChange={e=>chooseModel(models.find(x=>x.id===e.target.value))}>
            {!modelId&&<option value="">请选择模型权重</option>}
            {models.map(x=><option key={x.id} value={x.id} disabled={!x.backends.length}>{x.name}</option>)}
          </select>
        </label>
        {selected?.checkpoint&&<p className="hint model-path" title={selected.checkpoint}>
          {selected.checkpoint.split('/').pop()}
        </p>}
        <p className="hint">识别模型 · {selected?.recognition_version||'未配置'}</p>
        <label>推理 Backend
          <select aria-label="推理 Backend" value={backend} disabled={running||!backends.length}
            onChange={e=>{setBackend(e.target.value);setResult(null);setError('')}}>
            {backends.map(x=><option key={x} value={x}>{backendLabels[x]??x}</option>)}
          </select>
        </label>
        <p className="hint">{selected?.fixture?'Fixture 验证 · 非商品精度评估':selected?'使用所选权重及配套 SKU 图库':'尚未配置可用模型'}</p>
        <label>检测阈值<input aria-label="检测阈值" type="number" min="0" max="1" step="0.01" value={threshold} disabled={running} onChange={e=>{setThreshold(+e.target.value);setResult(null)}}/></label>
        </details>
        <button className="run" onClick={run}
          disabled={!file||!target.trim()||running||loadingCatalog||!catalog.length||!backends.includes(backend)}>
          <Play size={15}/>{running?'分析中…':analysis?'重新分析':'分析图片'}
        </button>
      </aside>
      <main>
        <nav aria-label="显示模式">
          {['Original','Overlay','Mask'].map(x=><button key={x} aria-pressed={mode===x} onClick={()=>setMode(x)}>{x}</button>)}
          <label className="opacity">透明度<input aria-label="Mask 透明度" type="range" min="0" max="1" step=".05" value={opacity} onChange={e=>setOpacity(+e.target.value)}/></label>
        </nav>
        <div className="canvas"><Canvas src={src} instances={instances} matched={Boolean(result?.instances?.length)}
          mode={mode} opacity={opacity} selectedId={selectedInstance} onSelect={id=>selectInstance(id,true)}/></div>
      </main>
      <section className="inspector">
        <h2>识别结果</h2>
        <p className="hint">检测只判断商品区域；SKU 相似度不代表正确率。</p>
        {error&&<p role="alert" className="error">{error}</p>}
        {result?<>
          <p role="status" className="result-status">{resultMessage}</p><p>{catalog.find(x=>x.sku_id===result.sku_id)?.product??result.sku_id}</p>
          {elapsed!==null&&<p className="hint">请求耗时 {(elapsed/1000).toFixed(2)} 秒 · {analysis?.candidates?.length??0} 个候选</p>}
          <p>{backendLabels[result.backend??backend]??result.backend??backend} · {result.timing?.total_ms.toFixed(1)??'—'} ms</p>
          {result.timing?.segmentation_ms!==undefined&&result.timing?.recognition_ms!==undefined&&<p className="hint">
            分割 {result.timing.segmentation_ms.toFixed(1)} ms · 识别 {result.timing.recognition_ms.toFixed(1)} ms
          </p>}
          {result.instances?.map(x=><InstanceResult key={x.instance_id} instance={x} selected={selectedInstance===x.instance_id} onSelect={selectInstance}/>)}
          {!result.target_found&&candidates.length>0&&<details className="diag" open={showCandidates}
            onToggle={event=>{setShowCandidates(event.currentTarget.open);if(!event.currentTarget.open)setSelectedInstance(null)}}>
            <summary>全部商品候选诊断 ({candidates.length})</summary>
            {candidates.map(c=><InstanceResult key={c.instance_id} instance={c} candidate selected={selectedInstance===c.instance_id} onSelect={selectInstance}/>)}
          </details>}
          <button onClick={exportResult}><Download size={14}/>导出 JSON</button>
        </>:!error&&<p className="hint">{running?'正在分析图片…':'等待分析'}</p>}
      </section>
    </div>
  </div>;
}
