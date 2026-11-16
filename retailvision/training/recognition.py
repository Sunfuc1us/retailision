import json,runpy,random
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image


def training_rows(meta: dict, batch_size: int, *, no_validation: bool = False) -> tuple[list[dict],list[dict]]:
    items=meta['items']
    if no_validation:
        if not meta.get('no_split') or any(x['split']!='all' for x in items):
            raise ValueError('--no-validation requires an explicitly unsplit dataset')
        train=items;val=[]
    else:
        train=[x for x in items if x['split']=='train'];val=[x for x in items if x['split']=='val']
        if not val:raise ValueError('independent validation data required; unsplit experiments must explicitly use --no-validation')
    counts=Counter(x['sku_id'] for x in train)
    if len(counts)<2 or any(n<2 for n in counts.values()):
        raise ValueError('need >=2 training SKU classes and >=2 train references per SKU')
    if any(x['sku_id'] not in counts for x in val):raise ValueError('validation SKU has no training references')
    if batch_size<4 or batch_size%2 or batch_size>2*len(counts):
        raise ValueError('batch_size must be even, >=4 and <= 2 * training SKU count')
    return train,val


def _load_roi_contexts(dataset_dir: Path, rows: list[dict], context_padding: float = 0.25, max_context_side: int = 512) -> list[tuple[Image.Image, np.ndarray]]:
    """Pre-crop generous scene context + GT binary masks so training can expose real background."""
    from retailvision.image_ops import mask_bbox, polygon_mask
    seg_dir = dataset_dir.parent / 'segmentation'
    masks_dir = dataset_dir / 'masks'
    if not (seg_dir / 'images').is_dir() or not (seg_dir / 'annotations').is_dir():
        raise FileNotFoundError(f'roi_augment requires sibling segmentation directory next to {dataset_dir}')
    image_paths: dict[int, Path] = {}
    ann_polys: dict[int, tuple[list, int, int]] = {}
    for split in sorted({r['split'] for r in rows}):
        ann_path = seg_dir / 'annotations' / f'{split}.json'
        if not ann_path.is_file():
            raise FileNotFoundError(f'missing segmentation annotations for roi_augment: {ann_path}')
        coco = json.loads(ann_path.read_text())
        ims = {im['id']: im for im in coco['images']}
        for im in coco['images']:
            image_paths[im['id']] = seg_dir / 'images' / im['file_name']
        for ann in coco['annotations']:
            im = ims[ann['image_id']]
            ann_polys[ann['id']] = (ann['segmentation'], im['width'], im['height'])

    contexts: list[tuple[Image.Image, np.ndarray] | None] = [None] * len(rows)
    by_image: dict[int, list[tuple[int, dict]]] = {}
    for idx, row in enumerate(rows):
        by_image.setdefault(row['image_id'], []).append((idx, row))

    for image_id, items in by_image.items():
        with Image.open(image_paths[image_id]) as im:
            rgb = im.convert('RGB')
            for idx, row in items:
                ann_id = row['annotation_id']
                if ann_id in ann_polys and isinstance(ann_polys[ann_id][0], list) and ann_polys[ann_id][0]:
                    segs, width, height = ann_polys[ann_id]
                    full_mask = polygon_mask(segs, width, height)
                else:
                    mask_path = masks_dir / f'{ann_id}.png'
                    if not mask_path.is_file():
                        raise FileNotFoundError(f'missing instance mask for roi_augment: {mask_path}')
                    with Image.open(mask_path) as m:
                        full_mask = np.asarray(m) > 0
                x1, y1, x2, y2 = mask_bbox(full_mask)
                pad = int(max(x2 - x1, y2 - y1) * context_padding)
                box = (max(0, x1 - pad), max(0, y1 - pad), min(rgb.width, x2 + pad), min(rgb.height, y2 + pad))
                ctx_rgb = rgb.crop(box)
                ctx_mask = full_mask[box[1]:box[3], box[0]:box[2]].copy()
                longest = max(ctx_rgb.width, ctx_rgb.height)
                if longest > max_context_side:
                    scale = max_context_side / longest
                    nw = max(1, int(round(ctx_rgb.width * scale)))
                    nh = max(1, int(round(ctx_rgb.height * scale)))
                    ctx_rgb = ctx_rgb.resize((nw, nh), Image.BILINEAR)
                    ctx_mask = np.asarray(
                        Image.fromarray(ctx_mask.astype(np.uint8) * 255).resize((nw, nh), Image.NEAREST)
                    ) > 0
                contexts[idx] = (ctx_rgb, ctx_mask)
    return contexts  # type: ignore[return-value]


def train(config: str, dataset: str, experiment, resume: str | None, *, no_validation: bool = False) -> dict:
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from pytorch_metric_learning import losses,miners,samplers
    from retailvision.inference.dino import DINOv2
    from retailvision.training.augmentations import perturbed_mask_crop, training_transform
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU fallback disabled')
    c=runpy.run_path(config)['config'];seed=c['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    loss_mode=c.get('loss','triplet')
    if loss_mode not in ('triplet','contrastive','cosine_ce','triplet+cosine_ce'):
        raise ValueError(f'unsupported recognition loss: {loss_mode}')
    roi_aug_raw=c.get('roi_augment',False)
    roi_aug_cfg=dict(roi_aug_raw) if isinstance(roi_aug_raw,dict) else ({} if roi_aug_raw else None)
    dataset_dir=Path(dataset)
    meta=json.loads((dataset_dir/'metadata.json').read_text())
    train_rows,val_rows=training_rows(meta,c['batch_size'],no_validation=no_validation)
    labels=sorted({i['sku_id'] for i in train_rows});mapping={s:i for i,s in enumerate(labels)}
    experiment.data.update(hyperparameters=c,validation_performed=not no_validation,
        validation_status='not_performed' if no_validation else 'pending',
        device='cuda:0',train_instances=len(train_rows),validation_instances=len(val_rows),
        backbone_frozen=True)
    experiment.save()
    train_contexts=_load_roi_contexts(dataset_dir,train_rows) if roi_aug_cfg is not None else None
    train_tf=training_transform()
    roi_rng=np.random.default_rng(seed)

    class ROIs(torch.utils.data.Dataset):
        def __init__(self,rows,augment=False,contexts=None):
            self.augment=augment;self.rows=rows;self.contexts=contexts
            self.labels=[mapping[i['sku_id']] for i in rows]
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):
            row=self.rows[i]
            tf=train_tf if self.augment else runtime.transform
            if self.augment and self.contexts is not None:
                ctx_rgb,ctx_mask=self.contexts[i]
                crop=perturbed_mask_crop(ctx_rgb,ctx_mask,roi_rng,**roi_aug_cfg)
                return tf(crop),mapping[row['sku_id']]
            with Image.open(dataset_dir/row['path']) as im:
                return tf(im.convert('RGB')),mapping[row['sku_id']]

    trainset=ROIs(train_rows,augment=True,contexts=train_contexts);valset=ROIs(val_rows)
    runtime=DINOv2(c['source'],c['checkpoint']);runtime.load()
    for p in runtime.model.parameters():p.requires_grad=False
    sampler=samplers.MPerClassSampler(trainset.labels,m=2,batch_size=c['batch_size'],length_before_new_iter=max(c['batch_size'],len(trainset)))
    loader=torch.utils.data.DataLoader(trainset,batch_size=c['batch_size'],sampler=sampler)
    head=torch.nn.Linear(768,c['embedding_dim']).cuda()
    use_ce=loss_mode in ('cosine_ce','triplet+cosine_ce')
    use_metric=loss_mode in ('triplet','contrastive','triplet+cosine_ce')
    prototypes=None
    if use_ce:
        prototypes=torch.nn.Parameter(torch.empty(len(labels),c['embedding_dim'],device='cuda'))
        torch.nn.init.xavier_uniform_(prototypes)
    opt_params=list(head.parameters())+([prototypes] if prototypes is not None else [])
    opt=torch.optim.AdamW(opt_params,lr=c['learning_rate'],weight_decay=float(c.get('weight_decay',0.0)))
    lossfn=(losses.TripletMarginLoss() if loss_mode in ('triplet','triplet+cosine_ce') else losses.ContrastiveLoss()) if use_metric else None
    miner=miners.TripletMarginMiner(type_of_triplets='hard') if loss_mode in ('triplet','triplet+cosine_ce') else None
    cos_scale=float(c.get('cosine_scale',16.0))
    cos_margin=float(c.get('cosine_margin',0.0))
    label_smoothing=float(c.get('label_smoothing',0.05))
    ce_weight=float(c.get('ce_weight',1.0))
    start=0;best=None if no_validation else -1.;best_margin=None if no_validation else -float('inf')
    if resume:
        state=torch.load(resume,map_location='cuda',weights_only=False)
        if state['dataset_version']!=meta['dataset_version']:raise ValueError('resume dataset version mismatch')
        if state.get('validation_performed',True)!=(not no_validation):raise ValueError('resume validation mode mismatch')
        head.load_state_dict(state['head'])
        if prototypes is not None and state.get('prototypes') is not None:
            prototypes.data.copy_(state['prototypes'].to(prototypes.device))
        opt.load_state_dict(state['optimizer']);start=state['epoch']+1;best=state['best']
        best_margin=state.get('best_margin',-float('inf'))
        if state.get('roi_rng') is not None:roi_rng.bit_generator.state=state['roi_rng']
        torch.set_rng_state(state['rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']]);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        import shutil
        if not no_validation:
            for name in ('best.pth','best_head.pth'):
                previous=Path(resume).parent/name
                if previous.is_file() and previous.resolve()!=(experiment.path/name).resolve():shutil.copy2(previous,experiment.path/name)
    initial={k:v.detach().clone() for k,v in head.state_dict().items()}
    writer=SummaryWriter(str(experiment.path/'tensorboard'))
    history=[]
    try:
        for epoch in range(start,c['epochs']):
            head.train();total=0.;metric_total=0.;ce_total=0.;triplets=0
            for x,y in loader:
                with torch.no_grad():feat=runtime.model(x.cuda())
                e=torch.nn.functional.normalize(head(feat),dim=-1);y=y.cuda()
                loss=e.new_tensor(0.)
                if use_metric and lossfn is not None:
                    if miner is not None:
                        mined=miner(e,y);triplets+=len(mined[0]);mloss=lossfn(e,y,mined)
                    else:
                        mloss=lossfn(e,y)
                    metric_total+=float(mloss);loss=loss+mloss
                if use_ce and prototypes is not None:
                    w=torch.nn.functional.normalize(prototypes,dim=-1)
                    cos=e@w.T
                    if cos_margin>0:
                        one_hot=torch.nn.functional.one_hot(y,num_classes=len(labels)).to(cos.dtype)
                        cos=cos-one_hot*cos_margin
                    celoss=torch.nn.functional.cross_entropy(cos_scale*cos,y,label_smoothing=label_smoothing)
                    ce_total+=float(celoss);loss=loss+ce_weight*celoss
                opt.zero_grad();loss.backward();opt.step();total+=float(loss)
            head.eval();accuracy=None;val_margin=None;improved=False
            if not no_validation:
                def features(ds):
                    es=[];ys=[]
                    with torch.no_grad():
                        for x,y in torch.utils.data.DataLoader(ds,batch_size=c['batch_size']):es.append(torch.nn.functional.normalize(head(runtime.model(x.cuda())),dim=-1));ys.append(y.cuda())
                    return torch.cat(es),torch.cat(ys)
                # Validation queries use training references, never themselves.
                ref,ry=features(ROIs(train_rows));q,qy=features(valset)
                sims=q@ref.T
                accuracy=float((ry[sims.argmax(1)]==qy).float().mean())
                sku_sims=torch.stack([sims[:,ry==cid].max(dim=1).values for cid in range(len(labels))],dim=1)
                true_sim=sku_sims.gather(1,qy.unsqueeze(1)).squeeze(1)
                other_mask=torch.nn.functional.one_hot(qy,num_classes=len(labels)).bool()
                other_sim=sku_sims.masked_fill(other_mask,-float('inf')).max(dim=1).values
                val_margin=float((true_sim-other_sim).mean())
                improved=(accuracy>best) or (accuracy==best and val_margin>best_margin)
                if improved:best=accuracy;best_margin=val_margin
            state=dict(head=head.state_dict(),prototypes=(prototypes.detach().cpu() if prototypes is not None else None),
                       optimizer=opt.state_dict(),epoch=epoch,best=best,best_margin=best_margin,
                       roi_rng=roi_rng.bit_generator.state,
                       rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                       python_rng=random.getstate(),numpy_rng=np.random.get_state(),config=c,
                       dataset_version=meta['dataset_version'],validation_performed=not no_validation)
            torch.save(state,experiment.path/'last.pth')
            torch.save(dict(head=head.state_dict()),experiment.path/'last_head.pth')
            if improved:
                torch.save(state,experiment.path/'best.pth')
                torch.save(dict(head=head.state_dict()),experiment.path/'best_head.pth')
            row=dict(epoch=epoch+1,loss=total/len(loader),mined_triplets=triplets)
            if use_metric and use_ce:
                row['metric_loss']=metric_total/len(loader);row['ce_loss']=ce_total/len(loader)
            if accuracy is not None:
                row['validation_top1']=accuracy;row['validation_margin']=val_margin
            history.append(row)
            (experiment.path/'loss_history.json').write_text(json.dumps(history,indent=2,allow_nan=False))
            writer.add_scalar('train/loss',row['loss'],epoch)
            if accuracy is not None:
                writer.add_scalar('val/top1',accuracy,epoch)
                writer.add_scalar('val/margin',val_margin,epoch)
            experiment.data.update(training_metrics=row,validation_metrics={} if no_validation else dict(top1=accuracy,margin=val_margin),
                last_checkpoint=str(experiment.path/'last.pth'),
                best_checkpoint=None if no_validation else str(experiment.path/'best.pth'))
            experiment.save();print(json.dumps(row),flush=True)
    finally:
        writer.close()
    delta=float(torch.sqrt(sum(((v-initial[k])**2).sum() for k,v in head.state_dict().items())))
    return dict(best_checkpoint=None if no_validation else str(experiment.path/'best.pth'),
        last_checkpoint=str(experiment.path/'last.pth'),projection_checkpoint=str(experiment.path/('last_head.pth' if no_validation else 'best_head.pth')),
        validation_performed=not no_validation,validation_status='not_performed' if no_validation else 'completed',
        validation_metrics={} if no_validation else dict(best_top1=best,best_margin=best_margin),parameter_change_l2=delta)
