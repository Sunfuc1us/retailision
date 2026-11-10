import json,runpy,random
from pathlib import Path
import numpy as np
from PIL import Image

def train(config: str, dataset: str, experiment, resume: str | None) -> dict:
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from pytorch_metric_learning import losses,miners,samplers
    from inference.recognition.dino import DINOv2
    from dataset.augmentations.recognition import training_transform
    c=runpy.run_path(config)['config'];seed=c['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    experiment.data['hyperparameters']=c;experiment.save()
    runtime=DINOv2(c['source'],c['checkpoint']);runtime.load()
    for p in runtime.model.parameters():p.requires_grad=False
    meta=json.loads((Path(dataset)/'metadata.json').read_text());items=meta['items'];labels=sorted(set(i['sku_id'] for i in items));mapping={s:i for i,s in enumerate(labels)}
    if len(labels)<2:raise ValueError('metric learning requires >=2 SKU classes')
    class ROIs(torch.utils.data.Dataset):
        def __init__(self,split):self.rows=[i for i in items if i['split']==split];self.labels=[mapping[i['sku_id']] for i in self.rows]
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):
            row=self.rows[i]
            with Image.open(Path(dataset)/row['path']) as im:return (training_transform() if row['split']=='train' else runtime.transform)(im.convert('RGB')),mapping[row['sku_id']]
    trainset=ROIs('train');valset=ROIs('val')
    if not valset.rows or any(trainset.labels.count(x)<2 for x in set(trainset.labels)):raise ValueError('need validation data and >=2 train references per SKU')
    sampler=samplers.MPerClassSampler(trainset.labels,m=2,batch_size=c['batch_size'],length_before_new_iter=max(c['batch_size'],len(trainset)))
    loader=torch.utils.data.DataLoader(trainset,batch_size=c['batch_size'],sampler=sampler)
    head=torch.nn.Linear(768,c['embedding_dim']).cuda();opt=torch.optim.AdamW(head.parameters(),lr=c['learning_rate'])
    lossfn=losses.TripletMarginLoss() if c['loss']=='triplet' else losses.ContrastiveLoss()
    miner=miners.TripletMarginMiner(type_of_triplets='hard');start=0;best=-1.
    if resume:
        state=torch.load(resume,map_location='cuda')
        if state['dataset_version']!=meta['dataset_version']:raise ValueError('resume dataset version mismatch')
        head.load_state_dict(state['head']);opt.load_state_dict(state['optimizer']);start=state['epoch']+1;best=state['best']
        torch.set_rng_state(state['rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']]);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        import shutil
        for name in ('best.pth','best_head.pth'):
            previous=Path(resume).parent/name
            if previous.is_file() and previous.resolve()!=(experiment.path/name).resolve():shutil.copy2(previous,experiment.path/name)
    writer=SummaryWriter(str(experiment.path/'tensorboard'))
    for epoch in range(start,c['epochs']):
        head.train();total=0.
        for x,y in loader:
            with torch.no_grad():feat=runtime.model(x.cuda())
            e=torch.nn.functional.normalize(head(feat),dim=-1);y=y.cuda();loss=lossfn(e,y,miner(e,y));opt.zero_grad();loss.backward();opt.step();total+=float(loss)
        head.eval()
        def features(ds):
            es=[];ys=[]
            with torch.no_grad():
                for x,y in torch.utils.data.DataLoader(ds,batch_size=c['batch_size']):es.append(torch.nn.functional.normalize(head(runtime.model(x.cuda())),dim=-1));ys.append(y.cuda())
            return torch.cat(es),torch.cat(ys)
        # Evaluate validation queries against train references, never against themselves.
        ref,ry=features(ROIs('train'));q,qy=features(valset);accuracy=float((ry[(q@ref.T).argmax(1)]==qy).float().mean())
        improved=accuracy>best;best=max(best,accuracy)
        state=dict(head=head.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),config=c,dataset_version=meta['dataset_version'])
        torch.save(state,experiment.path/'last.pth')
        if improved:
            torch.save(state,experiment.path/'best.pth')
            torch.save(dict(head=head.state_dict()),experiment.path/'best_head.pth')
        writer.add_scalar('train/loss',total/len(loader),epoch);writer.add_scalar('val/top1',accuracy,epoch)
        experiment.data.update(training_metrics=dict(loss=total/len(loader),epoch=epoch),validation_metrics=dict(top1=accuracy),best_checkpoint=str(experiment.path/'best.pth'));experiment.save()
    writer.close();return dict(best_checkpoint=str(experiment.path/'best.pth'),validation_metrics=dict(best_top1=best))
