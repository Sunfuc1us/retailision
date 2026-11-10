import json
from pathlib import Path
import faiss
import numpy as np

class Gallery:
    def __init__(self):
        self.ids: list[str]=[];self.vectors=np.empty((0,0),dtype='float32');self.index=None
    def build_index(self, embeddings: np.ndarray, sku_ids: list[str]) -> None:
        x=np.ascontiguousarray(embeddings,dtype='float32')
        if x.ndim!=2 or len(x)!=len(sku_ids) or not len(x) or not np.isfinite(x).all() or np.any(np.linalg.norm(x,axis=1)==0): raise ValueError('invalid gallery vectors')
        faiss.normalize_L2(x);self.vectors=x;self.ids=list(sku_ids)
        self.index=faiss.IndexFlatIP(x.shape[1]);self.index.add(x)
    def search(self, embedding: np.ndarray, k: int = 5) -> list[tuple[str,float]]:
        if self.index is None: raise RuntimeError('gallery not built')
        q=np.array(embedding,dtype='float32').reshape(1,-1)
        if not np.isfinite(q).all() or np.linalg.norm(q)==0:raise ValueError('invalid embedding')
        faiss.normalize_L2(q)
        # Aggregate reference images by SKU before computing top-1/top-2 margin.
        scores,indices=self.index.search(q,len(self.ids));best={}
        for s,i in zip(scores[0],indices[0]):
            sku=self.ids[int(i)];best[sku]=max(best.get(sku,-float('inf')),float(s))
        return sorted(best.items(),key=lambda x:x[1],reverse=True)[:k]
    def add_sku(self, sku_id: str, embeddings: np.ndarray) -> None:
        x=np.asarray(embeddings,dtype='float32')
        self.build_index(np.concatenate([self.vectors,x]) if self.ids else x,self.ids+[sku_id]*len(x))
    def remove_sku(self, sku_id: str) -> None:
        keep=[i for i,s in enumerate(self.ids) if s!=sku_id]
        if not keep:self.__init__();return
        self.build_index(self.vectors[keep],[self.ids[i] for i in keep])
    def save(self, path: str) -> None:
        p=Path(path);p.mkdir(parents=True,exist_ok=False)
        np.save(p/'embeddings.npy',self.vectors);(p/'skus.json').write_text(json.dumps(self.ids))
        faiss.write_index(self.index,str(p/'index.faiss'))
    @classmethod
    def load(cls,path: str):
        p=Path(path);g=cls();g.build_index(np.load(p/'embeddings.npy',allow_pickle=False),json.loads((p/'skus.json').read_text()));return g
