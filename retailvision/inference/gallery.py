import json
from pathlib import Path
import faiss
import numpy as np

class Gallery:
    def __init__(self, aggregation: str = 'max', support_k: int = 1):
        self._validate_matching(aggregation, support_k)
        self.ids: list[str]=[];self.vectors=np.empty((0,0),dtype='float32');self.index=None
        self.aggregation=aggregation;self.support_k=int(support_k)
        self.sku_ids=np.empty(0,dtype=str);self.sku_groups=np.empty(0,dtype=np.int64);self._sku_indices: list[np.ndarray]=[]
    @staticmethod
    def _validate_matching(aggregation: str, support_k: int) -> None:
        if aggregation not in ('max', 'topk_mean'):
            raise ValueError(f'unsupported gallery aggregation: {aggregation}')
        if int(support_k) < 1:
            raise ValueError('support_k must be >= 1')
    def build_index(self, embeddings: np.ndarray, sku_ids: list[str]) -> None:
        x=np.ascontiguousarray(embeddings,dtype='float32')
        if x.ndim!=2 or len(x)!=len(sku_ids) or not len(x) or not np.isfinite(x).all() or np.any(np.linalg.norm(x,axis=1)==0): raise ValueError('invalid gallery vectors')
        faiss.normalize_L2(x);self.vectors=x;self.ids=list(sku_ids)
        self.sku_ids,self.sku_groups=np.unique(self.ids,return_inverse=True)
        self._sku_indices=[np.flatnonzero(self.sku_groups==i) for i in range(len(self.sku_ids))]
        self.index=faiss.IndexFlatIP(x.shape[1]);self.index.add(x)
    def search(self, embedding: np.ndarray, k: int = 5, *, aggregation: str | None = None, support_k: int | None = None) -> list[tuple[str,float]]:
        if self.index is None: raise RuntimeError('gallery not built')
        mode = self.aggregation if aggregation is None else aggregation
        top_m = self.support_k if support_k is None else int(support_k)
        self._validate_matching(mode, top_m)
        q=np.array(embedding,dtype='float32').reshape(1,-1)
        if not np.isfinite(q).all() or np.linalg.norm(q)==0:raise ValueError('invalid embedding')
        faiss.normalize_L2(q)
        # Exact scores, reduced per SKU without sorting all reference vectors.
        scores=self.vectors@q[0]
        skus,inverse=self.sku_ids,self.sku_groups
        if mode == 'max' or top_m == 1:
            best=np.full(len(skus),-np.inf,dtype='float32')
            np.maximum.at(best,inverse,scores)
        else:
            best=np.empty(len(skus),dtype='float32')
            for idx, members in enumerate(self._sku_indices):
                vals=scores[members]
                if len(vals)<=top_m:
                    best[idx]=float(vals.mean())
                else:
                    best[idx]=float(np.partition(vals,-top_m)[-top_m:].mean())
        order=np.argsort(-best,kind='stable')[:k]
        return [(str(skus[i]),float(best[i])) for i in order]
    def add_sku(self, sku_id: str, embeddings: np.ndarray) -> None:
        x=np.asarray(embeddings,dtype='float32')
        self.build_index(np.concatenate([self.vectors,x]) if self.ids else x,self.ids+[sku_id]*len(x))
    def remove_sku(self, sku_id: str) -> None:
        keep=[i for i,s in enumerate(self.ids) if s!=sku_id]
        if not keep:self.__init__(aggregation=self.aggregation, support_k=self.support_k);return
        self.build_index(self.vectors[keep],[self.ids[i] for i in keep])
    def summary(self) -> dict:
        counts={}
        for s in self.ids:counts[s]=counts.get(s,0)+1
        return dict(ntotal=len(self.ids),unique_skus=len(counts),counts=counts)
    def save(self, path: str, overwrite: bool = False) -> None:
        p=Path(path);p.mkdir(parents=True,exist_ok=overwrite)
        if not len(self.ids) or self.index is None:raise ValueError('cannot save empty gallery')
        np.save(p/'embeddings.npy',self.vectors);(p/'skus.json').write_text(json.dumps(self.ids))
        faiss.write_index(self.index,str(p/'index.faiss'))
    @classmethod
    def load(cls,path: str, aggregation: str = 'max', support_k: int = 1):
        p=Path(path);g=cls(aggregation=aggregation, support_k=support_k);g.build_index(np.load(p/'embeddings.npy',allow_pickle=False),json.loads((p/'skus.json').read_text()));return g
