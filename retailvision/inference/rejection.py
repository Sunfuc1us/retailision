from dataclasses import dataclass,field
import numpy as np

@dataclass
class RejectionPolicy:
    similarity: float
    margin: float
    dataset_version: str
    gallery_version: str
    model_version: str
    sku_similarity: dict[str,float] = field(default_factory=dict)
    def similarity_for(self, sku_id: str) -> float:
        return self.sku_similarity.get(sku_id,self.similarity)
    def accepts(self, candidates: list[tuple[str,float]]) -> bool:
        return len(candidates)>=2 and candidates[0][1]>=self.similarity_for(candidates[0][0]) and candidates[0][1]-candidates[1][1]>=self.margin

def calibrate(rows: list[dict], dataset_version: str, gallery_version: str, model_version: str, max_false_accept_rate: float = .01) -> dict:
    if not 0<=max_false_accept_rate<=1:raise ValueError('invalid FAR limit')
    known=[r for r in rows if r['known']];unknown=[r for r in rows if not r['known']]
    if not known or not unknown:raise ValueError('independent validation requires known and unknown SKU samples')
    best=None
    for s in sorted(set([r['score'] for r in rows]+[1.000001])):
        for m in sorted(set([r['margin'] for r in rows]+[0.,2.000001])):
            far=sum(r['score']>=s and r['margin']>=m for r in unknown)/len(unknown)
            recall=sum(r['score']>=s and r['margin']>=m and r['correct'] for r in known)/len(known)
            if far<=max_false_accept_rate and (best is None or recall>best[0]):best=(recall,s,m,far)
    return dict(similarity=best[1],margin=best[2],dataset_version=dataset_version,gallery_version=gallery_version,model_version=model_version,calibration_metrics=dict(known_recall=best[0],unknown_false_accept_rate=best[3],known_count=len(known),unknown_count=len(unknown)))
