import random

def split_groups(groups: list[str], seed: int = 42) -> dict[str,str]:
    unique = sorted(set(groups))
    if len(unique)<3: raise ValueError('at least three independent capture groups required')
    random.Random(seed).shuffle(unique)
    n=len(unique); val=max(1,round(n*.15)); test=max(1,round(n*.15))
    return {g: ('test' if i<test else 'val' if i<test+val else 'train') for i,g in enumerate(unique)}
