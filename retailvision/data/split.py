import random
from collections import Counter

def split_groups(groups: list[str], seed: int = 42) -> dict[str,str]:
    """Reserve about 10% of images for validation without separating capture groups."""
    counts = Counter(groups)
    unique = sorted(set(groups))
    if len(unique)<2: raise ValueError('at least two independent capture groups required for train/val splitting')
    random.Random(seed).shuffle(unique)
    target = max(1, (len(groups) + 5) // 10)
    first = min(unique, key=lambda g: abs(counts[g] - target))
    validation = {first}
    size = counts[first]
    for group in unique:
        if group in validation or len(validation) == len(unique) - 1:
            continue
        if abs(size + counts[group] - target) < abs(size - target):
            validation.add(group)
            size += counts[group]
    return {g: 'val' if g in validation else 'train' for g in unique}
