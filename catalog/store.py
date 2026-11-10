import json
from pathlib import Path

class Catalog:
    def __init__(self, path: str | Path):
        items = json.loads(Path(path).read_text())
        self.items = {x['sku_id']: x for x in items}
        if len(items) != len(self.items):
            raise ValueError('duplicate sku_id in master catalog')
        if any(not x for x in self.items):
            raise ValueError('empty sku_id')

    def resolve(self, query: str) -> list[dict]:
        if query in self.items:
            return [self.items[query]]
        q = query.strip().casefold()
        if not q:
            return []
        return [x for x in self.items.values() if q in ' '.join(str(x.get(k, '')) for k in ('brand','product','flavor','specification')).casefold()]
