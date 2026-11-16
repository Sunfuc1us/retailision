import json
import unicodedata
from pathlib import Path

def _normalize(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())

class Catalog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        items = json.loads(self.path.read_text())
        self.items = {x['sku_id']: x for x in items}
        if len(items) != len(self.items):
            raise ValueError('duplicate sku_id in master catalog')
        if any(not x for x in self.items):
            raise ValueError('empty sku_id')

    def resolve(self, query: str) -> list[dict]:
        query = query.strip()
        if query in self.items:
            return [self.items[query]]
        q = _normalize(query)
        if not q:
            return []
        exact = [x for x in self.items.values() if _normalize(x['sku_id'])==q]
        if exact:
            return exact
        matches = []
        for item in self.items.values():
            fields = [_normalize(str(item.get(k, ''))).replace(' ','') for k in ('brand','product','flavor','specification','sku_id')]
            text = ''.join(fields)
            brand = fields[0]
            # Match compact Chinese names across fields, or all supplied keywords.
            # Brand + flavor also works when the product field sits between them.
            if all(term in text or (brand and term.startswith(brand) and term[len(brand):] in text)
                   for term in q.split()):
                matches.append(item)
        return matches

    def register(self, sku_id: str, product: str, brand: str = '', flavor: str = '', specification: str = '', **extra) -> dict:
        sku_id = sku_id.strip()
        product = product.strip()
        if not sku_id or Path(sku_id).name != sku_id or sku_id in ('.', '..'):
            raise ValueError('invalid sku_id')
        if not product:
            raise ValueError('product name required')
        entry = dict(sku_id=sku_id, brand=brand.strip(), product=product, flavor=flavor.strip(), specification=specification.strip(), **extra)
        self.items[sku_id] = entry
        return entry

    def remove(self, sku_id: str) -> None:
        if sku_id not in self.items:
            raise KeyError(sku_id)
        del self.items[sku_id]

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(list(self.items.values()), ensure_ascii=False, indent=2))
        tmp.replace(target)
        return target
