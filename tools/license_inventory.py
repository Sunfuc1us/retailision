import importlib.metadata as m,json
from pathlib import Path
rows=[]
for dist in m.distributions():
 meta=dist.metadata
 rows.append(dict(name=meta['Name'],version=dist.version,repository=meta.get('Home-page') or meta.get_all('Project-URL'),license=meta.get('License-Expression') or meta.get('License') or [x for x in meta.get_all('Classifier',[]) if x.startswith('License')],local_deployment='installed in isolated conda environment',commercial_use_notes='Review declared license and bundled notices; metadata is not legal approval'))
Path('environments/licenses-installed.json').write_text(json.dumps(sorted(rows,key=lambda x:x['name'].lower()),indent=2))
