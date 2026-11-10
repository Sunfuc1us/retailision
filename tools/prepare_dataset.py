import _bootstrap
import argparse,json
from dataset.converters.prepare import prepare
from dataset.validators.coco import validate
p=argparse.ArgumentParser();p.add_argument('--coco',required=True);p.add_argument('--images',required=True);p.add_argument('--catalog',required=True);p.add_argument('--sidecar');p.add_argument('--output');p.add_argument('--validate-only',action='store_true');p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
if a.validate_only:
    r=validate(a.coco,a.images,a.catalog,a.sidecar);r.pop('records');print(json.dumps(r,indent=2));raise SystemExit(0 if r['valid'] else 1)
if not a.output:p.error('--output required')
print(json.dumps(prepare(a.coco,a.images,a.catalog,a.output,a.sidecar,a.seed),indent=2))
