import _bootstrap
import argparse,json
from retailvision.data.prepare import prepare
from retailvision.data.validate import validate
p=argparse.ArgumentParser();p.add_argument('--coco',required=True);p.add_argument('--images',required=True);p.add_argument('--catalog',required=True);p.add_argument('--sidecar');p.add_argument('--output');p.add_argument('--validate-only',action='store_true');p.add_argument('--seed',type=int,default=42)
p.add_argument('--no-split',action='store_true',help='Convert all images without the default grouped 90%% train / 10%% val split; capture_group is optional')
p.add_argument('--validation-images',help='JSON list of held-out image filenames; explicitly allows same-capture image validation without an independence claim')
a=p.parse_args()
if a.no_split and a.validation_images:p.error('--no-split and --validation-images are mutually exclusive')
if a.validate_only:
    r=validate(a.coco,a.images,a.catalog,a.sidecar,require_capture_group=not a.no_split and not a.validation_images);r.pop('records');print(json.dumps(r,indent=2));raise SystemExit(0 if r['valid'] else 1)
if not a.output:p.error('--output required')
validation_images=json.loads(open(a.validation_images).read()) if a.validation_images else None
print(json.dumps(prepare(a.coco,a.images,a.catalog,a.output,a.sidecar,a.seed,no_split=a.no_split,validation_images=validation_images),indent=2))
