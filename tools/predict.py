import _bootstrap
import argparse,json
from PIL import Image
from inference.factory import load_pipeline
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--image',required=True);p.add_argument('--target',required=True)
a=p.parse_args();print(json.dumps(load_pipeline(a.config).predict(Image.open(a.image).convert('RGB'),a.target),indent=2,ensure_ascii=False))
