"""Publish validated weights, gallery and policy together to the Workbench."""
import _bootstrap
import argparse
import json
from retailvision.artifacts.publish import publish

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment', required=True, help='Completed joint experiment directory')
parser.add_argument('--version', required=True)
parser.add_argument('--policy', required=True, help='Explicit version-matched calibrated or provisional policy JSON')
parser.add_argument('--name')
parser.add_argument('--dataset', help='Required only for historical experiments without a saved dataset path')
args = parser.parse_args()
print(json.dumps(publish(**vars(args)), ensure_ascii=False, indent=2))
