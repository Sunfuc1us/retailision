"""Run existing offline model checks with access to a supplied legacy root denied."""
import _bootstrap
import argparse,json,runpy,sys
from pathlib import Path
from project_paths import PROJECT_ROOT
p=argparse.ArgumentParser();p.add_argument('--forbidden-root',required=True);a=p.parse_args();legacy=str(Path(a.forbidden_root).resolve())
assert all(legacy not in x for x in sys.path)
def audit(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        name=args[0].decode() if isinstance(args[0],bytes) else args[0]
        if legacy in name:raise RuntimeError('legacy project access denied during migration verification')
sys.addaudithook(audit)
runpy.run_path(str(PROJECT_ROOT/'tools/verify_models.py'),run_name='__main__')
origins={name:str(getattr(module,'__file__','')) for name,module in list(sys.modules.items()) if name.split('.')[0] in ('backend','inference','catalog','dataset','registry','mmdet','dinov2','project_paths')}
assert origins and all(legacy not in name for name in origins.values())
Path('results/migration-independence.json').write_text(json.dumps(dict(project_root=str(PROJECT_ROOT),legacy_file_access_denied=True,module_origins=origins),indent=2))
