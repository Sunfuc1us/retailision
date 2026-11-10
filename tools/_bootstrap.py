import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from project_paths import activate_project_root
activate_project_root()
