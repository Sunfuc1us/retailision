"""Project paths are anchored to this checkout, independent of the launch directory."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path

def activate_project_root() -> None:
    """CLI and local API entry points use project-relative arguments and outputs."""
    os.chdir(PROJECT_ROOT)
