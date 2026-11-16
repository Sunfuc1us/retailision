import logging
from pathlib import Path

def configure_logging(path: str) -> None:
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(p)],force=True)
