"""Environment configuration; credentials remain external and unlogged."""
import os
from pathlib import Path
from dotenv import load_dotenv

def configure():
    root=Path(__file__).resolve().parents[1]
    external=os.getenv('FINFORENSICS_ENV_FILE')
    if external:
        path=Path(external).expanduser()
        if not path.is_file():raise FileNotFoundError('Configured external environment file not found')
        load_dotenv(path,override=False)
    load_dotenv(root/'.env',override=False)
    model_cache=os.getenv('FINFORENSICS_MODEL_CACHE')
    if model_cache:os.environ.setdefault('HF_HOME',str(Path(model_cache).expanduser()))
