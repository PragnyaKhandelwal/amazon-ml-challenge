"""Paths and hyper-parameters (override the data / work dirs with env vars)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("ER_DATA_DIR", os.path.join(ROOT, "dataset"))
WORK_DIR = os.environ.get("ER_WORK_DIR", os.path.join(ROOT, "work"))
OUT_DIR = os.environ.get("ER_OUT_DIR", os.path.join(ROOT, "output"))
WORKERS = int(os.environ.get("ER_WORKERS", "4"))
SEED = 42

# validation: S1 entities whose id hash falls in this bucket are held out from every
# trained component (transliteration table, matcher, threshold tuning uses OOF only)
N_FOLDS = 5
VALID_FOLD = 0

os.makedirs(WORK_DIR, exist_ok=True)
