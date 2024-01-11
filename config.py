from pathlib import Path
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



DATA_DIR = Path(__file__).parent / 'data'
RUNS_DIR = DATA_DIR / 'runs'
DATASETS_DIR = DATA_DIR / 'datasets'
TMP_DIR = Path('/work/rathjjgf/tmp') if torch.cuda.is_available() else DATA_DIR / 'tmp'

PROJECT = 'tns/salvador-seahorse'
FILE_DIR = Path(__file__).parent / 'files'
