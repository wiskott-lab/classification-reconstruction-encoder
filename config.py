from pathlib import Path
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(__file__).parent / 'data'
DATASETS_DIR = DATA_DIR / 'datasets'
