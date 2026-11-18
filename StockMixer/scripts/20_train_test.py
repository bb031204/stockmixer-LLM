"""Stage 3: train and test according to config; predict using the selected checkpoint."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news_fusion.launcher import main

if __name__ == '__main__':
    main(default_step='train')
