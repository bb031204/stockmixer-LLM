"""Stage 2: FinBERT, mapping, panels, final checks, then reuse/train the matching E0."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news_fusion.launcher import main

if __name__ == '__main__':
    main(default_step='prepare')
