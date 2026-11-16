"""Single public training entry: settings live in configs/stockmixer.yaml."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from news_fusion.workflow import main

if __name__=='__main__':
    main()
