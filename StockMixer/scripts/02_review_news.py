"""Config-driven news preparation; see docs/NEWS_PIPELINE.md."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from news_fusion.pipeline import main

if __name__ == "__main__":
    main(["--step", "review", *sys.argv[1:]])
    main(["--step", "publish", *[a for a in sys.argv[1:] if a != "--new"]])
