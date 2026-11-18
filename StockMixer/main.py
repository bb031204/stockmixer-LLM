"""Config-driven retrace import, offline preparation, StockMixer training, test and prediction."""
from news_fusion.launcher import main

if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError, KeyError) as error:
        raise SystemExit('StockMixer全流程停止：' + str(error)) from error
