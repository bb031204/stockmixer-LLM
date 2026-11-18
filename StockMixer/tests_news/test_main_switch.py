import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml
from news_fusion.pipeline import main
from news_fusion.protocol import PROJECT


class MainSwitchTests(unittest.TestCase):
    def test_disabled_main_runs_baseline_without_loading_news_config(self):
        config=yaml.safe_load((PROJECT/'configs/stockmixer.yaml').read_text(encoding='utf-8-sig'))
        config['llm']={'enabled':False,'config':'missing_news_config.yaml'}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stockmixer.yaml';path.write_text(yaml.safe_dump(config),encoding='utf-8')
            with patch('news_fusion.workflow.run_workflow',return_value={'status':'completed'}) as run,\
                 patch('news_fusion.configuration.load_llm_config',side_effect=AssertionError('LLM config must stay unused')),\
                 patch('news_fusion.pipeline.Pipeline',side_effect=AssertionError('No news preprocessing while disabled')):
                main(['--config',str(path)])
            run.assert_called_once_with(path,PROJECT)

    def test_disabled_explicit_train_also_respects_public_switch(self):
        config=yaml.safe_load((PROJECT/'configs/stockmixer.yaml').read_text(encoding='utf-8-sig'))
        config['llm']['enabled']=False
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stockmixer.yaml';path.write_text(yaml.safe_dump(config),encoding='utf-8')
            with patch('news_fusion.workflow.run_workflow',return_value={'status':'completed'}) as run:
                main(['--config',str(path),'--step','train'])
            run.assert_called_once_with(path,PROJECT)
