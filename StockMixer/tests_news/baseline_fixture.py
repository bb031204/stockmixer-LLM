"""Baseline fixtures explicitly disable news, independent of the public switch."""
from pathlib import Path
from news_fusion.configuration import compile_config, read_yaml
from news_fusion.protocol import PROJECT, configuration as runtime_configuration


def configuration(path=None, dataset=None, project=None):
    if path is not None:
        return runtime_configuration(path, dataset, project)
    root = Path(project or PROJECT).resolve()
    source = root/'configs/stockmixer.yaml'
    raw = read_yaml(source)
    raw['llm']['enabled'] = False
    return root, compile_config(root, source, raw, dataset)
