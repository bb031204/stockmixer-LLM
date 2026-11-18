"""Install optional NLP dependencies without changing the price-model environment."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import ssl
from .io import digest_file, now, write_json

PACKAGES = ['transformers==4.44.2', 'huggingface-hub==0.25.2',
            'tokenizers==0.19.1', 'safetensors==0.4.5', 'pyarrow==17.0.0',
            'exchange-calendars==4.5.5']
CORE = ('torch', 'numpy', 'pandas', 'PyYAML')
FILES = ('config.json', 'model.safetensors', 'tokenizer_config.json',
         'special_tokens_map.json', 'vocab.txt', 'README.md')


def setup_environment(settings):
    expected = Path(settings['python']).resolve()
    if Path(sys.executable).resolve() != expected:
        raise ValueError('Use the configured interpreter: ' + str(expected))
    root = Path(sys.prefix).resolve()
    model = Path(settings['model_directory']).resolve()
    if root not in model.parents:
        raise ValueError('The model directory must be inside the requested environment')
    cache = root / 'cache' / 'stockmixer_news'
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PIP_CACHE_DIR=str(cache / 'pip'), HF_HOME=str(cache / 'huggingface'))
    before = {name: metadata.version(name) for name in CORE}
    constraints = cache / 'protected-core.txt'
    constraints.write_text(''.join(name + '==' + version + '\n' for name, version in before.items()), encoding='utf-8')
    complete=True
    for pinned in PACKAGES:
        name,version=pinned.split('==')
        try:complete=complete and metadata.version(name)==version
        except metadata.PackageNotFoundError:complete=False
    if not complete:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
                        '--constraint', str(constraints), *PACKAGES], env=env, check=True)
    after = {name: metadata.version(name) for name in CORE}
    if before != after:
        raise RuntimeError('Protected core packages changed')
    revision = settings['model_revision']
    import re
    if re.fullmatch('[a-f0-9]{40}', revision) is None:
        raise ValueError('Pin the full official FinBERT commit SHA')
    previous = model / 'download_manifest.json'
    old = json.loads(previous.read_text(encoding='utf-8')) if previous.exists() else {}
    if old.get('revision')==revision and all((model/name).is_file() and digest_file(model/name)==old.get('files',{}).get(name) for name in FILES):
        print('[SETUP] local environment and pinned model verified; no network download required',flush=True)
        return dict(old,verified_at=now(),protected_packages=after)
    url = 'https://huggingface.co/api/models/ProsusAI/finbert/revision/' + revision + '?blobs=true'
    import certifi
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, timeout=60, context=context) as response:
        info = json.load(response)
    if info['sha'] != revision:
        raise ValueError('Unexpected model revision')
    siblings = {r['rfilename']: r for r in info['siblings']}
    model.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = model / name
        expected_sha = siblings[name].get('lfs', {}).get('sha256')
        if target.exists():
            actual = digest_file(target)
            if actual == (expected_sha or old.get('files', {}).get(name)):
                continue
        download = 'https://huggingface.co/ProsusAI/finbert/resolve/' + revision + '/' + name
        partial = model / (name + '.partial')
        start = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(download, headers={'Range': 'bytes=' + str(start) + '-'} if start else {})
        print('[DOWNLOAD] ' + name, flush=True)
        with urllib.request.urlopen(request, timeout=120, context=context) as response:
            append = start > 0 and response.status == 206
            with partial.open('ab' if append else 'wb') as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
        if expected_sha and digest_file(partial) != expected_sha:
            raise ValueError('Downloaded model hash mismatch: ' + name)
        partial.replace(target)
    available = old.get('model_available_at') if old.get('revision') == revision else None
    manifest = {'revision': revision, 'model_available_at': available or now(),
                'source': url, 'license': info.get('cardData', {}).get('license'),
                'files': {name: digest_file(model / name) for name in FILES},
                'python': sys.executable, 'protected_packages': after,
                'packages': {p.split('==')[0]: metadata.version(p.split('==')[0]) for p in PACKAGES}}
    write_json(previous, manifest)
    return manifest
