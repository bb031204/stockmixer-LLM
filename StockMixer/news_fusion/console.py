"""Console layout and progress bar matching the sibling StockMixer project."""
import re
import shutil
import sys
import time
from datetime import datetime,timezone,timedelta
from decimal import Decimal
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT=Path(__file__).resolve().parents[1]
ANSI_RE=re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

class SimpleProgress:
    def __init__(self, iterable, total=None, desc="", log_every=10, min_interval=0.15):
        self.iterable = iterable
        self.total = total if total is not None else len(iterable)
        self.desc = desc
        self.log_every = max(1, log_every)
        self.min_interval = min_interval
        self.start_time = None
        self.last_render_time = 0.0
        self.last_line_len = 0
        self.postfix = ""
        self.stream = sys.__stdout__
        self.enabled = bool(getattr(self.stream, "isatty", lambda: False)())

    def set_postfix(self, **kwargs):
        if not kwargs:
            self.postfix = ""
            return
        self.postfix = " " + " ".join(f"{key}={value}" for key, value in kwargs.items())

    @staticmethod
    def terminal_columns():
        return max(64, shutil.get_terminal_size(fallback=(100, 20)).columns)

    @staticmethod
    def is_encodable(text, stream):
        encoding = getattr(stream, "encoding", None) or sys.getdefaultencoding()
        try:
            text.encode(encoding)
        except UnicodeEncodeError:
            return False
        return True

    @classmethod
    def make_bar(cls, current, total, width=28):
        ratio = current / max(1, total)
        filled = int(round(ratio * width))
        stream = sys.__stdout__
        fill, empty = ("█", "─")
        if not cls.is_encodable(fill + empty, stream):
            fill, empty = ("=", "-")
        return fill * filled + empty * (width - filled)

    def __iter__(self):
        self.start_time = time.time()
        for index, item in enumerate(self.iterable, start=1):
            yield item
            now = time.time()
            should_render = (
                index == 1
                or index == self.total
                or (index % self.log_every == 0 and now - self.last_render_time >= self.min_interval)
            )
            if should_render:
                self.render(index)
        if self.total == 0:
            self.render(0)
        self.finish_line()

    def render(self, index):
        if not self.enabled:
            return
        elapsed = time.time() - self.start_time
        rate = elapsed / max(1, index)
        remaining = rate * max(0, self.total - index)
        pct = index / max(1, self.total) * 100.0
        columns = self.terminal_columns()
        counter_width = len(str(max(1, self.total)))
        desc = self.desc.replace("Epoch", "Ep").replace("Train", "Trn")
        pct_text = f"{pct:5.1f}%"
        counter = f"{index:>{counter_width}}/{self.total}"
        timing = f"{format_duration(elapsed)}<{format_duration(remaining)}"
        fixed = f"{desc:<11} {pct_text} || {counter} {timing}{self.postfix}"
        bar_width = max(10, min(28, columns - len(fixed) - 2))
        bar = self.make_bar(index, self.total, width=bar_width)
        line = f"{desc:<11} {pct_text} |{bar}| {counter} {timing}{self.postfix}"
        max_len = max(1, columns - 1)
        if len(line) > max_len:
            line = line[:max_len]
        self.stream.write("\r\x1b[2K" + line)
        self.stream.flush()
        self.last_line_len = len(line)
        self.last_render_time = time.time()

    def finish_line(self):
        if not self.enabled or not self.last_line_len:
            return
        self.stream.write("\r\x1b[2K")
        self.stream.flush()
        self.last_line_len = 0


def progress_iter(iterable, total=None, desc=""):
    total = total if total is not None else len(iterable)
    log_every = max(1, total // 100)
    return SimpleProgress(iterable, total=total, desc=desc, log_every=log_every)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def format_number(value, digits=8):
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "N/A"
    text = f"{value:.{digits}f}"
    if value!=0 and float(text)==0:
        text=format(Decimal(format(value,f".{max(1,digits)}g")),"f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text


def format_metric(value, digits=3):
    text = format_number(value, digits)
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def format_loss(value):
    return format_number(value, 6)


def color_text(text, color):
    stream = sys.__stdout__
    if not getattr(stream, "isatty", lambda: False)():
        return text
    colors = {
        "green": "32",
        "bright_green": "92",
    }
    code = colors.get(color)
    if not code:
        return text
    return f"\033[{code}m{text}\033[0m"


def terminal_rule(char="-"):
    width = min(100, max(80, shutil.get_terminal_size(fallback=(100, 20)).columns))
    return char * width


def print_header(title):
    print("\n" + terminal_rule("="))
    print(title)
    print(terminal_rule("="))


def print_kv_line(title, items):
    formatted = "  ".join(f"{key}={value}" for key, value in items)
    print(f"[{title:<6}] {formatted}")


def compact_epoch_summary(epoch, total_epochs, train_loss, val_loss, val_perf, test_perf,
                          epoch_time, estimated_remaining, best_epoch, selection_metric, selection_current):
    best_marker = color_text("BEST MODEL", "bright_green") if epoch == best_epoch else "          "
    return (
        f"{best_marker} "
        f"Epoch={epoch:03d}/{total_epochs} "
        f"TrainLoss={format_loss(train_loss)} "
        f"ValLoss={format_loss(val_loss)} "
        f"ValIC={format_metric(val_perf['IC'])} "
        f"ValRIC={format_metric(val_perf['RIC'])} "
        f"ValPrec@10={format_metric(val_perf['prec_10'])} "
        f"ValSR={format_metric(val_perf['sharpe5'])} "
        f"TestIC={format_metric(test_perf['IC'])} "
        f"TestRIC={format_metric(test_perf['RIC'])} "
        f"TestSR={format_metric(test_perf['sharpe5'])} "
        f"Select={selection_metric}:{format_metric(selection_current)} "
        f"EpochTime={format_duration(epoch_time)} "
        f"ETA={format_duration(estimated_remaining)} "
        f"BestEpoch={best_epoch:03d}"
    )



def display_path(path):
    path=Path(path)
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def print_training_header(project,output,data,config,model,device,fusion=False):
    settings=config['training']
    original=getattr(data,'protocol','strict')=='original_reproduction'
    candidates=data.split.targets('train')
    steps=data.split.samples_per_epoch if original else len(candidates)
    print_header('StockMixer Training'+(' + Market News Fusion' if fusion else ''))
    print_kv_line('RUN',[('time',datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')),('output',display_path(output))])
    print_kv_line('ENV',[('Python',sys.version.split()[0]),('PyTorch',torch.__version__),('CUDA',torch.cuda.is_available()),
        ('GPU',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'),('device',device)])
    print_kv_line('SEED',[('numpy',settings['numpy_seed']),('torch',settings['torch_seed'])])
    shape=data.arrays['eod'].shape
    print_kv_line('DATA',[('market',data.axis.dataset),('dataset',display_path(data.directory)),
        ('stock_num',len(data.axis.stock_uids)),('actual_stocks',shape[0]),('fea_num',5),('actual_features',shape[2]),('trade_dates',len(data.axis.dates))])
    print_kv_line('SPLIT',[('train target candidates',str(int(candidates[0]))+'-'+str(int(candidates[-1]))),
        ('val target',str(data.axis.train_end)+'-'+str(data.axis.valid_end-1)),('test target',str(data.axis.valid_end)+'-'+str(len(data.axis.dates)-1))])
    print_kv_line('WINDOW',[('lookback',data.split.lookback),('steps',data.split.steps),('train_candidates',len(candidates)),
        ('train_steps',steps),('val_offsets',data.axis.valid_end-data.axis.train_end),('test_offsets',len(data.axis.dates)-data.axis.valid_end)])
    print_kv_line('TRAIN',[('epochs',settings['epochs']),('lr',format_number(settings['learning_rate'])),
        ('alpha',format_number(settings['alpha'])),('market_num',data.axis.source.get('market_num','N/A')),
        ('activation_tag','GELU'),('selection','val_loss (min)'),('protocol',getattr(data,'protocol','strict'))])
    print_kv_line('SCALES',[('branch_lengths',[16,16,8]),('temporal_dim',40)])
    print_kv_line('PARAM',[('total',sum(p.numel() for p in model.parameters())),('trainable',sum(p.numel() for p in model.parameters() if p.requires_grad))])
    print(terminal_rule('-'),flush=True)


def print_training_footer(output,best_epoch,test,total_time,epoch_times):
    output=Path(output)
    print(terminal_rule('-'))
    print(f"Final Test  best={best_epoch:03d} loss={format_loss(test['loss'])} mse={format_loss(test['mse_loss'])} rank={format_loss(test['rank_loss'])}")
    print(f"Metrics     IC={format_metric(test['IC'])} RIC={format_metric(test['RIC'])} P10={format_metric(test['prec_10'])} SR={format_metric(test['sharpe5'])}")
    print_kv_line('OUTPUT',[('total_time',format_duration(total_time)),('avg_epoch',format_duration(float(np.mean(epoch_times))) if epoch_times else 'N/A')])
    for label,path in (('best_model',output/'checkpoints/best_model.pt'),('history',output/'tables/history.csv'),('metrics',output/'tables/metrics.json'),('figures',output/'figures')):
        print(f"[FILE  ] {label}={display_path(path)}")
    print(terminal_rule('='),flush=True)
