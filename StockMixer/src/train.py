import argparse
import csv
import json
import os
import pickle
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as torch
import yaml

from evaluator import evaluate
from load_data import load_EOD_data
from model import get_loss, StockMixer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            output = data if getattr(stream, "isatty", lambda: False)() else ANSI_RE.sub("", data)
            stream.write(output)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


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


def display_path(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def print_header(title):
    print("\n" + terminal_rule("="))
    print(title)
    print(terminal_rule("="))


def print_kv_line(title, items):
    formatted = "  ".join(f"{key}={value}" for key, value in items)
    print(f"[{title:<6}] {formatted}")


SELECTION_METRICS = {
    "val_loss": ("min", "Validation total loss"),
    "val_mse_loss": ("min", "Validation MSE loss component"),
    "val_rank_loss": ("min", "Validation ranking loss component"),
    "val_mse": ("min", "Validation evaluator MSE"),
    "val_ic": ("max", "Validation IC"),
    "val_ric": ("max", "Validation RIC"),
    "val_prec_10": ("max", "Validation Prec@10"),
    "val_sharpe5": ("max", "Validation Sharpe@5"),
}


def selection_value(metric_name, val_loss, val_reg_loss, val_rank_loss, val_perf):
    values = {
        "val_loss": val_loss,
        "val_mse_loss": val_reg_loss,
        "val_rank_loss": val_rank_loss,
        "val_mse": val_perf["mse"],
        "val_ic": val_perf["IC"],
        "val_ric": val_perf["RIC"],
        "val_prec_10": val_perf["prec_10"],
        "val_sharpe5": val_perf["sharpe5"],
    }
    return values[metric_name]


def is_better_selection(value, best_value, mode):
    if not np.isfinite(value):
        return False
    if best_value is None:
        return True
    if mode == "min":
        return value < best_value
    return value > best_value


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


def default_config():
    return {
        "data": {
            "market_name": "NASDAQ",
            "dataset_root": "dataset",
            "stock_num": 1026,
            "fea_num": 5,
            "lookback_length": 16,
            "steps": 1,
            "valid_index": 756,
            "test_index": 1008,
        },
        "model": {
            "market_num": 20,
            "scale_factor": 3,
            "activation": "GELU",
        },
        "training": {
            "epochs": 100,
            "learning_rate": 0.001,
            "alpha": 0.1,
            "numpy_seed": 123456789,
            "torch_seed": 12345678,
            "device": "cuda",
            "selection_metric": "val_loss",
        },
        "output": {
            "save_checkpoints": True,
            "save_history": True,
            "save_figures": True,
        },
    }


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_args():
    parser = argparse.ArgumentParser(description="Train official StockMixer with config/output wrapper.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"), help="YAML config path.")
    return parser.parse_args()


def load_config(path):
    config = default_config()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a YAML mapping: {config_path}")
        deep_update(config, loaded)
    config["config_path"] = str(config_path)
    return config


def create_run_dir(market_name):
    outputs_root = PROJECT_ROOT / "outputs"
    now = datetime.now()
    safe_market = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(market_name)).upper()
    base = outputs_root / f"{now.strftime('%Y%m%d_%H%M%S')}_{safe_market}"
    run_dir = base
    suffix = 1
    while run_dir.exists():
        run_dir = base.with_name(f"{base.name}_{suffix:02d}")
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    for dirname in ["logs", "configs", "checkpoints", "tables", "figures"]:
        (run_dir / dirname).mkdir()
    return run_dir


def save_config(config, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def save_history(path, history):
    if not history:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(val) for val in value]
    if isinstance(value, tuple):
        return [to_jsonable(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_checkpoint(path, model, optimizer, epoch, best_valid_loss, config):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
        "config": config,
    }, path)


def plot_history(run_dir, history, enabled):
    if not enabled or not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN  ] matplotlib unavailable, skip figures: {exc}")
        return

    figures_dir = run_dir / "figures"
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train Loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="Validation Loss")
    plt.plot(epochs, [row["test_loss"] for row in history], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [row["val_ic"] for row in history], label="Validation IC")
    plt.plot(epochs, [row["test_ic"] for row in history], label="Test IC")
    plt.plot(epochs, [row["val_ric"] for row in history], label="Validation RIC")
    plt.plot(epochs, [row["test_ric"] for row in history], label="Test RIC")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "ic_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [row["val_sharpe5"] for row in history], label="Validation SR")
    plt.plot(epochs, [row["test_sharpe5"] for row in history], label="Test SR")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "sr_curve.png", dpi=150)
    plt.close()


def parameter_counts(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def select_device(name):
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def validate_dataset_files(dataset_path, market_name):
    dataset_path = Path(dataset_path)
    filenames = ["SP500.npy"] if market_name == "SP500" else [
        "eod_data.pkl", "mask_data.pkl", "gt_data.pkl", "price_data.pkl"
    ]
    invalid = []
    for filename in filenames:
        path = dataset_path / filename
        if not path.is_file():
            invalid.append(f"  - {path}: missing file")
        elif path.stat().st_size == 0:
            invalid.append(f"  - {path}: empty file (0 bytes)")
    if invalid:
        raise ValueError(
            f"{market_name} dataset is unavailable:\n" + "\n".join(invalid)
            + "\nPrepare the actual dataset before training; changing hyperparameters will not fix missing data."
            + ("\nNASDAQ/NYSE source: https://github.com/fulifeng/Temporal_Relational_Stock_Ranking"
               "\nUse the processed CSV data with load_EOD_data() in src/load_data.py and save its four arrays as pickle files."
               if market_name != "SP500" else "")
        )


def main():
    args = parse_args()
    config = load_config(args.config)

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]

    market_name = data_config["market_name"]
    validate_dataset_files(PROJECT_ROOT / data_config["dataset_root"] / market_name, market_name)
    run_dir = create_run_dir(market_name)
    logs_dir = run_dir / "logs"
    configs_dir = run_dir / "configs"
    checkpoints_dir = run_dir / "checkpoints"
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"

    log_file = open(logs_dir / "train.log", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    start_time = time.time()
    print_header("Official StockMixer Training")
    print_kv_line("RUN", [
        ("time", datetime.now().isoformat(timespec="seconds")),
        ("output", display_path(run_dir)),
    ])

    np.random.seed(training_config["numpy_seed"])
    torch.random.manual_seed(training_config["torch_seed"])
    device = select_device(training_config["device"])
    print_kv_line("ENV", [
        ("Python", sys.version.split()[0]),
        ("PyTorch", torch.__version__),
        ("CUDA", torch.cuda.is_available()),
        ("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"),
        ("device", device),
    ])
    print_kv_line("SEED", [
        ("numpy", training_config["numpy_seed"]),
        ("torch", training_config["torch_seed"]),
    ])
    save_config(config, configs_dir / "config.yaml")

    data_path = str(PROJECT_ROOT / data_config["dataset_root"])
    relation_name = "wikidata"
    stock_num = int(data_config["stock_num"])
    lookback_length = int(data_config["lookback_length"])
    epochs = int(training_config["epochs"])
    valid_index = int(data_config["valid_index"])
    test_index = int(data_config["test_index"])
    fea_num = int(data_config["fea_num"])
    market_num = int(model_config["market_num"])
    steps = int(data_config["steps"])
    learning_rate = float(training_config["learning_rate"])
    alpha = float(training_config["alpha"])
    selection_metric = training_config.get("selection_metric", "val_loss")
    if selection_metric not in SELECTION_METRICS:
        allowed = ", ".join(SELECTION_METRICS)
        raise ValueError(f"Unknown selection_metric={selection_metric!r}. Allowed: {allowed}")
    selection_mode, selection_description = SELECTION_METRICS[selection_metric]
    scale_factor = int(model_config["scale_factor"])
    activation = model_config["activation"]

    dataset_path = os.path.join(data_path, market_name)
    if market_name == "SP500":
        data = np.load(os.path.join(data_path, "SP500", "SP500.npy"))
        data = data[:, 915:, :]
        price_data = data[:, :, -1]
        mask_data = np.ones((data.shape[0], data.shape[1]))
        eod_data = data
        gt_data = np.zeros((data.shape[0], data.shape[1]))
        for ticket in range(0, data.shape[0]):
            for row in range(1, data.shape[1]):
                gt_data[ticket][row] = (data[ticket][row][-1] - data[ticket][row - steps][-1]) / \
                                       data[ticket][row - steps][-1]
    else:
        with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
            eod_data = pickle.load(f)
        with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
            mask_data = pickle.load(f)
        with open(os.path.join(dataset_path, "gt_data.pkl"), "rb") as f:
            gt_data = pickle.load(f)
        with open(os.path.join(dataset_path, "price_data.pkl"), "rb") as f:
            price_data = pickle.load(f)

    trade_dates = mask_data.shape[1]
    print_kv_line("DATA", [
        ("market", market_name),
        ("dataset", display_path(dataset_path)),
        ("stock_num", stock_num),
        ("actual_stocks", eod_data.shape[0]),
        ("fea_num", fea_num),
        ("actual_features", eod_data.shape[2]),
        ("trade_dates", trade_dates),
    ])
    print_kv_line("SPLIT", [
        ("train target candidates", f"{lookback_length + steps - 1}-{valid_index + lookback_length + steps - 2}"),
        ("val target", f"{valid_index}-{test_index - 1}"),
        ("test target", f"{test_index}-{trade_dates - 1}"),
    ])
    print_kv_line("WINDOW", [
        ("lookback", lookback_length),
        ("steps", steps),
        ("train_candidates", valid_index),
        ("train_steps", valid_index - lookback_length - steps + 1),
        ("val_offsets", test_index - valid_index),
        ("test_offsets", trade_dates - test_index),
    ])
    print_kv_line("TRAIN", [
        ("epochs", epochs),
        ("lr", format_number(learning_rate)),
        ("alpha", format_number(alpha)),
        ("market_num", market_num),
        ("scale", scale_factor),
        ("activation", activation),
        ("selection", f"{selection_metric} ({selection_mode})"),
    ])

    if eod_data.shape[0] != stock_num:
        raise ValueError(f"stock_num={stock_num} does not match loaded data stocks={eod_data.shape[0]}.")
    if eod_data.shape[2] != fea_num:
        raise ValueError(f"fea_num={fea_num} does not match loaded data features={eod_data.shape[2]}.")

    model = StockMixer(
        stocks=stock_num,
        time_steps=lookback_length,
        channels=fea_num,
        market=market_num,
        scale=scale_factor
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_params, trainable_params = parameter_counts(model)
    print_kv_line("PARAM", [
        ("total", total_params),
        ("trainable", trainable_params),
    ])
    (configs_dir / "model_summary.txt").write_text(str(model), encoding="utf-8")

    best_valid_loss = np.inf
    best_selection_value = None
    best_valid_perf = None
    best_test_perf = None
    best_epoch = 0
    best_test_loss = None
    best_test_reg_loss = None
    best_test_rank_loss = None
    # Match the original script: shuffle all candidates, then use train_steps entries.
    batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

    def validate(start_index, end_index):
        with torch.no_grad():
            cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
            cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
            loss = 0.
            reg_loss = 0.
            rank_loss = 0.
            for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
                data_batch, mask_batch, price_batch, gt_batch = map(

                    lambda x: torch.Tensor(x).to(device),
                    get_batch(cur_offset)
                )
                prediction = model(data_batch)
                cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                         stock_num, alpha)
                loss += cur_loss.item()
                reg_loss += cur_reg_loss.item()
                rank_loss += cur_rank_loss.item()
                cur_valid_pred[:, cur_offset - (start_index - lookback_length - steps + 1)] = cur_rr[:, 0].cpu()
                cur_valid_gt[:, cur_offset - (start_index - lookback_length - steps + 1)] = gt_batch[:, 0].cpu()
                cur_valid_mask[:, cur_offset - (start_index - lookback_length - steps + 1)] = mask_batch[:, 0].cpu()
            loss = loss / (end_index - start_index)
            reg_loss = reg_loss / (end_index - start_index)
            rank_loss = rank_loss / (end_index - start_index)
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
        return loss, reg_loss, rank_loss, cur_valid_perf

    def get_batch(offset=None):
        if offset is None:
            offset = random.randrange(0, valid_index)
        seq_len = lookback_length
        mask_batch = mask_data[:, offset: offset + seq_len + steps]
        mask_batch = np.min(mask_batch, axis=1)
        return (
            eod_data[:, offset:offset + seq_len, :],
            np.expand_dims(mask_batch, axis=1),
            np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
            np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1))

    history = []
    epoch_times = []
    print(terminal_rule("-"))

    for epoch in range(epochs):
        epoch_start = time.time()
        np.random.shuffle(batch_offsets)
        tra_loss = 0.0
        tra_reg_loss = 0.0
        tra_rank_loss = 0.0
        train_steps = valid_index - lookback_length - steps + 1
        train_bar = progress_iter(range(train_steps), total=train_steps, desc=f"Epoch {epoch + 1:03d} Train")
        for j in train_bar:
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(batch_offsets[j])
            )
            optimizer.zero_grad()
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                stock_num, alpha)
            cur_loss = cur_loss
            cur_loss.backward()
            optimizer.step()

            tra_loss += cur_loss.item()
            tra_reg_loss += cur_reg_loss.item()
            tra_rank_loss += cur_rank_loss.item()
            if hasattr(train_bar, "set_postfix"):
                train_bar.set_postfix(loss=format_loss(tra_loss / (j + 1)), rank=format_loss(tra_rank_loss / (j + 1)))
        tra_loss = tra_loss / train_steps
        tra_reg_loss = tra_reg_loss / train_steps
        tra_rank_loss = tra_rank_loss / train_steps

        val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
        test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)

        current_selection_value = selection_value(selection_metric, val_loss, val_reg_loss, val_rank_loss, val_perf)
        if is_better_selection(current_selection_value, best_selection_value, selection_mode):
            best_valid_loss = val_loss
            best_selection_value = current_selection_value
            best_valid_perf = val_perf
            best_test_perf = test_perf
            best_epoch = epoch + 1
            best_test_loss = test_loss
            best_test_reg_loss = test_reg_loss
            best_test_rank_loss = test_rank_loss
            if output_config.get("save_checkpoints", True):
                save_checkpoint(checkpoints_dir / "best_model.pt", model, optimizer, best_epoch, best_valid_loss, config)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        estimated_remaining = float(np.mean(epoch_times)) * max(0, epochs - (epoch + 1))
        history_row = {
            "epoch": epoch + 1,
            "train_loss": tra_loss,
            "train_mse_loss": tra_reg_loss,
            "train_rank_loss": tra_rank_loss,
            "val_loss": val_loss,
            "val_mse_loss": val_reg_loss,
            "val_rank_loss": val_rank_loss,
            "val_mse": val_perf["mse"],
            "val_ic": val_perf["IC"],
            "val_ric": val_perf["RIC"],
            "val_prec_10": val_perf["prec_10"],
            "val_sharpe5": val_perf["sharpe5"],
            "test_loss": test_loss,
            "test_mse_loss": test_reg_loss,
            "test_rank_loss": test_rank_loss,
            "test_mse": test_perf["mse"],
            "test_ic": test_perf["IC"],
            "test_ric": test_perf["RIC"],
            "test_prec_10": test_perf["prec_10"],
            "test_sharpe5": test_perf["sharpe5"],
            "epoch_time_sec": epoch_time,
        }
        history.append(history_row)
        if output_config.get("save_history", True):
            save_history(tables_dir / "history.csv", history)

        print(compact_epoch_summary(
            epoch=epoch + 1,
            total_epochs=epochs,
            train_loss=tra_loss,
            val_loss=val_loss,
            val_perf=val_perf,
            test_perf=test_perf,
            epoch_time=epoch_time,
            estimated_remaining=estimated_remaining,
            best_epoch=best_epoch,
            selection_metric=selection_metric,
            selection_current=current_selection_value,
        ))

    if output_config.get("save_checkpoints", True):
        save_checkpoint(checkpoints_dir / "last_model.pt", model, optimizer, epochs, best_valid_loss, config)

    metrics = {
        "selection_logic": f"stockmixer: record val_perf and test_perf when {selection_metric} improves",
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "selection_description": selection_description,
        "best_selection_value": best_selection_value,
        "best_epoch": best_epoch,
        "best_validation_loss": best_valid_loss,
        "best_valid_perf": best_valid_perf,
        "best_test_perf": best_test_perf,
        "test_loss": best_test_loss,
        "test_mse_loss_component": best_test_reg_loss,
        "test_rank_loss_component": best_test_rank_loss,
        "test_metrics": best_test_perf,
    }
    (tables_dir / "metrics.json").write_text(json.dumps(to_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
    plot_history(run_dir, history, output_config.get("save_figures", True))

    total_time = time.time() - start_time
    avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else np.nan
    print(terminal_rule("-"))
    print(
        f"Final Test  best={best_epoch:03d} "
        f"loss={format_loss(best_test_loss)} "
        f"mse={format_loss(best_test_reg_loss)} "
        f"rank={format_loss(best_test_rank_loss)}"
    )
    print(
        f"Metrics     IC={format_metric(best_test_perf['IC'])} "
        f"RIC={format_metric(best_test_perf['RIC'])} "
        f"P10={format_metric(best_test_perf['prec_10'])} "
        f"SR={format_metric(best_test_perf['sharpe5'])}"
    )
    print_kv_line("OUTPUT", [
        ("total_time", format_duration(total_time)),
        ("avg_epoch", format_duration(avg_epoch_time) if np.isfinite(avg_epoch_time) else "N/A"),
    ])
    print(f"[FILE  ] best_model={display_path(checkpoints_dir / 'best_model.pt')}")
    print(f"[FILE  ] history={display_path(tables_dir / 'history.csv')}")
    print(f"[FILE  ] metrics={display_path(tables_dir / 'metrics.json')}")
    print(f"[FILE  ] figures={display_path(figures_dir)}")
    print(terminal_rule("="))

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()


if __name__ == "__main__":
    main()
