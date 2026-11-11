"""Build the NYSE pickle files with the project's unchanged load_EOD_data()."""

import argparse
import hashlib
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from load_data import load_EOD_data


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inherit_directory_permissions(path):
    if os.name == "nt":
        # A same-volume rename keeps the staging file's restrictive Windows ACL.
        # Restore the destination directory's inherited ACL for normal user access.
        subprocess.run(["icacls", str(path), "/reset"], check=True, capture_output=True)


def build(source_root):
    data_root = source_root / "data"
    csv_root = data_root / "2013-01-01"
    ticker_file = data_root / "NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv"
    date_file = data_root / "NYSE_aver_line_dates.csv"
    tickers = np.genfromtxt(ticker_file, dtype=str, delimiter="\t")
    require(tickers.shape == (1737,) and len(set(tickers)) == 1737,
            "Expected 1737 unique qualified NYSE tickers.")
    # The source feature generator discards 29 warm-up trading days.
    full_dates = date_file.read_text(encoding="utf-8").splitlines()
    dates = [line[:10] for line in full_dates[29:]]
    require(len(dates) == 1245 and dates[0] == "2013-01-02" and dates[-1] == "2017-12-08",
            "Unexpected NYSE calendar after the source's 29-day warm-up.")
    require(all(a < b for a, b in zip(dates, dates[1:])), "Dates are not strictly increasing.")

    raw = np.empty((len(tickers), len(dates), 6), dtype=np.float32)
    source_digest = hashlib.sha256()
    print("Checking all qualified source CSV files...", flush=True)
    for index, ticker in enumerate(tickers):
        path = csv_root / f"NYSE_{ticker}_1.csv"
        content = path.read_bytes()
        array = np.genfromtxt(io.BytesIO(content), dtype=np.float32, delimiter=",")
        require(array.shape == (len(dates), 6), f"Unexpected shape: {path}: {array.shape}")
        require(np.isfinite(array).all(), f"Non-finite source values: {path}")
        require(np.array_equal(array[:, 0], np.arange(len(dates))), f"Date index mismatch: {path}")
        raw[index] = array
        source_digest.update(ticker.encode("utf-8") + b"\0" + hashlib.sha256(content).digest())

    print("Calling the project's load_EOD_data(steps=1)...", flush=True)
    arrays = dict(zip(
        ["eod_data", "mask_data", "gt_data", "price_data"],
        load_EOD_data(str(csv_root), "NYSE", tickers, steps=1),
    ))
    for name, array in arrays.items():
        expected_shape = (1737, 1245, 5) if name == "eod_data" else (1737, 1245)
        require(array.shape == expected_shape and array.dtype == np.float32,
                f"Unexpected generated shape/dtype: {name}")
        require(np.isfinite(array).all(), f"Non-finite generated values: {name}")

    eod, mask, gt, price = (arrays[k] for k in ["eod_data", "mask_data", "gt_data", "price_data"])
    raw_missing = np.abs(raw[:, :, -1] + 1234) < 1e-8
    require(np.array_equal(mask, (~raw_missing).astype(np.float32)), "Mask does not match raw missing closes.")
    expected_features = raw[:, :, 1:].copy()
    expected_features[np.abs(expected_features + 1234) < 1e-8] = np.float32(1.1)
    require(np.array_equal(eod, expected_features), "Features differ from the original replacement rule.")
    require(np.array_equal(price, eod[:, :, -1]), "Prices do not equal the final input feature.")
    require((price > 0).all(), "Non-positive base prices.")
    valid_pairs = (~raw_missing[:, 1:]) & (~raw_missing[:, :-1])
    expected_returns = (raw[:, 1:, -1] - raw[:, :-1, -1]) / raw[:, :-1, -1]
    label_error = float(np.max(np.abs(gt[:, 1:][valid_pairs] - expected_returns[valid_pairs])))
    require(label_error <= 1e-7 and np.all(gt[:, 0] == 0), "One-day return labels do not match raw closes.")
    # The original loader mutates missing prices to 1.1 before reading the next day.
    # Such next-day labels are retained unchanged but excluded by the window mask.
    excluded_after_missing = int((raw_missing[:, :-1] & ~raw_missing[:, 1:]).sum())
    window_mask = np.lib.stride_tricks.sliding_window_view(mask, 17, axis=1).min(axis=-1)
    require(window_mask.sum(axis=0).min() >= 10, "A window has fewer than ten valid stocks.")

    report = {
        "source_root": str(source_root),
        "source_ticker_file": str(ticker_file),
        "source_dates_file": str(date_file),
        "source_ticker_sha256": sha256(ticker_file),
        "source_dates_sha256": sha256(date_file),
        "source_csv_aggregate_sha256": source_digest.hexdigest(),
        "source_csv_digest_rule": "qualified ticker order: UTF-8 ticker + NUL + raw CSV SHA256 bytes",
        "loader": str(PROJECT_ROOT / "src/load_data.py"),
        "loader_sha256": sha256(PROJECT_ROOT / "src/load_data.py"),
        "steps": 1,
        "features": ["close_ma5", "close_ma10", "close_ma20", "close_ma30", "close"],
        "stocks": len(tickers),
        "trading_days": len(dates),
        "date_range": [dates[0], dates[-1]],
        "ignored_extra_csv_count": len(list(csv_root.glob("NYSE*_1.csv"))) - len(tickers),
        "missing_stock_days": int(raw_missing.sum()),
        "valid_stock_day_fraction": float(mask.mean(dtype=np.float64)),
        "valid_one_day_label_pairs": int(valid_pairs.sum()),
        "max_valid_label_absolute_error": label_error,
        "labels_after_missing_previous_day_excluded_by_window_mask": excluded_after_missing,
        "min_valid_stocks_per_17_day_window": int(window_mask.sum(axis=0).min()),
        "price_range": [float(price.min()), float(price.max())],
        "valid_return_range": [float(gt[:, 1:][valid_pairs].min()), float(gt[:, 1:][valid_pairs].max())],
        "splits": {
            "train_nominal": {"indices": [0, 755], "dates": [dates[0], dates[755]], "days": 756},
            "validation": {"indices": [756, 1007], "dates": [dates[756], dates[1007]], "days": 252},
            "test": {"indices": [1008, 1244], "dates": [dates[1008], dates[-1]], "days": 237},
        },
        "sampling_note": "Original training candidates can include validation targets 756..771; preserved.",
    }

    dataset_root = PROJECT_ROOT / "dataset"
    target = dataset_root / "NYSE"
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".NYSE-build-", dir=dataset_root) as staging_name:
        staging = Path(staging_name).resolve()
        require(staging.parent == dataset_root.resolve(), "Staging directory escaped dataset root.")
        report["arrays"] = {}
        for name, array in arrays.items():
            path = staging / f"{name}.pkl"
            with path.open("wb") as stream:
                pickle.dump(array, stream, protocol=4)
            with path.open("rb") as stream:
                restored = pickle.load(stream)
            require(np.array_equal(array, restored), f"Pickle round-trip mismatch: {name}")
            report["arrays"][name] = {
                "shape": list(array.shape), "dtype": str(array.dtype),
                "bytes": path.stat().st_size, "sha256": sha256(path),
            }
        (staging / "tickers.csv").write_text("ticker\n" + "\n".join(tickers.tolist()) + "\n", encoding="utf-8")
        (staging / "trading_dates.csv").write_text("date\n" + "\n".join(dates) + "\n", encoding="utf-8")
        names = [f"{name}.pkl" for name in arrays] + ["tickers.csv", "trading_dates.csv", "validation_report.json"]
        existing = [name for name in names if (target / name).exists()]
        backup = PROJECT_ROOT / "dataset_backups" / ("NYSE_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        if existing:
            backup.mkdir(parents=True)
            for name in existing:
                shutil.copy2(target / name, backup / name)
            report["previous_files_backup"] = str(backup)
        (staging / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        installed = []
        try:
            for name in names:
                (staging / name).replace(target / name)
                installed.append(name)
                inherit_directory_permissions(target / name)
        except Exception:
            for name in installed:
                if name in existing:
                    shutil.copy2(backup / name, target / name)
                else:
                    (target / name).unlink()
            raise
    print(json.dumps({"installed": str(target), "stocks": 1737, "days": 1245,
                      "missing_stock_days": report["missing_stock_days"],
                      "report": str(target / "validation_report.json")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=PROJECT_ROOT.parent / "Temporal_Relational_Stock_Ranking-master")
    args = parser.parse_args()
    build(args.source_root.resolve())
