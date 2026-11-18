"""Strict E0/fusion runtime used by the config-driven train_strict.py entry."""
import argparse
import csv
import importlib.util
import json
import platform
import random
import copy
import os
import uuid
import sys
import traceback
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .io import digest_file, file_manifest, new_artifact, now, verify_files, write_json
from .protocol import PROJECT, SCHEMA, load_market as MarketData, audit_news, backbone_contract, configuration, prepare_market, resolve, validate_config
from .validation import integer, finite_number
from .console import (progress_iter,format_number,format_metric,format_loss,terminal_rule,compact_epoch_summary,
    print_training_header,print_training_footer)
from .outputs import destination, run_destination, initialize, Tee, refresh_training, publish_training, publish_prediction, publish_training_prediction, verify_bundle

# Only this reviewed phase-1 BASELINE implementation is forward-compatible.
# Arbitrary changed source, legacy official checkpoints and old fusion/PCA artifacts remain rejected.
AUDITED_PHASE1_BASELINE={
    "protocol.py":"c526fd5022217dd1b86457beae8fb9c25c748b3a2a62fc010b3fa5df261449dc",
    "experiment.py":"fbb8a4f5474102309e12487dd4f5681be9da8278e5f2af8e76715669bf6705f8",
    "fusion.py":"dea2f0758fb4bbfc97e20948ba174703da6a0a5f0dcda25cd9e58e7608caeb64",
    "dataset.py":"37ac524f150ebb65e9391f5a5f55cff615aad5c4e41eba5d81b4d0619b3b1d06",
    "time_axis.py":"517ee92299fc1abb73c880c7298865123ecd598e7e512c3dea7deb47b564baea"}
AUDITED_REVIEW_BASELINE={
    "protocol.py":"22c88f3ef2f3bb57d456e716bbcef33ee9c51c3c9c2e8956da01efb0a4d95724",
    "experiment.py":"9d4d9767c78b8c039fd15facc2eed9be8b3209cf8c2071c8773e205964427de7",
    "fusion.py":"dea2f0758fb4bbfc97e20948ba174703da6a0a5f0dcda25cd9e58e7608caeb64",
    "dataset.py":"bb80c5821349a691e9ca0fb849e347995f3e2ac7dd7eee4fea81cf35cdab163e",
    "time_axis.py":"feb5b38b6ad46be32d15154e548d5158418d0eb3de239f3a514f084d72de9f2c",
    "panel.py":"c5964bb643b7925bd914188e053ee655892e3012d6530806b42aa2fc8e2fac01",
    "io.py":"61f423b8a8b25acdbca8985b55786da66ac1e479ebfcda2eaa5c2ae1095662da",
    "validation.py":"40a5e94375011861be579bebd3bed7c6d90eeac094c053baf0a6b0a82e52c909"}


AUDITED_OUTPUTS_BASELINE={'protocol.py': '22c88f3ef2f3bb57d456e716bbcef33ee9c51c3c9c2e8956da01efb0a4d95724', 'experiment.py': '15d9b490046de5fae75d798ef42f4cefdaa9f61fb194261c476a929a2713bfec', 'fusion.py': 'dea2f0758fb4bbfc97e20948ba174703da6a0a5f0dcda25cd9e58e7608caeb64', 'dataset.py': 'bb80c5821349a691e9ca0fb849e347995f3e2ac7dd7eee4fea81cf35cdab163e', 'time_axis.py': 'feb5b38b6ad46be32d15154e548d5158418d0eb3de239f3a514f084d72de9f2c', 'panel.py': 'c5964bb643b7925bd914188e053ee655892e3012d6530806b42aa2fc8e2fac01', 'io.py': '61f423b8a8b25acdbca8985b55786da66ac1e479ebfcda2eaa5c2ae1095662da', 'validation.py': '40a5e94375011861be579bebd3bed7c6d90eeac094c053baf0a6b0a82e52c909', 'outputs.py': 'c134aacbce5a276146d6e55d9caf67de882c141fed25ce60d6eb77ce237f732c'}


AUDITED_WORKFLOW_BASELINE={'protocol.py': '7887e5e3e73511dd22c22df99f86e821188ad32627ef4718d73f9778d565144a', 'experiment.py': '75f60406c5cad4911d038b650d3a3d78c0a57c7b7ed70fa84ced295ccc24b202', 'fusion.py': 'dea2f0758fb4bbfc97e20948ba174703da6a0a5f0dcda25cd9e58e7608caeb64', 'dataset.py': 'bb80c5821349a691e9ca0fb849e347995f3e2ac7dd7eee4fea81cf35cdab163e', 'time_axis.py': 'feb5b38b6ad46be32d15154e548d5158418d0eb3de239f3a514f084d72de9f2c', 'panel.py': 'c5964bb643b7925bd914188e053ee655892e3012d6530806b42aa2fc8e2fac01', 'io.py': '61f423b8a8b25acdbca8985b55786da66ac1e479ebfcda2eaa5c2ae1095662da', 'validation.py': '40a5e94375011861be579bebd3bed7c6d90eeac094c053baf0a6b0a82e52c909', 'outputs.py': 'c134aacbce5a276146d6e55d9caf67de882c141fed25ce60d6eb77ce237f732c', 'workflow.py': 'e150dd393d4dcbdbb33b12f76849a63d82a980bffded36645d130cbf74ee2379'}


AUDITED_ORIGINAL_METRICS_BASELINE={'protocol.py': '30b504a8de3f466870e3539f2c6a3c9c04f4dca47f5bed04c9ee02ca71b955df', 'experiment.py': '5796fe75cd625a283e464351881ed0ece9a091699fdc60734c68acbf4caefddd', 'fusion.py': 'dea2f0758fb4bbfc97e20948ba174703da6a0a5f0dcda25cd9e58e7608caeb64', 'dataset.py': 'bb80c5821349a691e9ca0fb849e347995f3e2ac7dd7eee4fea81cf35cdab163e', 'time_axis.py': 'feb5b38b6ad46be32d15154e548d5158418d0eb3de239f3a514f084d72de9f2c', 'panel.py': 'c5964bb643b7925bd914188e053ee655892e3012d6530806b42aa2fc8e2fac01', 'io.py': '61f423b8a8b25acdbca8985b55786da66ac1e479ebfcda2eaa5c2ae1095662da', 'validation.py': '40a5e94375011861be579bebd3bed7c6d90eeac094c053baf0a6b0a82e52c909', 'outputs.py': '63894568e8ad24c681fe304312bfbe24fafa4f72032b96170cece8e63bc58542', 'workflow.py': 'a2283dcd9ff7fec23b404319483cf119f3663167f2116f8334868e754d18778d', 'original_evaluator.py': '0ca7398f8c975b65be5fe003975dc28732acb27278716ed00fce0d628d0a0e60'}


AUDITED_TWO_CONFIGS_BASELINE={'protocol.py': '2eb1a5a60d546052bb7aff39dcae1d487e1ba31c6d42ddc4e3a6da0395c71620', 'experiment.py': '034f88b3a2737c97ae878e1606ac22955f0932f0e508571e9637eba990bc8e75', 'fusion.py': 'dea2f0758fb4bbfc97e20948ba174703da6a0a5f0dcda25cd9e58e7608caeb64', 'dataset.py': 'bb80c5821349a691e9ca0fb849e347995f3e2ac7dd7eee4fea81cf35cdab163e', 'time_axis.py': 'feb5b38b6ad46be32d15154e548d5158418d0eb3de239f3a514f084d72de9f2c', 'panel.py': 'c5964bb643b7925bd914188e053ee655892e3012d6530806b42aa2fc8e2fac01', 'io.py': '61f423b8a8b25acdbca8985b55786da66ac1e479ebfcda2eaa5c2ae1095662da', 'validation.py': '40a5e94375011861be579bebd3bed7c6d90eeac094c053baf0a6b0a82e52c909', 'outputs.py': '53af8e8c855c47bea4bcea3e7c6db1922ed9ad885268c8bc318c71420d18a753', 'workflow.py': '018f7cdae7d5745f32c099e92e5c9b8420a65bfe5852e29224200ba9326b6cdc', 'original_evaluator.py': '0ca7398f8c975b65be5fe003975dc28732acb27278716ed00fce0d628d0a0e60', 'configuration.py': '649a64b9e9d1ba6d9d6558e9c39ce2ec1affeca3a7b3dc59946ab4c066dfc3d9', 'original.py': '7edc8d0f2acc336204d892e06e959a18757dea390d316cedb50540fa619b278e'}


AUDITED_CONSOLE_BASELINE={'protocol.py': '2eb1a5a60d546052bb7aff39dcae1d487e1ba31c6d42ddc4e3a6da0395c71620', 'experiment.py': '499617b00cca80885c12b7cc5928b1eaadf28cc0e1a3939699d856f8d3e56bf6', 'fusion.py': 'dea2f0758fb4bbfc97e20948ba174703da6a0a5f0dcda25cd9e58e7608caeb64', 'dataset.py': 'bb80c5821349a691e9ca0fb849e347995f3e2ac7dd7eee4fea81cf35cdab163e', 'time_axis.py': 'feb5b38b6ad46be32d15154e548d5158418d0eb3de239f3a514f084d72de9f2c', 'panel.py': 'c5964bb643b7925bd914188e053ee655892e3012d6530806b42aa2fc8e2fac01', 'io.py': '61f423b8a8b25acdbca8985b55786da66ac1e479ebfcda2eaa5c2ae1095662da', 'validation.py': '40a5e94375011861be579bebd3bed7c6d90eeac094c053baf0a6b0a82e52c909', 'outputs.py': '7e4f4959a5443617c53941c239be0c7f5d311b5dcc934c9ec4c15ad417fc0e73', 'workflow.py': '3d7a99caebf2db6722bc785f043572f5f241b5723b73c4cad5791ab77725dfa6', 'original_evaluator.py': '0ca7398f8c975b65be5fe003975dc28732acb27278716ed00fce0d628d0a0e60', 'configuration.py': '649a64b9e9d1ba6d9d6558e9c39ce2ec1affeca3a7b3dc59946ab4c066dfc3d9', 'original.py': '7edc8d0f2acc336204d892e06e959a18757dea390d316cedb50540fa619b278e', 'console.py': '61525ce845ebe5bb42f13a4afcac239703e6848d599187c973dfd8428efea613'}


AUDITED_NAMING_BASELINE={'protocol.py': '2eb1a5a60d546052bb7aff39dcae1d487e1ba31c6d42ddc4e3a6da0395c71620', 'experiment.py': '559fbd9fdc20751bbf39f4fd2901d7e400b68623ef693b21249bbf5a82c98b6d', 'fusion.py': 'dea2f0758fb4bbfc97e20948ba174703da6a0a5f0dcda25cd9e58e7608caeb64', 'dataset.py': 'bb80c5821349a691e9ca0fb849e347995f3e2ac7dd7eee4fea81cf35cdab163e', 'time_axis.py': 'feb5b38b6ad46be32d15154e548d5158418d0eb3de239f3a514f084d72de9f2c', 'panel.py': 'c5964bb643b7925bd914188e053ee655892e3012d6530806b42aa2fc8e2fac01', 'io.py': '61f423b8a8b25acdbca8985b55786da66ac1e479ebfcda2eaa5c2ae1095662da', 'validation.py': '40a5e94375011861be579bebd3bed7c6d90eeac094c053baf0a6b0a82e52c909', 'outputs.py': '1e1b1d317649de88dae6e13c4d1b03153502ec94523a3240c34209cc64c040c2', 'workflow.py': 'cc2bdc02f3dbb825ddd8e35acc8eae226e6fea777504652cb19fff0e4369466b', 'original_evaluator.py': '0ca7398f8c975b65be5fe003975dc28732acb27278716ed00fce0d628d0a0e60', 'configuration.py': '649a64b9e9d1ba6d9d6558e9c39ce2ec1affeca3a7b3dc59946ab4c066dfc3d9', 'original.py': '7edc8d0f2acc336204d892e06e959a18757dea390d316cedb50540fa619b278e', 'console.py': '61525ce845ebe5bb42f13a4afcac239703e6848d599187c973dfd8428efea613'}


AUDITED_SINGLE_FOLDER_BASELINE={'protocol.py': '2eb1a5a60d546052bb7aff39dcae1d487e1ba31c6d42ddc4e3a6da0395c71620', 'experiment.py': '5ac25f8b4d08126074f0523b2f45f52b95456fbba3215b021c636f24eaa730af', 'fusion.py': 'dea2f0758fb4bbfc97e20948ba174703da6a0a5f0dcda25cd9e58e7608caeb64', 'dataset.py': 'bb80c5821349a691e9ca0fb849e347995f3e2ac7dd7eee4fea81cf35cdab163e', 'time_axis.py': 'feb5b38b6ad46be32d15154e548d5158418d0eb3de239f3a514f084d72de9f2c', 'panel.py': 'c5964bb643b7925bd914188e053ee655892e3012d6530806b42aa2fc8e2fac01', 'io.py': '61f423b8a8b25acdbca8985b55786da66ac1e479ebfcda2eaa5c2ae1095662da', 'validation.py': '40a5e94375011861be579bebd3bed7c6d90eeac094c053baf0a6b0a82e52c909', 'outputs.py': '9319648e32cf898922891132697baab64c76eee56a47f0c08e51a47f142a4db9', 'workflow.py': '06770f8bf23eec790948ed88e51f11fadb36f65fb19524ab4376af874cc6a8eb', 'original_evaluator.py': '0ca7398f8c975b65be5fe003975dc28732acb27278716ed00fce0d628d0a0e60', 'configuration.py': '649a64b9e9d1ba6d9d6558e9c39ce2ec1affeca3a7b3dc59946ab4c066dfc3d9', 'original.py': '7edc8d0f2acc336204d892e06e959a18757dea390d316cedb50540fa619b278e', 'console.py': '61525ce845ebe5bb42f13a4afcac239703e6848d599187c973dfd8428efea613'}


AUDITED_OFFLINE_ENTRIES_BASELINE={'protocol.py': '2eb1a5a60d546052bb7aff39dcae1d487e1ba31c6d42ddc4e3a6da0395c71620', 'experiment.py': 'e8bcc6536c94983d1cdc5f729084dbefe527200634f831ce047e936203858d69', 'fusion.py': 'dea2f0758fb4bbfc97e20948ba174703da6a0a5f0dcda25cd9e58e7608caeb64', 'dataset.py': '1f21deaff6dffdd293496b67a6136992e5c30aa9d37b4d6c6dde6373793f25da', 'time_axis.py': 'feb5b38b6ad46be32d15154e548d5158418d0eb3de239f3a514f084d72de9f2c', 'panel.py': '6cb744a3b6deb1f68acfa749ea9cab55ec065e231bbf1e88ff008b9a95782a1c', 'io.py': '61f423b8a8b25acdbca8985b55786da66ac1e479ebfcda2eaa5c2ae1095662da', 'validation.py': '40a5e94375011861be579bebd3bed7c6d90eeac094c053baf0a6b0a82e52c909', 'outputs.py': '9319648e32cf898922891132697baab64c76eee56a47f0c08e51a47f142a4db9', 'workflow.py': '06770f8bf23eec790948ed88e51f11fadb36f65fb19524ab4376af874cc6a8eb', 'original_evaluator.py': '0ca7398f8c975b65be5fe003975dc28732acb27278716ed00fce0d628d0a0e60', 'configuration.py': '1f6b465c933cc970786b65ef2f2da0128124008872b7608cdf2b7599c4b35b10', 'original.py': '7edc8d0f2acc336204d892e06e959a18757dea390d316cedb50540fa619b278e', 'console.py': '61525ce845ebe5bb42f13a4afcac239703e6848d599187c973dfd8428efea613', 'routing.py': 'e85553d109de5362fdf17a551e8098c698a04c8ebee965637a75f7c8bdc0a795'}


def model_api():
    spec = importlib.util.spec_from_file_location("strict_pinned_stockmixer", PROJECT / "src/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StockMixer, module.get_loss


AUDITED_AUTO_REVIEW_BASELINE={'protocol.py': '2eb1a5a60d546052bb7aff39dcae1d487e1ba31c6d42ddc4e3a6da0395c71620', 'experiment.py': '89021d7ef8d5097ae9c4d9002c7cd308638c593a8a0d7e6ea58c09351d813853', 'fusion.py': 'dea2f0758fb4bbfc97e20948ba174703da6a0a5f0dcda25cd9e58e7608caeb64', 'dataset.py': '1f21deaff6dffdd293496b67a6136992e5c30aa9d37b4d6c6dde6373793f25da', 'time_axis.py': 'feb5b38b6ad46be32d15154e548d5158418d0eb3de239f3a514f084d72de9f2c', 'panel.py': '6cb744a3b6deb1f68acfa749ea9cab55ec065e231bbf1e88ff008b9a95782a1c', 'io.py': '61f423b8a8b25acdbca8985b55786da66ac1e479ebfcda2eaa5c2ae1095662da', 'validation.py': '40a5e94375011861be579bebd3bed7c6d90eeac094c053baf0a6b0a82e52c909', 'outputs.py': '9319648e32cf898922891132697baab64c76eee56a47f0c08e51a47f142a4db9', 'workflow.py': '06770f8bf23eec790948ed88e51f11fadb36f65fb19524ab4376af874cc6a8eb', 'original_evaluator.py': '0ca7398f8c975b65be5fe003975dc28732acb27278716ed00fce0d628d0a0e60', 'configuration.py': '5bd99c8ff44ffc58876af0230acb2de3bc1ddb315b866e85e2e9528ef7d345ce', 'original.py': '7edc8d0f2acc336204d892e06e959a18757dea390d316cedb50540fa619b278e', 'console.py': '61525ce845ebe5bb42f13a4afcac239703e6848d599187c973dfd8428efea613', 'routing.py': 'e85553d109de5362fdf17a551e8098c698a04c8ebee965637a75f7c8bdc0a795'}


AUDITED_NEWS_PIPELINE_BASELINE={'protocol.py': '9d162f0f6bcd8b7f45c7867e9c2420d802dfedb66a3cfd03150e16ce6909c5a4', 'experiment.py': 'e9bcf9eacc19e14c85defc0a1bd65b63bd0f501e03e32e4108788976cfa40f99', 'fusion.py': 'dea2f0758fb4bbfc97e20948ba174703da6a0a5f0dcda25cd9e58e7608caeb64', 'dataset.py': '337f8a779c7198bb77cf1e2fca0612a70183ec60873e03cdc4a19c8fac8e2083', 'time_axis.py': 'feb5b38b6ad46be32d15154e548d5158418d0eb3de239f3a514f084d72de9f2c', 'panel.py': 'ec9dd7de9484fd71780a0feaaebf9c41000dbaf0b89862227b3a7846bd94d7e1', 'io.py': '61f423b8a8b25acdbca8985b55786da66ac1e479ebfcda2eaa5c2ae1095662da', 'validation.py': '40a5e94375011861be579bebd3bed7c6d90eeac094c053baf0a6b0a82e52c909', 'outputs.py': '9319648e32cf898922891132697baab64c76eee56a47f0c08e51a47f142a4db9', 'workflow.py': '06770f8bf23eec790948ed88e51f11fadb36f65fb19524ab4376af874cc6a8eb', 'original_evaluator.py': '0ca7398f8c975b65be5fe003975dc28732acb27278716ed00fce0d628d0a0e60', 'configuration.py': '391a0051ebc10d9329c25191f5fa03c87e7fa5c1fe5886332aa52b92774a9e41', 'original.py': '7edc8d0f2acc336204d892e06e959a18757dea390d316cedb50540fa619b278e', 'console.py': '61525ce845ebe5bb42f13a4afcac239703e6848d599187c973dfd8428efea613', 'routing.py': 'e85553d109de5362fdf17a551e8098c698a04c8ebee965637a75f7c8bdc0a795', 'relations.py': 'd5a407e387272c0d1d410de1646f39b4a984719007e75dcbf71b99356ae80c88'}


def code_hashes():
    return {name: digest_file(PROJECT / "news_fusion" / name) for name in
            ("protocol.py", "experiment.py", "fusion.py", "dataset.py", "time_axis.py", "panel.py", "io.py", "validation.py", "outputs.py", "workflow.py", "original_evaluator.py", "configuration.py", "original.py", "console.py", "routing.py", "relations.py", "semantic_review.py", "master_data.py", "industry_review.py", "industry_scope_review.py", "upstream.py", "industry_dictionary.py")}


def choose_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    return device


def metrics(prediction, truth, mask):
    """Explicit Pearson / Spearman / ICIR; original RIC retained as an ICIR alias."""
    prediction,truth,mask=(np.asarray(v) for v in (prediction,truth,mask))
    if prediction.ndim!=2 or prediction.shape!=truth.shape or prediction.shape!=mask.shape or not prediction.size or not np.isin(mask,[0,1]).all():
        raise ValueError("Metrics require matching [stocks,days] arrays and a binary mask.")
    prediction=prediction.astype(np.float64)
    truth=truth.astype(np.float64)
    valid = mask.astype(bool)
    if not valid.any() or not np.isfinite(prediction[valid]).all() or not np.isfinite(truth[valid]).all():
        raise ValueError("No finite evaluated observations.")
    ic, rank_ic, prec, daily5, counts = [], [], [], [], []
    for day in range(prediction.shape[1]):
        keep = valid[:, day]
        p, y = prediction[keep, day], truth[keep, day]
        counts.append(int(keep.sum()))
        if len(p) >= 2 and np.std(p) > 0 and np.std(y) > 0:
            ic.append(float(np.corrcoef(p, y)[0, 1]))
            rank_ic.append(float(np.corrcoef(pd.Series(p).rank(), pd.Series(y).rank())[0, 1]))
        order = np.argsort(p, kind="stable")
        if len(p) >= 10:
            prec.append(float(np.mean(y[order[-10:]] >= 0)))
        if len(p) >= 5:
            daily5.append(float(np.mean(y[order[-5:]])))
    def mean(items):
        return float(np.mean(items)) if items else None
    def ratio(items):
        return float(np.mean(items) / np.std(items)) if len(items) >= 2 and np.std(items) > 0 else None
    mse = float(np.mean((prediction[valid] - truth[valid]) ** 2))
    ir = ratio(ic)
    sharpe = ratio(daily5)
    return {"mse": mse, "rmse": mse ** .5, "mae": float(np.mean(np.abs(prediction[valid]-truth[valid]))),
            "IC": mean(ic), "RankIC": mean(rank_ic), "ICIR": ir, "RIC": ir,
            "RankICIR": ratio(rank_ic), "prec_10": mean(prec),
            "sharpe5": float(sharpe * np.sqrt(252)) if sharpe is not None else None,
            "correlation_days": len(ic), "evaluated_days": prediction.shape[1],
            "valid_stock_counts": counts, "valid_observations": int(valid.sum()),
            "portfolio_note": "Equal-weight top5 close-to-close diagnostic; no costs/execution assumptions; not tradable PnL."}


def original_metrics(prediction,truth,mask):
    """Exact official evaluator on float64 validation buffers; undefined ratios become JSON null."""
    from .original_evaluator import evaluate as official_evaluate
    p,y,m=(np.asarray(value,dtype=np.float64) for value in (prediction,truth,mask))
    if p.ndim!=2 or p.shape!=y.shape or p.shape!=m.shape or not p.size or not np.isin(m,[0,1]).all():
        raise ValueError("Official metrics require matching [stocks,days] arrays and binary masks.")
    if not np.isfinite(p).all() or not np.isfinite(y).all() or np.any(m.sum(axis=0)==0):
        raise ValueError("Official metrics require finite arrays and valid stocks on every evaluated day.")
    with np.errstate(divide="ignore",invalid="ignore"):
        result=official_evaluate(p,y,m)
    return {key:float(value) if np.isfinite(value) else None for key,value in result.items()}


def print_original_performance(label,performance):
    print(label+" performance:\n", "mse:{}, IC:{}, RIC:{}, prec@10:{}, SR:{}".format(
        format_loss(performance.get('mse')),*(format_metric(performance.get(key)) for key in ('IC','RIC','prec_10','sharpe5'))),flush=True)


def print_original_epoch(train_losses,validation,test=None):
    print('Train : loss:{}  =  {} + alpha*{}'.format(*(format_loss(value) for value in train_losses)),flush=True)
    print('Valid : loss:{}  =  {} + alpha*{}'.format(*(format_loss(validation[key]) for key in ('loss','mse_loss','rank_loss'))),flush=True)
    if test is not None:
        print('Test: loss:{}  =  {} + alpha*{}'.format(*(format_loss(test[key]) for key in ('loss','mse_loss','rank_loss'))),flush=True)
    print_original_performance('Valid',validation)
    if test is not None:
        print_original_performance('Test',test)
    print('\n',flush=True)


def tensor(array, device):
    return torch.as_tensor(np.array(array, copy=True), dtype=torch.float32, device=device)


def batch_forward(model, data, target, device, panel=None):
    x, price, truth, mask = [tensor(a, device) for a in data.window(int(target))]
    if panel is None:
        prediction = model(x)
    else:
        news = {k: tensor(v, device) for k, v in panel.decision(int(target)-1).items()
                if k not in ("decision_idx", "target_idx")}
        prediction = model(x, price, **news)
    if not torch.isfinite(prediction).all():
        raise ValueError("Nonfinite model output.")
    return prediction, truth, price, mask


def evaluate(model, data, targets, device, alpha, panel=None, metric_protocol="stockmixer_original"):
    _, get_loss = model_api()
    losses, predictions, truth, masks = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for target in targets:
            prediction, y, price, mask = batch_forward(model, data, target, device, panel)
            loss, mse, rank, rr = get_loss(prediction, y, price, mask, len(data.axis.stock_uids), alpha)
            if not all(torch.isfinite(value).all() for value in (loss,mse,rank,rr)):
                raise ValueError("Nonfinite validation/test loss or returns.")
            losses.append([loss.item(), mse.item(), rank.item()])
            predictions.append(rr.cpu().numpy()[:, 0]); truth.append(y.cpu().numpy()[:, 0]); masks.append(mask.cpu().numpy()[:, 0])
    p, y, m = (np.stack(rows, axis=1) for rows in (predictions, truth, masks))
    result = original_metrics(p,y,m) if metric_protocol=="stockmixer_original" else metrics(p,y,m) if metric_protocol=="valid_only" else None
    if result is None:
        raise ValueError("Unknown metric protocol.")
    averaged=[sum(row[i] for row in losses)/len(losses) for i in range(3)] if getattr(data,"protocol","strict")=="original_reproduction" else np.mean(losses,axis=0).tolist()
    result.update(zip(("loss", "mse_loss", "rank_loss"), averaged))
    return result, (p, y, m)


def validate_checkpoint(checkpoint, data, panel=None):
    if checkpoint.get("schema_version") != SCHEMA or checkpoint.get("protocol") != getattr(data,"protocol","strict"):
        raise ValueError("Checkpoint protocol differs from prepared data; original and strict checkpoints cannot be mixed.")
    if checkpoint["data_manifest_sha256"] != digest_file(data.directory / "manifest.json"):
        raise ValueError("Checkpoint preprocessing/data artifact mismatch.")
    if checkpoint["backbone"] != backbone_contract():
        raise ValueError("Checkpoint backbone mismatch.")
    exact=checkpoint["code_sha256"]==code_hashes()
    prior=checkpoint.get("mode")=="baseline" and checkpoint["code_sha256"]==AUDITED_PHASE1_BASELINE
    review=checkpoint.get("mode")=="baseline" and checkpoint["code_sha256"]==AUDITED_REVIEW_BASELINE
    outputs=checkpoint.get("mode")=="baseline" and checkpoint["code_sha256"]==AUDITED_OUTPUTS_BASELINE
    workflow=checkpoint.get("mode")=="baseline" and checkpoint["code_sha256"]==AUDITED_WORKFLOW_BASELINE
    original_metrics=checkpoint.get("mode")=="baseline" and checkpoint["code_sha256"]==AUDITED_ORIGINAL_METRICS_BASELINE
    two_configs=checkpoint.get("mode")=="baseline" and checkpoint["code_sha256"]==AUDITED_TWO_CONFIGS_BASELINE
    console=checkpoint.get("mode")=="baseline" and checkpoint["code_sha256"]==AUDITED_CONSOLE_BASELINE
    naming=checkpoint.get("mode")=="baseline" and checkpoint["code_sha256"]==AUDITED_NAMING_BASELINE
    single_folder=checkpoint.get("mode")=="baseline" and checkpoint["code_sha256"]==AUDITED_SINGLE_FOLDER_BASELINE
    offline_entries=checkpoint.get("mode")=="baseline" and checkpoint["code_sha256"]==AUDITED_OFFLINE_ENTRIES_BASELINE
    auto_review=checkpoint.get("mode")=="baseline" and checkpoint["code_sha256"]==AUDITED_AUTO_REVIEW_BASELINE
    news_pipeline=checkpoint.get("mode")=="baseline" and checkpoint["code_sha256"]==AUDITED_NEWS_PIPELINE_BASELINE
    if not news_pipeline and not auto_review and not exact and not prior and not review and not outputs and not workflow and not original_metrics and not two_configs and not console and not naming and not single_folder and not offline_entries:
        raise ValueError("Checkpoint code mismatch: only explicitly audited earlier baseline versions are compatible.")
    validate_config(checkpoint["config"])
    expected={"dataset":data.axis.dataset,"stock_axis_hash":data.axis.stock_hash,"date_axis_hash":data.axis.date_hash,
        "train_end_date":data.axis.dates[data.axis.train_end-1],"selection_end_date":data.axis.dates[data.axis.valid_end-1]}
    if any(checkpoint.get(k)!=value for k,value in expected.items()) or checkpoint["config"]["dataset"]!=data.axis.dataset:
        raise ValueError("Checkpoint dataset, axes or training/selection boundary mismatch.")
    if checkpoint.get("mode") not in ("baseline","fusion") or not isinstance(checkpoint.get("smoke_only"),bool):
        raise ValueError("Invalid checkpoint mode/classification.")
    if (checkpoint["mode"]=="fusion")!=(panel is not None):
        raise ValueError("A fusion checkpoint requires its verified panel; baseline requires no panel.")
    if panel is not None and checkpoint.get("panel_manifest_sha256") != digest_file(panel.directory / "manifest.json"):
        raise ValueError("Checkpoint/news panel mismatch.")
    return "exact_code" if exact else "audited_news_pipeline_baseline" if news_pipeline else "audited_auto_review_baseline" if auto_review else "audited_offline_entries_baseline" if offline_entries else "audited_single_folder_baseline" if single_folder else "audited_naming_baseline" if naming else "audited_console_baseline" if console else "audited_two_configs_baseline" if two_configs else "audited_original_metrics_baseline" if original_metrics else "audited_workflow_baseline" if workflow else "audited_outputs_baseline" if outputs else "audited_review_baseline" if review else "audited_phase1_baseline"


def load_panel(path, data, config, require_training_activity=True):
    from .dataset import NewsPanel
    panel = NewsPanel(path)
    panel.assert_experiment(data.axis, config["fusion"]["time_mode"],lookback=data.split.lookback)
    if panel.manifest['feature_dim']!=config['fusion']['news_dim']:
        raise ValueError('News panel dimension differs from the configured fusion input')
    if panel.manifest.get("article_encoder",{}).get("model_id")!="ProsusAI/finbert":
        raise ValueError("A real, pinned FinBERT artifact is required; synthetic fixtures cannot train an experiment.")
    channels = config["fusion"]["enabled_channels"]
    if not channels or not set(channels) <= {"company", "industry", "market"}:
        raise ValueError("Select at least one valid fusion channel; E0 uses baseline mode.")
    if "market" in channels and panel.manifest.get("market_routing_verified") is not True:
        raise ValueError("Market routing has not been verified. Build reviewed market links before fusion training.")
    if "market" in channels:
        panel.assert_market_routing()
    if set(channels)&{'company','industry'}:
        relation_proof=panel.assert_historical_relations()
        from .relations import require_industry_timing
        require_industry_timing(relation_proof,config['fusion']['time_mode'],channels)
    decisions = data.split.targets("train") - 1
    eligible_price=None
    for channel in channels if require_training_activity else ():
        active = panel.masks[channel][decisions] if panel.masks[channel].ndim == 1 else panel.masks[channel][:, decisions]
        if not active.any():
            raise ValueError("Enabled channel has no valid training news: " + channel)
        if eligible_price is None:
            targets=decisions+1
            missing=np.pad(np.cumsum(np.asarray(data.arrays['mask'])<=0,axis=1,dtype=np.int32),((0,0),(1,0)))
            eligible_price=missing[:,targets+1]==missing[:,targets-data.split.lookback]
        usable=active & eligible_price.any(axis=0) if active.ndim==1 else active & eligible_price
        if not usable.any():
            raise ValueError('Enabled channel has no news with valid training price/label windows: '+channel)
    return panel


def make_model(data, config, device, fusion=False):
    StockMixer, _ = model_api()
    model = StockMixer(stocks=len(data.axis.stock_uids), time_steps=16, channels=5,
                       market=int(data.axis.source["market_num"]), scale=int(config["model"]["scale_argument"]))
    if fusion:
        from .fusion import StockMixerWithNews
        settings = config["fusion"]
        if not settings["freeze_backbone"]:
            raise ValueError("First-stage fusion requires a frozen, separately trained strict E0 backbone.")
        model = StockMixerWithNews(model, **{k: settings[k] for k in
            ("news_dim", "hidden_dim", "dropout", "freeze_backbone", "enabled_channels")})
    return model.to(device)


def load_run_checkpoint(path):
    path = Path(path).resolve()
    if path.parent.name=="checkpoints" and (path.parent.parent/"manifest.json").exists():
        bundle=verify_bundle(path.parent.parent)
        if bundle["kind"]!="training_bundle" or path.name not in ("best_model.pt","last_model.pt"):
            raise ValueError("Expected a best/last checkpoint from a completed training output.")
        return load_run_checkpoint(path.parent.parent/"_artifact"/path.name)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "strict_training_run" or path.name not in ("best_model.pt","last_model.pt"):
        raise ValueError("Checkpoint must belong to a completed, hashed strict training run.")
    verify_files(path.parent, manifest, required=("best_model.pt","last_model.pt","run_status.json","config.json",
        "history.json","test_metrics.json","test_predictions.npz","sample_ledger.json"))
    status=json.loads((path.parent/"run_status.json").read_text(encoding="utf-8"))
    policy=manifest.get("test_policy","final_best_once")
    valid_policy=(policy=="final_best_once" and manifest.get("final_test_evaluations")==1) or (
        policy=="per_epoch" and manifest.get("epoch_test_evaluations")==manifest.get("epochs")
        and manifest.get("test_evaluations")==manifest.get("epochs") and manifest.get("final_test_evaluations")==0)
    if status.get("status")!="completed" or not valid_policy:
        raise ValueError("Checkpoint run is incomplete or has a different test protocol.")
    checkpoint=torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint,dict):
        raise ValueError("Expected a complete strict checkpoint mapping.")
    keys=("schema_version","protocol","mode","smoke_only","backbone","code_sha256","dataset","data_manifest_sha256",
        "stock_axis_hash","date_axis_hash","train_end_date","selection_end_date","parent_baseline_sha256","panel_manifest_sha256")
    if any(checkpoint.get(key)!=manifest.get(key) for key in keys):
        raise ValueError("Checkpoint metadata differs from completed run manifest.")
    saved_config=json.loads((path.parent/"config.json").read_text(encoding="utf-8"))
    if checkpoint.get("config")!=saved_config:
        raise ValueError("Checkpoint config differs from run config snapshot.")
    expected_epoch=manifest["best_epoch"] if path.name=="best_model.pt" else manifest["epochs"]
    if checkpoint.get("epoch")!=expected_epoch:
        raise ValueError("Checkpoint epoch does not match the selected run artifact.")
    return checkpoint


def atomic_checkpoint(checkpoint,path):
    """Never replace the previous good weights until the new serialization is complete."""
    path=Path(path)
    temporary=path.with_name("."+path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(checkpoint,stream)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def training_directory(output, structured=False):
    """Keep expensive training checkpoints on interruption; only a final manifest is loadable."""
    root=Path(output).resolve()
    if root.exists():
        raise FileExistsError("Training output exists; choose a new run directory.")
    root.parent.mkdir(parents=True,exist_ok=True)
    root.mkdir()
    directory=root/"_artifact" if structured else root
    if structured:
        directory.mkdir();initialize(root)
    write_json(directory/"run_status.json",{"status":"in_progress","started_at":now()})
    try:
        if structured:
            with (root/"logs/train.log").open("w",encoding="utf-8",buffering=1) as log:
                try:
                    with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
                        yield directory
                except BaseException:
                    traceback.print_exc(file=log)
                    raise
        else:
            yield directory
    except BaseException as error:
        write_json(directory/"run_status.json",{"status":"interrupted_or_failed","recorded_at":now(),
            "error_type":type(error).__name__,"error":str(error),"partial_checkpoints_retained":True})
        if structured:
            refresh_training(directory,root)
        raise


def train(project, config, data_dir, output=None, epochs=None, smoke=False, baseline=None, panel_path=None):
    config = copy.deepcopy(config)
    settings = config["training"]
    if epochs is not None:
        settings["epochs"] = epochs
    validate_config(config)
    data = MarketData(data_dir, project)
    if data.axis.dataset != config["dataset"]:
        raise ValueError("Selected dataset differs from prepared artifact.")
    if output is None:
        output=destination(project,dataset=data.axis.dataset,kind="fusion" if baseline is not None else "E0")
    original=getattr(data,"protocol","strict")=="original_reproduction"
    if original != (config["data"]["preprocessing"]=="original"):
        raise ValueError("Configured original/strict protocol differs from prepared market data.")
    random.seed(int(settings["numpy_seed"])); np.random.seed(int(settings["numpy_seed"]))
    torch.manual_seed(int(settings["torch_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(settings["torch_seed"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = not original
    device = choose_device(settings["device"])
    fusion = baseline is not None
    if fusion and panel_path is None and config["news"]["panel"] is not None:
        panel_path=resolve(project,config["news"]["panel"])
    if fusion and panel_path is None:
        raise ValueError("Fusion needs a real, verified panel; no dummy news is generated.")
    if not fusion and panel_path is not None:
        raise ValueError("A panel is only used together with a strict E0 checkpoint.")
    panel = load_panel(panel_path, data, config) if fusion else None
    model = make_model(data, config, device, fusion)
    parent_hash = None
    if fusion:
        parent = load_run_checkpoint(baseline)
        validate_checkpoint(parent, data)
        if parent["mode"] != "baseline" or (parent["smoke_only"] and not smoke):
            raise ValueError("A benchmark fusion run requires a non-smoke strict E0 checkpoint.")
        model.stock_mixer.load_state_dict(parent["model_state_dict"], strict=True)
        parent_hash = digest_file(baseline)
        # This is an invariant, not a comparison of different checkpoints/train modes.
        x, price, _, _ = [tensor(a, device) for a in data.window(int(data.split.targets("train")[0]))]
        model.eval()
        with torch.inference_mode():
            if not torch.equal(model(x, price), model.stock_mixer(x)):
                raise ValueError("Zero-news equality failed before fusion training.")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(settings["learning_rate"]))
    _, get_loss = model_api()
    history, ledger = [], []
    per_epoch=settings["test_policy"]=="per_epoch"
    metric_protocol=settings.get("metric_protocol","valid_only")
    test_evaluations=0
    best_test=None
    batch_offsets=np.arange(data.axis.train_end,dtype=int) if original else None
    epoch_samples=[]
    for phase in ("train", "validation", "test"):
        for target in data.split.targets(phase):
            ledger.append({"phase": "train_candidate" if original and phase=="train" else phase, "target_idx": int(target), "target_date": data.axis.dates[target],
                           "decision_idx": int(target-1), "offset": int(target-16)})
    common = {"schema_version": SCHEMA, "protocol": getattr(data,"protocol","strict"), "mode": "fusion" if fusion else "baseline",
        "runtime_contract":"strict_outputs_epoch_test_v1",
        "test_policy":settings["test_policy"],"metric_protocol":metric_protocol,
        "smoke_only": bool(smoke), "config": config, "backbone": backbone_contract(), "code_sha256": code_hashes(),
        "data_manifest_sha256": digest_file(data.directory / "manifest.json"), "dataset": data.axis.dataset,
        "data_directory": str(data.directory), "stock_axis_hash": data.axis.stock_hash, "date_axis_hash": data.axis.date_hash,
        "train_end_date": data.axis.dates[data.axis.train_end-1],
        "selection_end_date": data.axis.dates[data.axis.valid_end-1],
        "parent_baseline_sha256": parent_hash,
        "panel_directory": str(Path(panel_path).resolve()) if panel_path else None,
        "panel_manifest_sha256": digest_file(Path(panel_path)/"manifest.json") if panel_path else None}
    with training_directory(output,structured=True) as directory:
        start_time=time.time()
        epoch_times=[]
        print_training_header(project,output,data,config,model,device,fusion)
        write_json(directory / "config.json", config)
        write_json(directory / "sample_ledger.json", ledger)
        (directory / "model.txt").write_text(str(model), encoding="utf-8")
        best = float("inf")
        best_epoch=0
        for epoch in range(1, int(settings["epochs"])+1):
            epoch_start=time.time()
            model.train()
            if original:
                np.random.shuffle(batch_offsets)
                offsets=batch_offsets[:data.split.samples_per_epoch].copy()
                targets=offsets+data.split.lookback+data.split.steps-1
                epoch_samples.append({"epoch":epoch,"offsets":offsets.tolist(),"targets":targets.tolist()})
            else:
                targets = data.split.targets("train").copy()
                np.random.shuffle(targets)
            train_losses=np.zeros(3,dtype=np.float64)
            train_bar=progress_iter(targets,total=len(targets),desc=f"Epoch {epoch:03d} Train")
            for batch_index,target in enumerate(train_bar,start=1):
                optimizer.zero_grad(set_to_none=True)
                prediction, y, price, mask = batch_forward(model, data, target, device, panel)
                loss, mse_loss, rank_loss, _ = get_loss(prediction, y, price, mask, len(data.axis.stock_uids), float(settings["alpha"]))
                if not all(torch.isfinite(value).all() for value in (loss,mse_loss,rank_loss)):
                    raise ValueError("Nonfinite training loss.")
                loss.backward(); optimizer.step()
                train_losses += [loss.item(),mse_loss.item(),rank_loss.item()]
                train_bar.set_postfix(loss=format_loss(train_losses[0]/batch_index),rank=format_loss(train_losses[2]/batch_index))
            train_losses/=len(targets)
            validation, _ = evaluate(model, data, data.split.targets("validation"), device, float(settings["alpha"]), panel,metric_protocol)
            epoch_test,test_arrays=None,None
            if per_epoch:
                epoch_test,test_arrays=evaluate(model,data,data.split.targets("test"),device,float(settings["alpha"]),panel,metric_protocol)
                test_evaluations+=1
            row={"epoch":epoch,"train_loss":float(train_losses[0]),"train_mse_loss":float(train_losses[1]),"train_rank_loss":float(train_losses[2])}
            for prefix,perf in (("val",validation),("test",epoch_test)):
                if perf is not None:
                    for key,source in (("loss","loss"),("mse_loss","mse_loss"),("rank_loss","rank_loss"),
                        ("mse","mse"),("ic","IC"),("ric","RIC"),("prec_10","prec_10"),("sharpe5","sharpe5")):
                        row[prefix+"_"+key]=perf.get(source)
            history.append(row)
            checkpoint = dict(common, epoch=epoch, validation=validation, epoch_test=epoch_test, model_state_dict=model.state_dict(),
                              optimizer_state_dict=optimizer.state_dict())
            atomic_checkpoint(checkpoint, directory / "last_model.pt")
            if validation["loss"] < best:
                best = validation["loss"]
                best_epoch=epoch
                atomic_checkpoint(checkpoint, directory / "best_model.pt")
                if per_epoch:
                    best_test=epoch_test
                    np.savez(directory/"test_predictions.npz",prediction=test_arrays[0],truth=test_arrays[1],mask=test_arrays[2],target_indices=data.split.targets("test"))
                    write_json(directory/"test_metrics.json",best_test)
            epoch_time=time.time()-epoch_start
            epoch_times.append(epoch_time)
            estimated_remaining=float(np.mean(epoch_times))*max(0,int(settings["epochs"])-epoch)
            row["epoch_time_sec"]=epoch_time
            write_json(directory / "history.json", history)
            if original:
                write_json(directory/"epoch_samples.json",epoch_samples)
            print(compact_epoch_summary(epoch,int(settings["epochs"]),float(train_losses[0]),validation["loss"],validation,
                epoch_test or {"IC":None,"RIC":None,"sharpe5":None},epoch_time,estimated_remaining,best_epoch,"val_loss",validation["loss"]),flush=True)
            refresh_training(directory,output)
        checkpoint = torch.load(directory / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if per_epoch:
            # Reuse the test result of the validation-selected epoch; no extra test or test-driven selection.
            test=best_test
        else:
            test, arrays = evaluate(model, data, data.split.targets("test"), device, float(settings["alpha"]), panel,metric_protocol)
            test_evaluations+=1
            np.savez(directory / "test_predictions.npz", prediction=arrays[0], truth=arrays[1], mask=arrays[2],target_indices=data.split.targets("test"))
            write_json(directory / "test_metrics.json", test)
        write_json(directory / "run_status.json", {"status":"completed","finished_at":now()})
        manifest = {k: v for k, v in common.items() if k != "config"}
        manifest.update(kind="strict_training_run", created_at=now(), best_epoch=checkpoint["epoch"],
            epochs=int(settings["epochs"]), final_test_evaluations=0 if per_epoch else 1,
            test_evaluations=test_evaluations,epoch_test_evaluations=test_evaluations if per_epoch else 0,
            test=test, validation=checkpoint["validation"],
            environment={"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                         "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
            metric_definitions={"protocol":metric_protocol,"IC":"official Pearson on prediction*mask and truth*mask, including zeroed invalid rows" if metric_protocol=="stockmixer_original" else "valid-only daily Pearson mean",
                "RIC":"daily Pearson mean/std, ddof=0; not Spearman",
                "sharpe5":"official top5 mean/std * 15.87" if metric_protocol=="stockmixer_original" else "top5 mean/std * sqrt(252)"},
            files=file_manifest(directory))
        write_json(directory / "manifest.json", manifest)
        print_training_footer(output,manifest['best_epoch'],test,time.time()-start_time,epoch_times)
        publish_training(directory,output,data_contract=getattr(data,"manifest",None))
    return manifest


def predict(checkpoint_path, output=None, as_of="latest", device="auto", allow_replay=False, data_dir=None, panel_path=None, dataset=None, attach_to_training=False):
    checkpoint = load_run_checkpoint(checkpoint_path)
    data = MarketData(data_dir or checkpoint["data_directory"])
    if dataset is not None and dataset!=checkpoint["dataset"]:
        raise ValueError("Requested prediction dataset differs from checkpoint.")
    if output is None:
        output=destination(PROJECT,dataset=data.axis.dataset,kind="prediction")
    if attach_to_training:
        output=Path(output).resolve()
        bundle=verify_bundle(output)
        if bundle["kind"]!="training_bundle" or digest_file(output/"checkpoints/best_model.pt")!=digest_file(checkpoint_path):
            raise ValueError("Integrated prediction must use this training run's selected best checkpoint.")
        if (output/"_prediction_artifact").exists() or (output/"predictions/predictions.csv").exists():
            raise FileExistsError("This training run already contains a forecast; do not overwrite it.")
    date = data.axis.dates[-1] if as_of == "latest" else as_of
    if date not in data.axis.dates:
        raise ValueError("as-of must be a date on the verified data axis; latest means dataset end, not today.")
    if date <= checkpoint["selection_end_date"] and not allow_replay:
        raise ValueError("Checkpoint training/selection uses later or same-date data. Use an earlier checkpoint or explicitly request in-sample replay.")
    config = checkpoint["config"]
    if checkpoint["mode"]=="baseline" and panel_path is not None:
        raise ValueError("A baseline checkpoint cannot accept a news panel.")
    panel = load_panel(panel_path or checkpoint["panel_directory"], data, config,require_training_activity=False) if checkpoint["mode"] == "fusion" else None
    compatibility=validate_checkpoint(checkpoint, data, panel)
    device = choose_device(device)
    model = make_model(data, config, device, checkpoint["mode"] == "fusion")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True); model.eval()
    decision = data.axis.dates.index(date)
    x, price, mask = [tensor(a, device) for a in data.decision(decision)]
    with torch.inference_mode():
        if panel is None:
            base = prediction = model(x)
            deltas = {c: torch.zeros_like(price) for c in ("company", "industry", "market")}
            news = {}
        else:
            news = {k: tensor(v, device) for k, v in panel.decision(decision).items() if k != "decision_idx"}
            prediction, details = model(x, price, **news, return_details=True)
            base, deltas = details["baseline_price"], details["deltas"]
    target_date, target_basis = None, "calendar unavailable; target date not guessed"
    if decision+1 < len(data.axis.dates):
        target_date, target_basis = data.axis.dates[decision+1], "verified dataset axis"
    else:
        try:
            import exchange_calendars as xcals
            calendar = xcals.get_calendar("XNYS", start=date, end=(pd.Timestamp(date)+pd.Timedelta(days=20)).strftime("%Y-%m-%d"))
            target_date, target_basis = calendar.next_session(date).strftime("%Y-%m-%d"), "exchange_calendars:XNYS"
        except ImportError:
            pass
    with (data.directory / "stock_identity.csv").open(encoding="utf-8-sig", newline="") as stream:
        identities = list(csv.DictReader(stream))
    rows = []
    for i, identity in enumerate(identities):
        row = {"dataset": data.axis.dataset, "as_of_date": date, "target_date": target_date,
            "array_index": i, "stock_uid": identity["stock_uid"], "ticker": identity["ticker"],
            "prediction_valid": bool(mask[i].item()), "price_space": "dataset-normalized price; not USD" if checkpoint["protocol"]=="original_reproduction" else "train-max normalized price; not USD",
            "baseline_pred_price": base[i].item(), "final_pred_price": prediction[i].item(),
            "baseline_pred_return": ((base[i]-price[i])/price[i]).item(),
            "final_pred_return": ((prediction[i]-price[i])/price[i]).item()}
        for channel in deltas:
            row[channel+"_delta_return"] = deltas[channel][i].item()
            nm = news.get(channel+"_mask")
            row[channel+"_news_mask"] = bool(nm[min(i, len(nm)-1)].item()) if nm is not None else False
        rows.append(row)
    artifact_target=output/"_prediction_artifact" if attach_to_training else output
    with new_artifact(artifact_target) as outer:
        directory=outer if attach_to_training else outer/"_artifact"
        if not attach_to_training:
            directory.mkdir()
        with (directory / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        result = {"schema_version": SCHEMA, "kind": "strict_prediction", "protocol":checkpoint["protocol"], "created_at": now(), "dataset": data.axis.dataset,
            "as_of_date": date, "target_date": target_date, "target_date_basis": target_basis,
            "rows": len(rows), "valid_rows": sum(r["prediction_valid"] for r in rows),
            "in_sample_replay": date <= checkpoint["selection_end_date"], "smoke_checkpoint": checkpoint["smoke_only"],
            "historical_model_replay":True,"news_time_mode":config["fusion"]["time_mode"] if panel else None,
            "config":config,"stock_axis_hash":data.axis.stock_hash,"date_axis_hash":data.axis.date_hash,
            "checkpoint_sha256": digest_file(checkpoint_path), "data_manifest_sha256": checkpoint["data_manifest_sha256"],
            "checkpoint_code_compatibility":compatibility,"inference_code_sha256":code_hashes(),
            "panel_manifest_sha256": checkpoint["panel_manifest_sha256"], "files": file_manifest(directory)}
        write_json(directory / "manifest.json", result)
        if not attach_to_training:
            publish_prediction(directory,outer)
    if attach_to_training:
        publish_training_prediction(output/"_prediction_artifact",output)
    return result


def main(argv=None, required_mode=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", choices=("NASDAQ", "NYSE", "SP500"))
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "audit-news"):
        commands.add_parser(command).add_argument("--output", type=Path)
    fit = commands.add_parser("train")
    fit.add_argument("--data", type=Path, required=True); fit.add_argument("--output", type=Path)
    fit.add_argument("--epochs", type=int); fit.add_argument("--smoke", action="store_true")
    fit.add_argument("--baseline", type=Path); fit.add_argument("--panel", type=Path)
    pred = commands.add_parser("predict")
    pred.add_argument("--checkpoint", type=Path, required=True); pred.add_argument("--output", type=Path)
    pred.add_argument("--as-of", default="latest"); pred.add_argument("--device", default="auto")
    pred.add_argument("--allow-in-sample-replay", action="store_true")
    pred.add_argument("--data",type=Path,help="Relocated byte-identical prepared data artifact.")
    pred.add_argument("--panel",type=Path,help="Relocated byte-identical news panel.")
    args = parser.parse_args(argv)
    if required_mode=="predict" and args.command!="predict":
        parser.error("Prediction mode only supports the predict subcommand.")
    if args.command=="train" and required_mode=="fusion" and args.baseline is None:
        parser.error("Fusion training requires --baseline <strict E0 checkpoint>.")
    if args.command=="train" and required_mode=="baseline" and args.baseline is not None:
        parser.error("Set llm.enabled=true in configs/stockmixer.yaml for fusion training.")
    if args.command == "predict":
        if args.config is not None or args.project_root is not None:
            parser.error("Prediction uses the checkpoint config. Relocate artifacts with predict --data/--panel.")
        output=run_destination(PROJECT,args.output,dataset=args.dataset or load_run_checkpoint(args.checkpoint)["dataset"],kind="prediction") if args.output is not None else None
        result = predict(args.checkpoint, output, args.as_of, args.device, args.allow_in_sample_replay,
                         data_dir=args.data,panel_path=args.panel,dataset=args.dataset)
    else:
        project, config = configuration(args.config, args.dataset, args.project_root)
        category={"prepare":"artifacts/market","audit-news":"reports"}.get(args.command)
        output_factory=run_destination if args.command in ("train","audit-news") else destination
        kind=("fusion" if args.baseline else "E0")+("_smoke" if args.smoke else "") if args.command=="train" else args.command
        output=output_factory(project,args.output,dataset=config["dataset"],kind=kind,category=category)
        if args.command == "prepare":
            result = prepare_market(project, config, output)
        elif args.command == "audit-news":
            result = audit_news(project, config)
            with new_artifact(output) as directory:
                write_json(directory / "news_snapshot_audit.json", result)
        else:
            result = train(project, config, args.data, output, args.epochs, args.smoke, args.baseline, args.panel)
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "current_source_hashes", "protected_source_hashes")},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ImportError, KeyError) as error:
        raise SystemExit("Strict experiment stopped: " + str(error))
