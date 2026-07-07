from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

from src.common import progress_bar
from src.training import run_batch


def safe_binary_metrics(labels, preds):
    if len(labels) == 0:
        return 0.0, 0.0
    acc = accuracy_score(labels, preds)
    f_score = f1_score(labels, preds, average="weighted", zero_division=0)
    return acc, f_score


def compute_metrics(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    non_zeros = np.array([i for i, label in enumerate(labels) if label != 0])

    mae = np.mean(np.absolute(preds - labels))
    if len(preds[non_zeros]) > 1 and np.std(preds[non_zeros]) > 0 and np.std(labels[non_zeros]) > 0:
        corr = pearsonr(preds[non_zeros], labels[non_zeros])[0]
    else:
        corr = 0.0

    non_zero_preds = preds[non_zeros]
    non_zero_labels = labels[non_zeros]
    acc_non0, f_score_non0 = safe_binary_metrics(non_zero_labels >= 0, non_zero_preds >= 0)
    acc_all, f_score_all = safe_binary_metrics(labels >= 0, preds >= 0)
    rounded_preds = np.clip(np.rint(preds), -3, 3).astype(np.int64)
    rounded_labels = np.clip(np.rint(labels), -3, 3).astype(np.int64)
    acc_7 = accuracy_score(rounded_labels, rounded_preds)
    return {
        "mae": mae,
        "corr": corr,
        "accNon0": acc_non0,
        "f_scoreNon0": f_score_non0,
        "accAll": acc_all,
        "f_scoreAll": f_score_all,
        "ACC7": acc_7,
    }


def evaluate(model, data_loader, device, desc, show_progress=True):
    model.eval()
    losses = []
    preds = []
    labels = []
    with torch.no_grad():
        iterator = data_loader
        if show_progress:
            iterator = progress_bar(data_loader, desc=desc, leave=False, position=1)
        for batch in iterator:
            loss, logits, label_ids = run_batch(model, batch, device)
            losses.append(loss.detach().cpu().item())
            preds.extend(logits.detach().cpu().view(-1).numpy().tolist())
            labels.extend(label_ids.detach().cpu().view(-1).numpy().tolist())
    metrics = compute_metrics(preds, labels)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def format_metrics(prefix, metrics):
    keys = ["loss", "mae", "corr", "accNon0", "f_scoreNon0", "accAll", "f_scoreAll", "ACC7"]
    return ", ".join("{}_{}:{:.6f}".format(prefix, key, metrics[key]) for key in keys if key in metrics)


def compute_selection_score(metrics):
    return float(metrics["accNon0"]) + 1.5 * float(metrics["accAll"] + 0.2 * float(metrics["ACC7"]))
