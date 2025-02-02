import evaluate
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import json
import datetime
from collections import defaultdict

DEVICE = "cuda" if torch.cuda.is_available() else "CPU"



import numpy as np
from sklearn.metrics import f1_score, mean_squared_error

# FPB FiQA-SA
def acc(items):
    return items

def acc_agg(items):
    return sum(items) / len(items)

# FPB FiQA-SA
def f1(items, references=None):
    from lm_eval.api.task import TaskManager
    task = TaskManager.get_current_task()
    golds = [doc["gold"] for doc in task.dataset["test"]]
    preds = [pred for pred in items]  # 假设items是预测的类别索引
    return list(zip(golds, preds))

def f1_agg(items):
    golds, preds = zip(*items)
    return f1_score(golds, preds, average="macro")


# TSA
def rmse(references, predictions):
    items = []
    for gold, pred_text in zip(references, predictions):
        try:
            pred = float(re.findall(r"-?\d+\.?\d*", pred_text)[0])
            pred = np.clip(pred, -1.0, 1.0)
        except (IndexError, ValueError):
            pred = None
        items.append((gold, pred))
    return items

# TSA
def rmse_agg(items):
    valid_pairs = [(g, p) for g, p in items if p is not None]
    if not valid_pairs:
        return float("inf")
    golds, preds = zip(*valid_pairs)
    return np.sqrt(mean_squared_error(golds, preds))

# TSA
def missing_rate(references, predictions):
    missing_count = sum(1 for pred in predictions if pred is None)
    return missing_count / len(predictions)

def missing_rate_agg(items):
    return sum(items) / len(items)



