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
    
# FPB FiQA-SA
def acc_agg(items):
    return sum(items) / len(items)

# FPB FiQA-SA
def f1(items, references=None):
    from lm_eval.api.task import TaskManager
    task = TaskManager.get_current_task()
    golds = [doc["gold"] for doc in task.dataset["test"]]
    preds = [pred for pred in items]
    return list(zip(golds, preds))

# FPB FiQA-SA
def f1_agg(items):
    golds, preds = zip(*items)
    return f1_score(golds, preds, average="macro")


# TSA
NUM_PATTERN = re.compile(r"-?\d+\.?\d*")
def rmse(references, predictions):
    items = []
    for gold_str, pred_text in zip(references, predictions):
        try:
            gold = float(gold_str)
        except ValueError:
            gold = 0.0
        
        pred = None
        try:
            match = NUM_PATTERN.search(str(pred_text))
            if match:
                pred = float(match.group())
                pred = np.clip(pred, -1.0, 1.0)
        except (ValueError, TypeError):
            pass
        

        items.append( (gold, pred) )  
    return items  # List[Tuple[float, Optional[float]]]
    
# TSA
def rmse_agg(items):
    flattened = []
    for item in items:
        if isinstance(item, list): 
            flattened.extend(item)
        else:
            flattened.append(item)
    

    if not all(isinstance(it, tuple) and len(it)==2 for it in flattened):
        bad_sample = next((it for it in flattened if not isinstance(it, tuple)), None)
        raise ValueError(f"Invalid item format. Sample: {bad_sample}")
    
    valid_pairs = [it for it in flattened if it[1] is not None]
    if not valid_pairs:
        return float("inf")
    
    golds, preds = zip(*valid_pairs)
    return np.sqrt(mean_squared_error(golds, preds))

# TSA
def missing_rate(references, predictions):
    missing_count = sum(1 for pred in predictions if pred is None)
    return missing_count / len(predictions)

# TSA
def missing_rate_agg(items):
    return sum(items) / len(items)



