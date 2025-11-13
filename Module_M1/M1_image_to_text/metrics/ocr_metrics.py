from multiprocessing import Pool
from collections import defaultdict
from typing import List
import numpy as np
import nltk
from nltk import edit_distance
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_metrics(pred: str, gt: str, minlen: int = 4) -> dict:
    metrics = {}
    if len(pred) < minlen or len(gt) < minlen:
        return metrics

    # Normalized edit distance
    metrics["edit_dist"] = edit_distance(pred, gt) / max(len(pred), len(gt))

    # Tokenized BLEU with smoothing
    reference = gt.split()
    hypothesis = pred.split()
    smoothie = SmoothingFunction().method4
    metrics["bleu"] = sentence_bleu([reference], hypothesis, smoothing_function=smoothie)

    return metrics

def get_metrics(gt_list: List[str], pred_list: List[str], use_pool: bool = True) -> dict:
    metrics = defaultdict(list)

    if use_pool:
        with Pool() as p:
            _metrics = p.starmap(compute_metrics, zip(pred_list, gt_list))
    else:
        _metrics = [compute_metrics(p, g) for p, g in zip(pred_list, gt_list)]

    for m in _metrics:
        for key, value in m.items():
            metrics[key].append(value)

    return {key: sum(values) / len(values) for key, values in metrics.items() if values}
