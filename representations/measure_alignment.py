import os
import argparse 

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

import metrics
from pprint import pprint



def prepare_features(feats, q=0.95, exact=False):
    """
    Prepare features by removing outliers and normalizing
    Args:
        feats: a torch tensor of any share
        q: the quantile to remove outliers
    Returns:
        feats: a torch tensor of the same shape as the input
    """
    if isinstance(feats, torch.Tensor):
        feats = metrics.remove_outliers(feats.float(), q=q, exact=exact)
        return feats
    elif isinstance(feats, list):
        return [metrics.remove_outliers(f.float(), q=q, exact=exact) for f in feats]
    else:
        raise ValueError(f"Unsupported input type for prepare_features: {type(feats)}")


def compute_score(x_feats, y_feats, metric="mutual_knn", topk=10, normalize=True, x_shape=None):
    """
    Uses different layer combinations of x_feats and y_feats to find the best alignment
    Args:
        x_feats: a torch tensor of shape N x L x D
        y_feats: a torch tensor of shape N x L x D
    Returns:
        best_alignment_score: the best alignment score
        best_alignment: the indices of the best alignment
    """

    raw = 'raw' in metric
    if isinstance(x_feats, torch.Tensor):
        x_feats = [x_feats[:, i, :] for i in range(x_feats.shape[1])]

    if isinstance(y_feats, torch.Tensor):
        y_feats = [y_feats[:, j, :] for j in range(y_feats.shape[1])]

    best_alignment_indices = None
    best_alignment_score = 0 if not raw else np.zeros(x_shape)

    for i, x in enumerate(x_feats):
        for j, y in enumerate(y_feats):
            if normalize:
                x_aligned = F.normalize(x, p=2, dim=-1)
                y_aligned = F.normalize(y, p=2, dim=-1)
            else:
                x_aligned = x
                y_aligned = y

            kwargs = {}
            if 'knn' in metric:
                kwargs['topk'] = topk

            score = metrics.AlignmentMetrics.measure(metric, x_aligned, y_aligned, **kwargs)
            
            if not raw:
                if score > best_alignment_score:
                    best_alignment_score = score
                    best_alignment_indices = (i, j)
            else:
                best_alignment_score = np.array([max(score[k], best_alignment_score[k]) for k in range(len(score))])
                best_alignment_indices = (-1,-1) # not tracked for raw
    return best_alignment_score, best_alignment_indices

    
def compute_alignment(x_feat_paths, y_feat_paths, metric, topk, precise=True, output_dir="results/alignment"):
    """
    Args:
        x_feat_paths: list of paths to x features
        y_feat_paths: list of paths to y features
        metric: the metric to use
        topk: the number of nearest neighbors to use (specific to knn metrics)
        precise: if true use exact quantiling. (helpful to set to false if running on cpu)
            this is more of a feature to speed up matmul if using float32 
            used in measure_alignment.py
    Returns:
        alignment_scores: a numpy array of shape len(x_feat_paths) x len(y_feat_paths)
        alignment_indices: a numpy array of shape len(x_feat_paths) x len(y_feat_paths) x 2
    """
    
    os.makedirs(output_dir, exist_ok=True)

    symmetric_metric = (x_feat_paths == y_feat_paths)
    if metric == "cycle_knn":
        symmetric_metric = False

    alignment_scores = np.zeros((len(x_feat_paths), len(y_feat_paths)))
    alignment_indices = np.zeros((len(x_feat_paths), len(y_feat_paths), 2))

    pbar = tqdm(total=len(y_feat_paths) * len(x_feat_paths))

    for i, x_fp in enumerate(x_feat_paths):
        # raw_x = torch.load(x_fp, map_location="cuda:0")["feats"]
        raw_x = torch.load(x_fp)["feats"]
        if isinstance(raw_x, torch.Tensor):
            x_feats = prepare_features(raw_x.float(), exact=precise)
        else:
            x_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_x]
        
        # x_feats = prepare_features(torch.load(x_fp, map_location="cuda:0")["feats"].float(), exact=precise)
            
        for j, y_fp in enumerate(y_feat_paths):
            if symmetric_metric:
                if i > j:
                    pbar.update(1)
                    continue           

            # raw_y = torch.load(y_fp, map_location="cuda:0")["feats"]
            raw_y = torch.load(y_fp)["feats"]
            if isinstance(raw_y, torch.Tensor):
                y_feats = prepare_features(raw_y.float(), exact=precise)
            else:
                y_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_y]
            best_score, best_indices = compute_score(y_feats, x_feats, metric=metric, topk=topk)
            
            alignment_scores[i, j] = best_score
            alignment_indices[i, j] = best_indices
            
            if symmetric_metric:
                alignment_scores[j, i] = best_score
                alignment_indices[j, i] = best_indices[::-1]

            pbar.update(1)

            del y_feats
            torch.cuda.empty_cache()

    return alignment_scores, alignment_indices
