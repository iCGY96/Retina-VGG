import os
import glob
import shutil
import requests
import zipfile
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

########################################
# Metric functions
########################################

def compute_cc(pred, gt):
    """
    Compute the Pearson Correlation Coefficient (CC) between two maps.
    """
    pred_mean = np.mean(pred)
    gt_mean = np.mean(gt)
    numerator = np.sum((pred - pred_mean) * (gt - gt_mean))
    denominator = np.sqrt(np.sum((pred - pred_mean) ** 2) * np.sum((gt - gt_mean) ** 2)) + 1e-8
    return numerator / denominator

def compute_similarity(pred, gt):
    """
    Compute the similarity metric between two maps.
    Both maps are normalized to sum to 1, and the element-wise minimum is summed.
    """
    pred_norm = pred / (np.sum(pred) + 1e-8)
    gt_norm = gt / (np.sum(gt) + 1e-8)
    return np.sum(np.minimum(pred_norm, gt_norm))

def compute_kl_div(pred, gt):
    """
    Compute the Kullback-Leibler divergence between two maps.
    Both maps are normalized to sum to 1.
    """
    eps = 1e-8
    pred_norm = pred / (np.sum(pred) + eps)
    gt_norm = gt / (np.sum(gt) + eps)
    return np.sum(gt_norm * np.log((gt_norm + eps) / (pred_norm + eps)))

def compute_emd(pred, gt, downsize=32):
    """
    Compute the Earth Mover's Distance (EMD) between two saliency maps.

    Steps:
    1. Resize both the prediction and ground truth maps to a lower resolution (downsize x downsize) to speed up computation.
    2. Construct a signature for each pixel in the form [weight, x, y].
    3. Use OpenCV's EMD function with L2 distance to compute the optimal transport cost between the two distributions.
    """
    # Resize the saliency maps to a lower resolution
    pred_resized = cv2.resize(pred, (downsize, downsize), interpolation=cv2.INTER_LINEAR)
    gt_resized = cv2.resize(gt, (downsize, downsize), interpolation=cv2.INTER_LINEAR)
    pred_resized = pred_resized.astype(np.float32)
    gt_resized = gt_resized.astype(np.float32)
    pred_resized /= (pred_resized.sum() + 1e-8)
    gt_resized /= (gt_resized.sum() + 1e-8)

    h, w = pred_resized.shape
    signature1 = []
    signature2 = []
    for i in range(h):
        for j in range(w):
            signature1.append([pred_resized[i, j], float(i), float(j)])
            signature2.append([gt_resized[i, j], float(i), float(j)])
    signature1 = np.array(signature1, dtype=np.float32)
    signature2 = np.array(signature2, dtype=np.float32)
    
    # cv2.EMD returns a tuple (emd, lowerBound, flow)
    emd_value, _, _ = cv2.EMD(signature1, signature2, cv2.DIST_L2)
    return emd_value

def compute_nss(pred, fixation):
    """
    Compute the Normalized Scanpath Saliency (NSS).

    Steps:
    1. Normalize the prediction map by subtracting its mean and dividing by its standard deviation.
    2. Compute the average normalized value at the fixation locations (where fixation > 0).
    """
    std = np.std(pred)
    if std < 1e-8:
        return np.nan
    pred_norm = (pred - np.mean(pred)) / std
    if np.sum(fixation) == 0:
        return np.nan
    return np.mean(pred_norm[fixation > 0])

def compute_auc_judd(pred, fixation, eps=1e-7):
    """
    Compute the AUC Judd metric.

    Steps:
    1. Use the saliency values at fixation locations as thresholds.
    2. Construct the ROC curve by computing the True Positive Rate (TPR) and False Positive Rate (FPR).
    3. Compute the area under the ROC curve.
    """
    pred_flat = pred.flatten()
    fixation_flat = fixation.flatten().astype(bool)
    pos = pred_flat[fixation_flat]
    num_fix = len(pos)
    if num_fix == 0:
        return np.nan
    thresholds = np.sort(pos)[::-1]
    tpr, fpr = [], []
    num_pixels = len(pred_flat)
    num_nonfix = num_pixels - num_fix
    for thresh in thresholds:
        tpr_val = np.sum(pos >= thresh) / float(num_fix)
        fpr_val = np.sum(pred_flat[~fixation_flat] >= thresh) / float(num_nonfix + eps)
        tpr.append(tpr_val)
        fpr.append(fpr_val)
    return np.trapz(tpr, fpr)

def compute_auc_borji(pred, fixation, num_random=100, steps=100):
    """
    Compute the AUC Borji metric.

    Steps:
    1. Normalize the prediction map to the range [0, 1].
    2. For each random trial, randomly select negative samples from non-fixation locations,
       where the number of negatives equals the number of fixation points.
    3. Construct the ROC curve and compute the area under the curve (AUC).
    4. Return the average AUC over all random trials.
    """
    # Normalize the saliency map
    saliency = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    fixation = fixation.astype(bool)
    num_fixations = np.sum(fixation)
    if num_fixations == 0:
        return np.nan

    saliency_flat = saliency.flatten()
    fixation_flat = fixation.flatten()
    pos_scores = saliency_flat[fixation_flat]
    all_indices = np.arange(saliency_flat.shape[0])
    neg_indices = all_indices[~fixation_flat]
    if len(neg_indices) < num_fixations:
        return np.nan
    aucs = []
    for _ in range(num_random):
        rand_neg = np.random.choice(neg_indices, size=num_fixations, replace=False)
        neg_scores = saliency_flat[rand_neg]
        thresh_min = min(pos_scores.min(), neg_scores.min())
        thresh_max = max(pos_scores.max(), neg_scores.max())
        thresholds = np.linspace(thresh_min, thresh_max, steps)
        tpr, fpr = [], []
        for thresh in thresholds:
            tpr.append(np.sum(pos_scores >= thresh) / float(num_fixations))
            fpr.append(np.sum(neg_scores >= thresh) / float(num_fixations))
        aucs.append(np.trapz(tpr, fpr))
    return np.mean(aucs)

def compute_auc_shuffled(pred, fixation, other_fixations, num_random=100, steps=100):
    """
    Compute the AUC Shuffled metric.

    Steps:
    1. Normalize the prediction map to the range [0, 1].
    2. Use other_fixations (binarized) as the source of negative samples.
    3. For each random trial, randomly select negatives from other_fixations,
       matching the number of fixation points.
    4. Construct the ROC curve and compute the area under the curve (AUC).
    5. Return the average AUC over all random trials.
    """
    saliency = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    fixation = fixation.astype(bool)
    num_fixations = np.sum(fixation)
    if num_fixations == 0:
        return np.nan

    saliency_flat = saliency.flatten()
    fixation_flat = fixation.flatten()
    pos_scores = saliency_flat[fixation_flat]
    other_fixations = other_fixations.astype(bool)
    neg_indices = np.where(other_fixations.flatten())[0]
    if len(neg_indices) < num_fixations:
        return np.nan
    aucs = []
    for _ in range(num_random):
        rand_neg = np.random.choice(neg_indices, size=num_fixations, replace=False)
        neg_scores = saliency_flat[rand_neg]
        thresh_min = min(pos_scores.min(), neg_scores.min())
        thresh_max = max(pos_scores.max(), neg_scores.max())
        thresholds = np.linspace(thresh_min, thresh_max, steps)
        tpr, fpr = [], []
        for thresh in thresholds:
            tpr.append(np.sum(pos_scores >= thresh) / float(num_fixations))
            fpr.append(np.sum(neg_scores >= thresh) / float(num_fixations))
        aucs.append(np.trapz(tpr, fpr))
    return np.mean(aucs)

def evaluate_metrics(pred, gt):
    """
    Evaluate multiple saliency metrics:
    CC, Similarity, EMD, KLdiv, NSS, AUC Judd, AUC Borji, and AUC Shuffled.
    
    Parameters:
    - pred and gt are expected to be in the range [0, 1].
    - The fixation map is generated by thresholding gt at 50% of its maximum value.
    """
    cc = compute_cc(pred, gt)
    sim = compute_similarity(pred, gt)
    emd = compute_emd(pred, gt)
    kl_div = compute_kl_div(pred, gt)
    fixmap = (gt >= 0.5 * np.max(gt)).astype(np.float32)
    nss = compute_nss(pred, fixmap)
    auc_judd = compute_auc_judd(pred, fixmap)
    auc_borji = compute_auc_borji(pred, fixmap)
    # For AUC Shuffled, use a randomly shuffled version of fixmap as negative samples
    fixmap_shuffled = np.random.permutation(fixmap.flatten()).reshape(fixmap.shape)
    auc_shuffled = compute_auc_shuffled(pred, fixmap, fixmap_shuffled)
    return {
        'CC': cc,
        'Similarity': sim,
        'EMD': emd,
        'KLdiv': kl_div,
        'NSS': nss,
        'AUCJudd': auc_judd,
        'AUCBorji': auc_borji,
        'AUCshuffled': auc_shuffled
    }
