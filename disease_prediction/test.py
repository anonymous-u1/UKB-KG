import argparse
import hashlib
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from disease_prediction.model import NN5Layer


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(1234)


class Dataset(TorchDataset):
    """Dataset for multi-label disease prediction evaluation."""

    def __init__(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Feature and label sample counts do not match: "
                f"x={x.shape[0]}, y={y.shape[0]}"
            )

        self.x_data = torch.as_tensor(x, dtype=torch.float32)
        self.y_data = torch.as_tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


def args_argument():
    parser = argparse.ArgumentParser(
        description="Multi-Disease Prediction Testing Script"
    )

    parser.add_argument(
        "--save_dir",
        required=True,
        help="Directory where model checkpoint is stored",
    )
    parser.add_argument(
        "--test_name",
        required=True,
        help="Directory name where metrics are stored",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["binary", "ene"],
        required=True,
        help="Input feature type: binary or ene",
    )

    parser.add_argument(
        "--val_feature_path",
        type=str,
        default=None,
        help="Path to validation binary feature .npy",
    )
    parser.add_argument(
        "--test_feature_path",
        type=str,
        default=None,
        help="Path to test binary feature .npy",
    )
    parser.add_argument(
        "--val_ene_path",
        type=str,
        default=None,
        help="Path to validation KGE embedding feature .npy",
    )
    parser.add_argument(
        "--test_ene_path",
        type=str,
        default=None,
        help="Path to test KGE embedding feature .npy",
    )
    parser.add_argument(
        "--train_label_path",
        type=str,
        required=True,
        help="Path to train label .npy",
    )
    parser.add_argument(
        "--val_label_path",
        type=str,
        required=True,
        help="Path to validation label .npy",
    )
    parser.add_argument(
        "--test_label_path",
        type=str,
        required=True,
        help="Path to test label .npy",
    )

    parser.add_argument(
        "-hs",
        "--hidden_size",
        type=int,
        default=150,
        help="Hidden layer size (default: 150)",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--bin_choice",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Which prevalence bins to use",
    )
    parser.add_argument(
        "--sample_idx_dir",
        type=str,
        default="disease_prediction/data/data_raw",
        help="Directory holding shared 1:10 evaluation sampling indices",
    )
    parser.add_argument(
        "--min_val_pos",
        type=int,
        default=10,
        help="Minimum number of distinct validation positives required for per-label F1-max threshold",
    )
    parser.add_argument(
        "--pooled_mode",
        type=str,
        choices=["rate", "score"],
        default="rate",
        help="Rare-label fallback: 'rate' uses pooled F1-max positive rate; "
        "'score' uses pooled absolute score threshold",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed used when sampling indices are first built",
    )
    parser.add_argument("--debug", type=int, default=0, help="Debug mode")

    return parser.parse_args()


def check_finite(name, array, chunk_size=5000):
    """Check a NumPy array for NaN, +Inf, and -Inf values in chunks."""

    nan_count = 0
    posinf_count = 0
    neginf_count = 0

    for start in range(0, array.shape[0], chunk_size):
        chunk = array[start:start + chunk_size]

        try:
            nan_count += int(np.isnan(chunk).sum())
            posinf_count += int(np.isposinf(chunk).sum())
            neginf_count += int(np.isneginf(chunk).sum())
        except TypeError as e:
            raise TypeError(
                f"{name} must contain numeric data, but got dtype={array.dtype}"
            ) from e

    if nan_count > 0 or posinf_count > 0 or neginf_count > 0:
        raise ValueError(
            f"{name} contains invalid values: "
            f"NaN={nan_count:,}, +Inf={posinf_count:,}, -Inf={neginf_count:,}"
        )

    print(f"[INFO] {name} finite-value check passed.")


def validate_data(
    train_label,
    val_feature,
    val_label,
    test_feature,
    test_label,
):
    """Validate data shapes, sample alignment, dimensions, and finite values."""

    arrays = {
        "train_label": train_label,
        "val_feature": val_feature,
        "val_label": val_label,
        "test_feature": test_feature,
        "test_label": test_label,
    }

    for name, array in arrays.items():
        if array.ndim != 2:
            raise ValueError(f"{name} must be 2D, but got shape {array.shape}")

    if val_feature.shape[0] != val_label.shape[0]:
        raise ValueError(
            "Validation feature/label sample count mismatch: "
            f"{val_feature.shape[0]} vs {val_label.shape[0]}"
        )

    if test_feature.shape[0] != test_label.shape[0]:
        raise ValueError(
            "Test feature/label sample count mismatch: "
            f"{test_feature.shape[0]} vs {test_label.shape[0]}"
        )

    if val_feature.shape[1] != test_feature.shape[1]:
        raise ValueError(
            "Validation/test feature dimension mismatch: "
            f"{val_feature.shape[1]} vs {test_feature.shape[1]}"
        )

    if not train_label.shape[1] == val_label.shape[1] == test_label.shape[1]:
        raise ValueError(
            "Train/validation/test label dimensions do not match: "
            f"{train_label.shape[1]}, {val_label.shape[1]}, {test_label.shape[1]}"
        )

    print("[INFO] Shape consistency check passed.")

    for name, array in arrays.items():
        check_finite(name, array)


def load_data(args):
    print(
        "\n============================== "
        "Data Loading "
        "=============================="
    )
    print(f"→ Data type: {args.data_type}")

    if args.data_type == "binary":
        if args.val_feature_path is None or args.test_feature_path is None:
            raise ValueError(
                "For data_type='binary', "
                "--val_feature_path and "
                "--test_feature_path are required."
            )

        val_feature = np.load(args.val_feature_path)
        test_feature = np.load(args.test_feature_path)
        val_feature_path = args.val_feature_path
        test_feature_path = args.test_feature_path

    elif args.data_type == "ene":
        if args.val_ene_path is None or args.test_ene_path is None:
            raise ValueError(
                "For data_type='ene', "
                "--val_ene_path and "
                "--test_ene_path are required."
            )

        val_feature = np.load(args.val_ene_path)
        test_feature = np.load(args.test_ene_path)
        val_feature_path = args.val_ene_path
        test_feature_path = args.test_ene_path

    else:
        raise ValueError(f"Invalid data type: {args.data_type}")

    train_label = np.load(args.train_label_path)
    val_label = np.load(args.val_label_path)
    test_label = np.load(args.test_label_path)

    validate_data(
        train_label,
        val_feature,
        val_label,
        test_feature,
        test_label,
    )

    if args.debug == 1:
        val_feature = val_feature[:1000, :50]
        test_feature = test_feature[:1000, :50]
        train_label = train_label[:1000, :50]
        val_label = val_label[:1000, :50]
        test_label = test_label[:1000, :50]

        print("[DEBUG MODE] Using small subset of data")

    print(f"Val Feature File:  {val_feature_path}")
    print(f"Test Feature File: {test_feature_path}")
    print(f"Val Feature Shape:  {val_feature.shape}")
    print(f"Test Feature Shape: {test_feature.shape}")
    print(f"Train Label Shape: {train_label.shape}")
    print(f"Val Label Shape:   {val_label.shape}")
    print(f"Test Label Shape:  {test_label.shape}")
    print("=============================================================================\n")

    return (
        train_label,
        val_feature,
        val_label,
        test_feature,
        test_label,
    )


def labels_sha256(labels, chunk_rows=4096):
    """
    Compute a SHA256 fingerprint over the label matrix shape, dtype, and content.
    """

    labels = np.asarray(labels)
    hasher = hashlib.sha256()

    hasher.update(str(labels.shape).encode("utf-8"))
    hasher.update(str(labels.dtype).encode("utf-8"))

    for start in range(0, labels.shape[0], chunk_rows):
        chunk = np.ascontiguousarray(labels[start:start + chunk_rows])
        hasher.update(chunk.view(np.uint8))

    return hasher.hexdigest()


def sampling_plan(n_pos, n_neg, neg_to_pos_ratio):
    """Return the POPDx sampling configuration for a label."""

    if n_neg / n_pos >= neg_to_pos_ratio:
        rounds = 50
        pos_each = 16
        with_replacement = n_pos < 1000
    else:
        rounds = 200
        pos_each = 64
        with_replacement = False

    if n_pos < pos_each:
        with_replacement = True

    return rounds, pos_each, with_replacement


def build_sample_idx(
    labels,
    label_cols,
    neg_to_pos_ratio=10,
    seed=1234,
):
    """Build fixed 1:10 patient sampling indices for each label."""

    rng = random.Random(seed)
    sample_idx = {}

    for i in sorted(int(x) for x in label_cols):
        y = labels[:, i]

        pos_idx = np.where(y == 1)[0].tolist()
        neg_idx = np.where(y != 1)[0].tolist()

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        rounds, pos_each, with_replacement = sampling_plan(
            len(pos_idx),
            len(neg_idx),
            neg_to_pos_ratio,
        )

        neg_each = pos_each * neg_to_pos_ratio
        neg_with_replacement = len(neg_idx) < neg_each
        selected = []

        for _ in range(rounds):
            if with_replacement:
                selected.extend(rng.choices(pos_idx, k=pos_each))
            else:
                selected.extend(rng.sample(pos_idx, pos_each))

            if neg_with_replacement:
                selected.extend(rng.choices(neg_idx, k=neg_each))
            else:
                selected.extend(rng.sample(neg_idx, neg_each))

        sample_idx[i] = np.asarray(selected, dtype=np.int32)

    return sample_idx


def _sample_idx_paths(sample_idx_dir, split, neg_to_pos_ratio):
    base = os.path.join(
        sample_idx_dir,
        f"eval_sample_idx_{split}_r{neg_to_pos_ratio}",
    )
    return base + ".npy", base + "_manifest.json"


def _validate_sample_idx(
    sample_idx,
    manifest,
    labels,
    label_cols,
    neg_to_pos_ratio,
    idx_path,
):
    """
    Validate the shared sampling cache against the current label matrix,
    expected labels, index bounds, and sampling ratio.
    """

    problems = []

    n_rows = labels.shape[0]
    n_labels = labels.shape[1]

    if int(manifest.get("n_rows", -1)) != int(n_rows):
        problems.append(
            f"n_rows {manifest.get('n_rows')} != {n_rows}"
        )

    if int(manifest.get("n_labels_total", -1)) != int(n_labels):
        problems.append(
            f"n_labels_total {manifest.get('n_labels_total')} != {n_labels}"
        )

    if int(manifest.get("neg_to_pos_ratio", -1)) != int(neg_to_pos_ratio):
        problems.append(
            f"neg_to_pos_ratio {manifest.get('neg_to_pos_ratio')} "
            f"!= {neg_to_pos_ratio}"
        )

    current_hash = labels_sha256(labels)
    cached_hash = manifest.get("labels_sha256")

    if cached_hash is None:
        problems.append(
            "The manifest is missing labels_sha256; this cache was generated by an older version of the code."
        )
    elif cached_hash != current_hash:
        problems.append(
            "The labels_sha256 is inconsistent with the current label matrix."
        )

    cached_keys = sorted(int(k) for k in sample_idx.keys())
    manifest_keys = sorted(
        int(k)
        for k in manifest.get("label_keys", [])
    )
    expected_keys = sorted(int(k) for k in label_cols)

    if manifest_keys != cached_keys:
        problems.append(
            "The manifest label_keys are inconsistent with the npy content."
        )

    if cached_keys != expected_keys:
        missing = sorted(set(expected_keys) - set(cached_keys))
        extra = sorted(set(cached_keys) - set(expected_keys))

        if missing:
            problems.append(
                f"The cache is missing {len(missing)} labels "
                f"(example: {missing[:5]})"
            )

        if extra:
            problems.append(
                f"The cache contains {len(extra)} extra labels "
                f"(example: {extra[:5]})"
            )

    n_per_label_manifest = manifest.get("n_per_label", {})

    for k, v in sample_idx.items():
        k = int(k)
        v = np.asarray(v, dtype=np.int32)

        if v.size == 0:
            problems.append(
                f"The sampling indices for label {k} are empty."
            )
            continue

        v_min = int(v.min())
        v_max = int(v.max())

        if v_min < 0 or v_max >= n_rows:
            problems.append(f"The index of label {k} is out of bounds [{v_min}, {v_max}], exceeding [0, {n_rows})")
            continue

        expected_size = n_per_label_manifest.get(str(k))

        if (
            expected_size is not None
            and int(expected_size) != int(v.size)
        ):
            problems.append(
                f"label {k} sample size "
                f"{v.size} != manifest "
                f"{expected_size}"
            )

        sampled_y = labels[v, k]

        n_pos = int(np.sum(sampled_y == 1))
        n_neg = int(np.sum(sampled_y != 1))

        if n_pos == 0:
            problems.append(
                f"The sampling results for label {k} do not contain any positive samples."
            )
            continue

        expected_neg = n_pos * neg_to_pos_ratio

        if n_neg != expected_neg:
            problems.append(f"label {k} not maintained at 1:{neg_to_pos_ratio}: pos={n_pos}, neg={n_neg}")

    if problems:
        raise RuntimeError(
            f"The shared sampling index {idx_path} does not match the current data:\n  - \n  - ".join(problems)
            + "\nPlease delete the sampling cache and its manifest, and then run it again to generate a new version of the shared index."
        )


def build_or_load_sample_idx(
    labels,
    label_cols,
    split,
    sample_idx_dir,
    neg_to_pos_ratio=10,
    seed=1234,
):
    """
    Load shared 1:10 sampling indices across models, or build and cache them
    when they do not exist.
    """

    idx_path, manifest_path = _sample_idx_paths(
        sample_idx_dir,
        split,
        neg_to_pos_ratio,
    )

    usable_cols = []

    for i in sorted(int(x) for x in label_cols):
        y = labels[:, i]

        if np.any(y == 1) and np.any(y != 1):
            usable_cols.append(int(i))

    if os.path.exists(idx_path) and os.path.exists(manifest_path):
        sample_idx = np.load(
            idx_path,
            allow_pickle=True,
        ).item()

        sample_idx = {
            int(k): np.asarray(v, dtype=np.int32)
            for k, v in sample_idx.items()
        }

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        _validate_sample_idx(
            sample_idx,
            manifest,
            labels,
            usable_cols,
            neg_to_pos_ratio,
            idx_path,
        )

        print(
            f"[sample_idx] reuse {idx_path} "
            f"({len(sample_idx)} labels, seed={manifest.get('seed')})"
        )

        return sample_idx

    print(
        f"[sample_idx] building {idx_path} "
        f"(seed={seed}) ..."
    )

    sample_idx = build_sample_idx(
        labels,
        usable_cols,
        neg_to_pos_ratio,
        seed,
    )

    manifest = {
        "split": split,
        "n_rows": int(labels.shape[0]),
        "n_labels_total": int(labels.shape[1]),
        "labels_sha256": labels_sha256(labels),
        "neg_to_pos_ratio": int(neg_to_pos_ratio),
        "seed": int(seed),
        "label_keys": [
            int(k)
            for k in sorted(sample_idx.keys())
        ],
        "n_per_label": {
            str(k): int(v.size)
            for k, v in sorted(sample_idx.items())
        },
    }

    os.makedirs(sample_idx_dir, exist_ok=True)

    # Use atomic writes because multiple model runs may start concurrently.
    tmp_idx = idx_path + f".tmp{os.getpid()}"
    tmp_manifest = manifest_path + f".tmp{os.getpid()}"

    with open(tmp_idx, "wb") as f:
        np.save(
            f,
            sample_idx,
            allow_pickle=True,
        )

    with open(tmp_manifest, "w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    if os.path.exists(idx_path) and os.path.exists(manifest_path):
        os.remove(tmp_idx)
        os.remove(tmp_manifest)

        return build_or_load_sample_idx(
            labels,
            label_cols,
            split,
            sample_idx_dir,
            neg_to_pos_ratio,
            seed,
        )

    os.replace(tmp_idx, idx_path)
    os.replace(tmp_manifest, manifest_path)

    _validate_sample_idx(
        sample_idx,
        manifest,
        labels,
        usable_cols,
        neg_to_pos_ratio,
        idx_path,
    )

    print(
        f"[sample_idx] saved {idx_path} "
        f"({len(sample_idx)} labels)"
    )

    return sample_idx


def f1_max_threshold(y, s):
    """Find the F1-max threshold on a set of labels and scores."""

    y = np.asarray(y).astype(np.int8)
    s = np.asarray(s, dtype=np.float64)

    n_pos = int(y.sum())

    if n_pos == 0 or n_pos == y.size:
        return None, False

    order = np.argsort(-s, kind="mergesort")

    y_desc = y[order]
    s_desc = s[order]

    # Evaluate thresholds only at distinct score boundaries.
    cut = np.r_[
        np.nonzero(np.diff(s_desc))[0],
        s_desc.size - 1,
    ]

    tp = np.cumsum(y_desc)[cut]
    fp = np.cumsum(1 - y_desc)[cut]
    fn = n_pos - tp

    denom = 2.0 * tp + fp + fn

    f1 = np.where(
        denom > 0,
        2.0 * tp / np.maximum(denom, 1.0),
        0.0,
    )

    k = int(np.argmax(f1))
    cand = s_desc[cut]

    if k + 1 < cand.size:
        threshold = 0.5 * (cand[k] + cand[k + 1])
    else:
        threshold = float(
            np.nextafter(
                cand[k],
                -np.inf,
            )
        )

    return float(threshold), True


def rate_threshold(s, rate):
    """Return a threshold that predicts approximately the top `rate` fraction."""

    s = np.asarray(s, dtype=np.float64)
    n = s.size

    if n == 0:
        return None

    target = max(1.0, rate * n)

    srt = np.sort(s)[::-1]

    cut = np.r_[
        np.nonzero(np.diff(srt))[0],
        n - 1,
    ]

    counts = cut + 1

    j = int(
        np.argmin(
            np.abs(counts - target)
        )
    )

    if cut[j] + 1 < n:
        return float(
            0.5
            * (
                srt[cut[j]]
                + srt[cut[j] + 1]
            )
        )

    return float(
        np.nextafter(
            srt[cut[j]],
            -np.inf,
        )
    )


def _label_pairs(
    scores,
    labels,
    label_col,
    sample_idx,
):
    """Return truth labels and scores for one label."""

    if sample_idx is None:
        return (
            labels[:, label_col],
            scores[:, label_col],
        )

    sel = sample_idx.get(int(label_col))

    if sel is None:
        return None, None

    return (
        labels[sel, label_col],
        scores[sel, label_col],
    )


def compute_thresholds(
    val_scores,
    val_labels,
    label_cols,
    val_col_sum,
    min_val_pos=10,
    sample_idx=None,
    pooled_mode="rate",
):
    """
    Select per-label F1-max thresholds on validation data and use a pooled
    fallback for rare labels.
    """

    label_cols = [int(x) for x in label_cols]

    if pooled_mode not in {"rate", "score"}:
        raise ValueError(
            f"Invalid pooled_mode: "
            f"{pooled_mode!r}"
        )

    pooled_y = []
    pooled_s = []

    for i in label_cols:
        y, s = _label_pairs(
            val_scores,
            val_labels,
            i,
            sample_idx,
        )

        if y is None:
            continue

        pooled_y.append(np.asarray(y).ravel())
        pooled_s.append(np.asarray(s).ravel())

    pooled_threshold = None
    pooled_rate = 0.5

    if pooled_y:
        pooled_s_all = np.concatenate(pooled_s)

        pooled_threshold, ok = f1_max_threshold(
            np.concatenate(pooled_y),
            pooled_s_all,
        )

        if ok:
            pooled_rate = float(
                (
                    pooled_s_all
                    >= pooled_threshold
                ).mean()
            )
        else:
            pooled_threshold = None

    if pooled_threshold is None:
        pooled_threshold = 0.5
        pooled_rate = 0.5

        print("[threshold] pooled F1-max unavailable, revert to 0.5")

    thresholds = {}
    sources = {}

    for i in label_cols:
        threshold = None

        if int(val_col_sum[i]) >= min_val_pos:
            y, s = _label_pairs(
                val_scores,
                val_labels,
                i,
                sample_idx,
            )

            if y is not None:
                threshold, ok = f1_max_threshold(y, s)

                if not ok:
                    threshold = None

        if threshold is not None:
            thresholds[i] = float(threshold)
            sources[i] = "f1max"
            continue

        if pooled_mode == "rate":
            _, s = _label_pairs(
                val_scores,
                val_labels,
                i,
                sample_idx,
            )

            if s is None:
                s = val_scores[:, i]

            threshold = rate_threshold(
                s,
                pooled_rate,
            )

            sources[i] = "pooled_rate"

        if threshold is None:
            threshold = pooled_threshold
            sources[i] = "pooled_score"

        thresholds[i] = float(threshold)

    n_pooled = sum(
        1
        for source in sources.values()
        if source != "f1max"
    )

    scope = (
        "1:10 sampled"
        if sample_idx is not None
        else "full"
    )

    print(
        f"[threshold] {scope} validation: "
        f"{len(thresholds) - n_pooled} labels "
        f"use per-label F1-max, "
        f"{n_pooled} fall back to pooled "
        f"({pooled_mode}) "
        f"(min_val_pos={min_val_pos}, "
        f"pooled_score={pooled_threshold:.6g}, "
        f"pooled_rate={pooled_rate:.4f})"
    )

    return (
        thresholds,
        sources,
        float(pooled_threshold),
    )


def sampling_test(
    test_outputs,
    test_truth,
    output_dir,
    threshold,
    train_col_sum,
    val_col_sum,
    test_col_sum,
    nonzero_col_idx,
    sample_idx,
    neg_to_pos_ratio=10,
    bin_choice=1,
):
    print(
        "\n============================== "
        "Sampling-based Evaluation "
        "=============================="
    )

    print(
        f"Setting negative-to-positive ratio "
        f"= {neg_to_pos_ratio}"
    )

    print(
        f"Using shared sampling indices "
        f"for {len(sample_idx)} labels"
    )

    precision_ = {}
    recall_ = {}
    fpr = {}
    tpr = {}

    auprc = {}
    auroc = {}
    f1 = {}

    all_probs = {}
    all_truth = {}

    test_probs_all = (
        torch.sigmoid(test_outputs)
        .cpu()
        .numpy()
    )

    test_truth_all = (
        test_truth
        .cpu()
        .numpy()
    )

    for i in tqdm(
        sorted(int(x) for x in nonzero_col_idx),
        desc="Per-label sampling evaluation",
    ):
        selected = sample_idx.get(i)

        if selected is None:
            print(f"[WARNING] label {i} No shared sampling indices, skip.")
            continue

        probs = test_probs_all[selected, i]
        truth = test_truth_all[selected, i]

        preds = (
            probs
            >= threshold[i]
        ).astype(np.float32)

        all_truth[i] = truth
        all_probs[i] = probs

        precision, recall, _ = precision_recall_curve(
            truth,
            probs,
        )

        fpr_i, tpr_i, _ = roc_curve(
            truth,
            probs,
        )

        precision_[i] = precision
        recall_[i] = recall
        fpr[i] = fpr_i
        tpr[i] = tpr_i

        auprc[i] = auc(
            recall,
            precision,
        )

        auroc[i] = roc_auc_score(
            truth,
            probs,
        )

        f1[i] = f1_score(
            truth,
            preds,
            average="binary",
        )

    print(f"✅ Sampling evaluation done for {len(auroc)} labels.")

    np.save(
        os.path.join(
            output_dir,
            "truth.npy",
        ),
        all_truth,
    )

    np.save(
        os.path.join(
            output_dir,
            "probs.npy",
        ),
        all_probs,
    )

    # Save threshold labels and values in exactly the same order.
    threshold_labels = np.asarray(
        sorted(int(i) for i in threshold.keys()),
        dtype=np.int32,
    )

    threshold_values = np.asarray(
        [
            threshold[int(i)]
            for i in threshold_labels
        ],
        dtype=np.float32,
    )

    np.save(
        os.path.join(
            output_dir,
            "threshold_labels.npy",
        ),
        threshold_labels,
    )

    np.save(
        os.path.join(
            output_dir,
            "thresholds.npy",
        ),
        threshold_values,
    )

    eval_cols = np.asarray(
        sorted(int(x) for x in nonzero_col_idx),
        dtype=np.int32,
    )

    np.save(
        os.path.join(
            output_dir,
            "probs_full_labels.npy",
        ),
        eval_cols,
    )

    np.save(
        os.path.join(
            output_dir,
            "probs_full.npy",
        ),
        test_probs_all[:, eval_cols].astype(np.float32),
    )

    train_col_sum = pd.Series(
        train_col_sum,
        name="train_col_sum",
    )

    val_col_sum = pd.Series(
        val_col_sum,
        name="val_col_sum",
    )

    test_col_sum = pd.Series(
        test_col_sum,
        name="test_col_sum",
    )

    model_auc = pd.DataFrame.from_dict(
        auroc,
        orient="index",
    )

    model_prc = pd.DataFrame.from_dict(
        auprc,
        orient="index",
    )

    model_f1 = pd.DataFrame.from_dict(
        f1,
        orient="index",
    )

    data_concat = pd.concat(
        [
            train_col_sum,
            val_col_sum,
            test_col_sum,
            model_auc,
            model_prc,
            model_f1,
        ],
        axis=1,
        join="inner",
    )

    data_concat.columns = [
        "train_col_sum",
        "val_col_sum",
        "test_col_sum",
        "auroc",
        "auprc",
        "f1",
    ]

    data_concat.to_csv(
        os.path.join(
            output_dir,
            "metrics.csv",
        )
    )

    if bin_choice == 1:
        bins = [
            (0, 10),
            (10, 100),
            (100, 1000),
            (1000, np.inf),
        ]

    elif bin_choice == 2:
        bins = [
            (0, 10),
            (10, 100),
            (100, 500),
            (500, np.inf),
        ]

    elif bin_choice == 3:
        bins = [
            (0, 10),
            (10, 50),
            (50, 100),
            (100, 500),
            (500, np.inf),
        ]

    else:
        raise ValueError(
            f"Invalid bin_choice: "
            f"{bin_choice}"
        )

    metrics_results = pd.DataFrame(
        0,
        index=range(len(bins)),
        columns=[
            "bin_label",
            "auroc",
            "auprc",
            "f1",
            "count",
        ],
    )

    for j, (low, high) in enumerate(bins):
        if high == np.inf:
            bin_label = f"{low}+"
        else:
            bin_label = f"{low}-{high}"

        metrics_results.loc[
            j,
            "bin_label",
        ] = bin_label

        subset = data_concat[
            (data_concat["train_col_sum"] > low)
            & (data_concat["train_col_sum"] <= high)
        ]

        metrics_results.loc[
            j,
            "auroc",
        ] = subset["auroc"].mean()

        metrics_results.loc[
            j,
            "auprc",
        ] = subset["auprc"].mean()

        metrics_results.loc[
            j,
            "f1",
        ] = subset["f1"].mean()

        metrics_results.loc[
            j,
            "count",
        ] = len(subset)

    metrics_results.to_csv(
        os.path.join(
            output_dir,
            "mean.csv",
        )
    )

    np.save(
        os.path.join(
            output_dir,
            "rareseeker_fpr.npy",
        ),
        fpr,
    )

    np.save(
        os.path.join(
            output_dir,
            "rareseeker_tpr.npy",
        ),
        tpr,
    )

    np.save(
        os.path.join(
            output_dir,
            "rareseeker_precision.npy",
        ),
        precision_,
    )

    np.save(
        os.path.join(
            output_dir,
            "rareseeker_recall.npy",
        ),
        recall_,
    )

    np.save(
        os.path.join(
            output_dir,
            "rareseeker_AUROC.npy",
        ),
        auroc,
    )

    np.save(
        os.path.join(
            output_dir,
            "rareseeker_AUPRC.npy",
        ),
        auprc,
    )

    np.save(
        os.path.join(
            output_dir,
            "rareseeker_f1.npy",
        ),
        f1,
    )

    print(
        f"✅ Sampling-based metrics saved "
        f"to {output_dir}\n"
    )


def test(
    train_label,
    val_feature,
    val_label,
    test_feature,
    test_label,
    model_checkpoint_loc,
    output_dir,
    use_cuda=True,
    hidden_size=150,
    batch_size=128,
    bin_choice=1,
    sample_idx_dir="disease_prediction/data/data_raw",
    min_val_pos=10,
    seed=1234,
    neg_to_pos_ratio=10,
    pooled_mode="rate",
):
    print(
        "\n============================== "
        "Model Testing "
        "=============================="
    )

    device = torch.device(
        "cuda:0"
        if (
            use_cuda
            and torch.cuda.is_available()
        )
        else "cpu"
    )

    net = NN5Layer(
        feature_num=test_feature.shape[1],
        label_num=test_label.shape[1],
        hidden_size=hidden_size,
    )

    checkpoint = torch.load(
        model_checkpoint_loc,
        map_location=device,
        weights_only=False,
    )

    net.load_state_dict(
        checkpoint["model_state_dict"]
    )

    net = net.to(device)
    net.eval()

    print(f"Model loaded from {model_checkpoint_loc}")
    print(f"Evaluating on device: {device}")

    test_data = Dataset(
        test_feature,
        test_label,
    )

    val_data = Dataset(
        val_feature,
        val_label,
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
    )

    print(
        f"Val batches: {len(val_loader)} "
        f"| Test batches: {len(test_loader)}"
    )

    def infer(loader, infer_type):
        outputs = []
        truths = []

        with torch.no_grad():
            for inputs, labels in tqdm(
                loader,
                desc=f"{infer_type} Inferencing",
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = net(inputs)

                outputs.append(output.cpu())
                truths.append(labels.cpu())

        return (
            torch.cat(outputs, dim=0),
            torch.cat(truths, dim=0),
        )

    val_outputs, _ = infer(
        val_loader,
        "Val",
    )

    test_outputs, test_truth = infer(
        test_loader,
        "Test",
    )

    print("✅ Inference completed.")

    train_col_sum = np.sum(
        train_label,
        axis=0,
    )

    val_col_sum = np.sum(
        val_label,
        axis=0,
    )

    test_col_sum = np.sum(
        test_label,
        axis=0,
    )

    # AUROC requires both positive and negative test samples.
    train_has_pos = train_col_sum > 0
    test_has_pos = test_col_sum > 0
    test_has_neg = test_col_sum < test_label.shape[0]

    nonzero_col_idx = np.where(
        train_has_pos
        & test_has_pos
        & test_has_neg
    )[0]

    if len(nonzero_col_idx) == 0:
        raise ValueError(
            "No evaluable labels found."
        )

    print(
        f"Labels to evaluate "
        f"(train>0, test>0, test negatives>0): "
        f"{len(nonzero_col_idx)}"
    )

    test_sample_idx = build_or_load_sample_idx(
        test_label,
        nonzero_col_idx,
        "test",
        sample_idx_dir,
        neg_to_pos_ratio,
        seed,
    )

    val_sample_idx = build_or_load_sample_idx(
        val_label,
        nonzero_col_idx,
        "val",
        sample_idx_dir,
        neg_to_pos_ratio,
        seed,
    )

    val_scores = (
        torch.sigmoid(val_outputs)
        .cpu()
        .numpy()
    )

    threshold, threshold_src, _ = compute_thresholds(
        val_scores,
        val_label,
        nonzero_col_idx,
        val_col_sum,
        min_val_pos=min_val_pos,
        sample_idx=val_sample_idx,
        pooled_mode=pooled_mode,
    )

    threshold_full, threshold_full_src, _ = compute_thresholds(
        val_scores,
        val_label,
        nonzero_col_idx,
        val_col_sum,
        min_val_pos=min_val_pos,
        sample_idx=None,
        pooled_mode=pooled_mode,
    )

    thr_df = pd.DataFrame(
        {
            "label": sorted(
                int(x)
                for x in nonzero_col_idx
            )
        }
    )

    thr_df["train_col_sum"] = [
        int(train_col_sum[i])
        for i in thr_df["label"]
    ]

    thr_df["val_col_sum"] = [
        int(val_col_sum[i])
        for i in thr_df["label"]
    ]

    thr_df["test_col_sum"] = [
        int(test_col_sum[i])
        for i in thr_df["label"]
    ]

    thr_df["threshold_sampled"] = [
        threshold[i]
        for i in thr_df["label"]
    ]

    thr_df["source_sampled"] = [
        threshold_src[i]
        for i in thr_df["label"]
    ]

    thr_df["threshold_full"] = [
        threshold_full[i]
        for i in thr_df["label"]
    ]

    thr_df["source_full"] = [
        threshold_full_src[i]
        for i in thr_df["label"]
    ]

    threshold_csv_path = os.path.join(
        output_dir,
        "thresholds.csv",
    )

    thr_df.to_csv(
        threshold_csv_path,
        index=False,
    )

    print(
        f"✅ Threshold detail saved to "
        f"{threshold_csv_path}"
    )

    print(
        "✅ Thresholds computed. "
        "Calculating final metrics..."
    )

    sampling_test(
        test_outputs,
        test_truth,
        output_dir,
        threshold,
        train_col_sum,
        val_col_sum,
        test_col_sum,
        nonzero_col_idx,
        test_sample_idx,
        neg_to_pos_ratio=neg_to_pos_ratio,
        bin_choice=bin_choice,
    )


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    if h > 0:
        return f"{h}h {m}m {s}s"

    if m > 0:
        return f"{m}m {s}s"

    return f"{s}s"


def main(args):
    save_dir = args.save_dir

    model_checkpoint_loc = os.path.join(
        save_dir,
        "best_classifier.pth.tar",
    )

    if args.debug == 1:
        output_dir = os.path.join(
            save_dir,
            "debug",
        )

        # Keep debug sampling caches separate because the label matrix differs.
        args.sample_idx_dir = os.path.join(
            args.sample_idx_dir,
            "debug",
        )

    else:
        output_dir = os.path.join(
            save_dir,
            args.test_name,
        )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print(
        f"Results will be saved to: "
        f"{output_dir}"
    )

    (
        train_label,
        val_feature,
        val_label,
        test_feature,
        test_label,
    ) = load_data(args)

    test(
        train_label,
        val_feature,
        val_label,
        test_feature,
        test_label,
        model_checkpoint_loc,
        output_dir,
        use_cuda=True,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        bin_choice=args.bin_choice,
        sample_idx_dir=args.sample_idx_dir,
        min_val_pos=args.min_val_pos,
        seed=args.seed,
        pooled_mode=args.pooled_mode,
    )


if __name__ == "__main__":
    start_time = time.time()

    args = args_argument()
    print(args)

    main(args)

    total_time = time.time() - start_time

    print(f"\nTotal Time Used: {format_time(total_time)}")