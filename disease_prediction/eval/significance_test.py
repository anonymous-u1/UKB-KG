# significance_tests_parallel.py

import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from MLstatkit import Permutation_test
from disease_prediction.eval.MLstatkit.custom_permutation import Permutation_test_new


def load_model_results(model_dir):
    """Load truth/probs/thresholds for one model."""
    return {
        "truth": np.load(os.path.join(model_dir, "truth.npy"), allow_pickle=True).item(),
        "probs": np.load(os.path.join(model_dir, "probs.npy"), allow_pickle=True).item(),
        "thresholds": np.load(os.path.join(model_dir, "thresholds.npy"), allow_pickle=True).item()
    }


def load_train_col_sum(model_dir):
    df = pd.read_csv(os.path.join(model_dir, "metrics.csv"))
    return dict(zip(df["Unnamed: 0"], df["train_col_sum"]))


def eval_one_label(label, modelA, modelB, train_col_sum_dict):
    y = modelA["truth"][label]
    prob_A = modelA["probs"][label]
    prob_B = modelB["probs"][label]
    thr_A = modelA["thresholds"][label]
    thr_B = modelB["thresholds"][label]

    # AUROC
    auc_a, auc_b, p_auc, diff_auc, _, _ = Permutation_test(
        y, prob_A, prob_B, metric_str="roc_auc"
    )

    # AUPRC
    prc_a, prc_b, p_prc, diff_prc, _, _ = Permutation_test(
        y, prob_A, prob_B, metric_str="pr_auc"
    )

    # F1
    f1_a, f1_b, p_f1, diff_f1, _, _ = Permutation_test_new(
        y, prob_A, prob_B, metric_str="f1",
        threshold_A=thr_A, threshold_B=thr_B, average="binary"
    )

    return [
        label,
        train_col_sum_dict[label],
        auc_a, auc_b, p_auc, diff_auc,
        prc_a, prc_b, p_prc, diff_prc,
        f1_a, f1_b, p_f1, diff_f1
    ]


def compare_two_models_parallel(modelA_dir, modelB_dir, output_csv, num_workers=None):

    modelA = load_model_results(modelA_dir)
    modelB = load_model_results(modelB_dir)
    train_col_sum = load_train_col_sum(modelA_dir)

    labels = sorted(modelA["truth"].keys())

    if num_workers is None:
        num_workers = max(2, cpu_count() - 2)

    print(f"🚀 Using {num_workers} parallel workers for significance test...")

    with Pool(processes=num_workers) as pool:
        func = partial(eval_one_label, modelA=modelA, modelB=modelB, train_col_sum_dict=train_col_sum)
        results = pool.map(func, labels)

    df = pd.DataFrame(results, columns=[
        "label", "train_col_sum",
        "AUROC_A", "AUROC_B", "AUROC_p", "AUROC_diff",
        "AUPRC_A", "AUPRC_B", "AUPRC_p", "AUPRC_diff",
        "F1_A", "F1_B", "F1_p", "F1_diff",
    ])
    df.to_csv(output_csv, index=False)
    print(f"Saved significance results → {output_csv}")

    bins = [(0,10), (10,100), (100,500), (500, np.inf)]

    rows = []
    for (low, high) in bins:
        subset = df[(df["train_col_sum"] > low) & (df["train_col_sum"] <= high)]
        if high == np.inf:
            label_bin = f"{low}+"
        else:
            label_bin = f"{low}-{high}"

        rows.append([
            label_bin,
            subset["AUROC_A"].mean(), subset["AUROC_B"].mean(), subset["AUROC_p"].mean(), subset["AUROC_diff"].mean(),
            subset["AUPRC_A"].mean(), subset["AUPRC_B"].mean(), subset["AUPRC_p"].mean(), subset["AUPRC_diff"].mean(),
            subset["F1_A"].mean(),    subset["F1_B"].mean(),    subset["F1_p"].mean(),    subset["F1_diff"].mean(),
            len(subset)
        ])

    df_mean = pd.DataFrame(rows, columns=[
        "bin",
        "mean_AUROC_A", "mean_AUROC_B", "mean_AUROC_p", "mean_AUROC_diff",
        "mean_AUPRC_A", "mean_AUPRC_B", "mean_AUPRC_p", "mean_AUPRC_diff",
        "mean_F1_A", "mean_F1_B", "mean_F1_p", "mean_F1_diff",
        "count"
    ])
    df_mean.to_csv(output_csv.replace(".csv", "_mean.csv"), index=False)
    print(f"Saved binned mean → {output_csv.replace('.csv', '_mean.csv')}")


if __name__ == "__main__":
    modelA_dir = f"disease_prediction/save/nn5_binary/test"
    modelB_dir = f"disease_prediction/save/nn5_ModE_ene/test"
    output_csv = os.path.join(modelB_dir, "significance.csv")

    compare_two_models_parallel(modelA_dir, modelB_dir, output_csv)
