#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot binned AUROC / AUPRC / F1 bar charts across models.

Metrics are loaded from the sampling-based rareseeker_*.npy outputs generated
by test.py. Each bar shows the mean across Phecode labels within a prevalence
bin, with a bootstrap 95% CI for that mean.

python -m disease_prediction.eval.plot_bar
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METRICS = ["AUROC", "AUPRC", "F1"]
X_LABEL = "Number of Patients Per Phecode Label in Training"

METRIC_FILES = {
    "AUROC": "rareseeker_AUROC.npy",
    "AUPRC": "rareseeker_AUPRC.npy",
    "F1": "rareseeker_f1.npy",
}

LEGEND_NAME_MAP = {
    "binary": "Baseline",
    "ModE": "ModE-enhanced",
    "HAKE": "HAKE-enhanced",
    "ComplEx": "ComplEx-enhanced",
    "TransE": "TransE-enhanced",
    "RotatE": "RotatE-enhanced",
}


def get_bins(bin_choice):
    if bin_choice == 1:
        return [
            (0, 10, "(0, 10]"),
            (10, 100, "(10, 100]"),
            (100, 1000, "(100, 1000]"),
            (1000, np.inf, "(1000,+Inf]"),
        ]
    if bin_choice == 2:
        return [
            (0, 10, "(0, 10]"),
            (10, 100, "(10, 100]"),
            (100, 500, "(100, 500]"),
            (500, np.inf, "(500,+Inf]"),
        ]
    if bin_choice == 3:
        return [
            (0, 10, "(0, 10]"),
            (10, 50, "(10, 50]"),
            (50, 100, "(50, 100]"),
            (100, 500, "(100, 500]"),
            (500, np.inf, "(500,+Inf]"),
        ]
    raise ValueError(f"Invalid bin_choice: {bin_choice}")


def load_metric_dict(path, metric):
    """Load a rareseeker_*.npy dictionary with normalized integer label IDs."""
    data = np.load(path, allow_pickle=True).item()
    df = pd.DataFrame.from_dict(data, orient="index", columns=[metric])

    try:
        df.index = df.index.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains non-integer label IDs") from exc

    df.index.name = "label"

    if df.index.duplicated().any():
        duplicated = df.index[df.index.duplicated(keep=False)].unique().tolist()
        raise ValueError(
            f"{path} contains duplicated label IDs after integer conversion "
            f"(examples: {duplicated[:5]})"
        )

    return df.sort_index()


def load_run_train_counts(model_path):
    """Load the per-label training prevalence recorded by test.py."""
    path = os.path.join(model_path, "metrics.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Metrics file required for label/prevalence validation not found: {path}"
        )

    df = pd.read_csv(path, index_col=0)

    try:
        df.index = df.index.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains non-integer label IDs") from exc

    if df.index.duplicated().any():
        duplicated = df.index[df.index.duplicated(keep=False)].unique().tolist()
        raise ValueError(
            f"{path} contains duplicated label IDs "
            f"(examples: {duplicated[:5]})"
        )

    if "train_col_sum" not in df.columns:
        raise ValueError(f"{path} is missing required column 'train_col_sum'")

    return df["train_col_sum"].sort_index()


def validate_train_alignment(model_name, model_path, metric_frames, train_col_sum):
    """
    Verify that metric label IDs are valid for the current labels_train.npy and
    that their training prevalences match those recorded by the model run.
    """
    metric_labels = set().union(*(set(df.index) for df in metric_frames.values()))
    train_labels = set(train_col_sum.index)

    invalid = sorted(metric_labels - train_labels)
    if invalid:
        raise ValueError(
            f"{model_name}: {len(invalid)} metric label IDs are outside the "
            f"current labels_train.npy label range (examples: {invalid[:5]})"
        )

    run_train_col_sum = load_run_train_counts(model_path)
    missing = sorted(metric_labels - set(run_train_col_sum.index))
    if missing:
        raise ValueError(
            f"{model_name}: metrics.csv is missing {len(missing)} labels present "
            f"in rareseeker_*.npy (examples: {missing[:5]})"
        )

    labels = sorted(metric_labels)
    current = train_col_sum.loc[labels].to_numpy(dtype=float)
    recorded = run_train_col_sum.loc[labels].to_numpy(dtype=float)

    mismatch = ~np.isclose(current, recorded, rtol=0.0, atol=0.0, equal_nan=False)
    if mismatch.any():
        bad_labels = np.asarray(labels)[mismatch].tolist()
        raise ValueError(
            f"{model_name}: current labels_train.npy does not match the training "
            f"prevalence recorded by this run for {len(bad_labels)} labels "
            f"(examples: {bad_labels[:5]})"
        )

    print(f"  {model_name:<8} training-label alignment verified")


def validate_model_pairing(reference_sets, current_sets, reference_model, current_model):
    """Require identical valid label sets across models for each metric."""
    for metric in METRICS:
        reference = reference_sets[metric]
        current = current_sets[metric]

        if reference == current:
            continue

        missing = sorted(reference - current)
        extra = sorted(current - reference)

        details = []
        if missing:
            details.append(
                f"missing {len(missing)} labels in {current_model} "
                f"(examples: {missing[:5]})"
            )
        if extra:
            details.append(
                f"extra {len(extra)} labels in {current_model} "
                f"(examples: {extra[:5]})"
            )

        raise ValueError(
            f"{metric} label sets are not strictly paired between "
            f"{reference_model} and {current_model}: " + "; ".join(details)
        )


def bin_labels(df, bins):
    """Assign labels to prevalence bins using training positive counts."""
    frames = []
    for low, high, name in bins:
        temp = df[
            (df["train_col_sum"] > low) & (df["train_col_sum"] <= high)
        ][METRICS].copy()
        temp[X_LABEL] = name
        frames.append(temp)

    return pd.concat(frames, axis=0, ignore_index=True)


def process_model(model_name, model_path, train_col_sum, bins):
    metric_frames = {
        metric: load_metric_dict(os.path.join(model_path, filename), metric)
        for metric, filename in METRIC_FILES.items()
    }

    validate_train_alignment(
        model_name, model_path, metric_frames, train_col_sum
    )

    metrics_df = pd.concat(
        [train_col_sum, *metric_frames.values()],
        axis=1,
    )

    n_before = len(metrics_df)
    metrics_df = metrics_df.dropna(subset=METRICS, how="all")

    label_sets = {
        metric: set(metrics_df.index[metrics_df[metric].notna()].astype(int))
        for metric in METRICS
    }
    per_metric_n = {metric: len(label_sets[metric]) for metric in METRICS}

    print(
        f"  {model_name:<8} labels: {n_before} -> {len(metrics_df)} evaluated, "
        f"per-metric n = {per_metric_n}"
    )

    df = bin_labels(metrics_df, bins)
    df["Model"] = model_name
    return df, label_sets


def plot_metric_barplot_ax(
    df, metric, ax, show_xlabel=False, show_ylabel=True, bar_labels=True
):
    sns.barplot(
        data=df,
        x=X_LABEL,
        y=metric,
        hue="Model",
        errorbar=("ci", 95),
        capsize=.4,
        err_kws={"linewidth": .8},
        ax=ax,
    )

    if bar_labels:
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.3f",
                fontsize=8,
                padding=3,
                rotation=90,
            )

    ax.set_ylabel(
        metric if show_ylabel else "",
        fontdict={"weight": "bold", "size": 13},
    )
    ax.set_xlabel(
        X_LABEL if show_xlabel else "",
        fontdict={"weight": "bold", "size": 13},
    )

    if ax.get_legend() is not None:
        ax.get_legend().remove()


def main(args):
    bins = get_bins(args.bin_choice)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=0.9)
    sns.set_palette(args.palette)

    train_label = np.load(args.train_label_path)
    if train_label.ndim != 2:
        raise ValueError(
            f"{args.train_label_path} must be a 2D label matrix, "
            f"got shape {train_label.shape}"
        )

    train_col_sum = pd.Series(
        train_label.sum(axis=0),
        index=np.arange(train_label.shape[1]),
        name="train_col_sum",
    )

    print("Loading per-label metrics:")

    result_list = []
    reference_sets = None
    reference_model = None

    for model in models:
        if model == "binary":
            path = os.path.join(
                args.baseline_dir,
                f"test_bin_{args.bin_choice}",
            )
        else:
            path = os.path.join(
                f"{args.ene_dir_prefix}_{model}_ene",
                f"test_bin_{args.bin_choice}",
            )

        model_df, label_sets = process_model(
            model,
            path,
            train_col_sum,
            bins,
        )

        if reference_sets is None:
            reference_sets = label_sets
            reference_model = model
        else:
            validate_model_pairing(
                reference_sets,
                label_sets,
                reference_model,
                model,
            )

        result_list.append(model_df)

    result_df = pd.concat(result_list, axis=0).reset_index(drop=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 7.6),
        gridspec_kw={
            "width_ratios": [1, 1],
            "height_ratios": [1, 1],
        },
    )
    ax_auroc, ax_auprc = axes[0, 0], axes[0, 1]
    ax_f1, ax_legend = axes[1, 0], axes[1, 1]

    bar_labels = not args.no_bar_labels
    plot_metric_barplot_ax(
        result_df,
        "AUROC",
        ax_auroc,
        show_xlabel=False,
        bar_labels=bar_labels,
    )
    plot_metric_barplot_ax(
        result_df,
        "AUPRC",
        ax_auprc,
        show_xlabel=False,
        bar_labels=bar_labels,
    )
    plot_metric_barplot_ax(
        result_df,
        "F1",
        ax_f1,
        show_xlabel=True,
        bar_labels=bar_labels,
    )

    ax_auroc.set_ylim(0.5, 0.8)
    ax_auprc.set_ylim(0.0, 0.4)
    ax_f1.set_ylim(0.0, 0.4)

    ax_legend.axis("off")
    handles, labels = ax_auroc.get_legend_handles_labels()
    ax_legend.legend(
        handles,
        [LEGEND_NAME_MAP.get(label, label) for label in labels],
        loc="center",
        frameon=True,
        title="Method",
        title_fontsize=13,
        fontsize=12,
    )

    caption = (
        "Bars: mean over phecode labels within a prevalence bin. "
        "Error bars: bootstrap 95% CI of that mean, i.e. across-label variability, "
        "not model uncertainty."
    )
    fig.text(
        0.5,
        0.012,
        caption,
        ha="center",
        va="bottom",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout(rect=(0, 0.035, 1, 1))
    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    suffix = f"_{args.tag}" if args.tag else ""
    save_path = os.path.join(args.output_dir, f"combined_bar{suffix}")

    plt.savefig(f"{save_path}.png", dpi=300)
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.close()

    print(f"\n[Saved] {save_path}.png / .pdf")


def args_argument():
    p = argparse.ArgumentParser(description="Binned bar plots of AUROC/AUPRC/F1 across models")
    p.add_argument(
        "--models",
        default="binary,HAKE,ComplEx,ModE,RotatE,TransE",
        help="Comma-separated model names; 'binary' denotes the baseline",
    )
    p.add_argument(
        "--baseline_dir",
        default="disease_prediction/save/nn5_binary",
    )
    p.add_argument(
        "--ene_dir_prefix",
        default="disease_prediction/save/nn5_gpt-5-minimal_12r_",
        help="KGE directory prefix: {prefix}_{model}_ene/test_bin_{bin}",
    )
    p.add_argument("--train_label_path", default="disease_prediction/data/data_raw/labels_train.npy")
    p.add_argument("--output_dir", default="disease_prediction/plots")
    p.add_argument("--bin_choice", type=int, default=2)
    p.add_argument("--no_bar_labels", action="store_true", help="Hide numerical values above bars",)
    p.add_argument("--palette", default="vlag")
    p.add_argument("--tag", default="0906", help="Output filename suffix")
    return p.parse_args()


if __name__ == "__main__":
    main(args_argument())