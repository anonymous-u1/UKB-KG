import argparse
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, auc, roc_curve, precision_recall_curve, f1_score
)
from utils.data import Dataset
from disease_prediction.model import nn_5layer


def args_argument():
    parser = argparse.ArgumentParser(description="Multi-Disease Prediction Testing Script")
    parser.add_argument('--save_dir', required=True, help='Directory where model checkpoint is stored')
    parser.add_argument('--test_name', required=True, help='Directory name where metrics are stored')
    parser.add_argument('--data_type', type=str, choices=['binary', 'binary_ene', 'ene'], required=True, help="Type of input features.")
    parser.add_argument('--val_feature_path', type=str, required=True, help="Path to val feature .npy")
    parser.add_argument('--test_feature_path', type=str, required=True, help="Path to test feature .npy")
    parser.add_argument('--val_ene_path', type=str, default=None, help="Path to val ene feature .npy")
    parser.add_argument('--test_ene_path', type=str, default=None, help="Path to test ene feature .npy")
    parser.add_argument('--train_label_path', type=str, required=True, help="Path to train label .npy")
    parser.add_argument('--val_label_path', type=str, required=True, help="Path to val label .npy")
    parser.add_argument('--test_label_path', type=str, required=True, help="Path to test label .npy")
    parser.add_argument('-hs', '--hidden_size', type=int, default=150, help='Hidden layer size (default: 150)')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--debug', type=int, default=0, help='Debug mode (use small subset if set to 1)')
    return parser.parse_args()


# Load Data
def load_data(args):
    print("\n============================== Data Loading ==============================")
    print(f"→ Data type: {args.data_type}")

    if args.data_type == 'binary':
        val_feature = np.load(args.val_feature_path, allow_pickle=True)
        test_feature = np.load(args.test_feature_path, allow_pickle=True)
    elif args.data_type == 'ene':
        val_feature = np.load(args.val_ene_path, allow_pickle=True)
        test_feature = np.load(args.test_ene_path, allow_pickle=True)
    else:
        raise ValueError('Invalid data type')

    train_label = np.load(args.train_label_path)
    val_label = np.load(args.val_label_path)
    test_label = np.load(args.test_label_path)

    # debug mode
    if args.debug == 1:
        val_feature, test_feature = val_feature[:1000, :50], test_feature[:1000, :50]
        train_label, val_label, test_label = train_label[:1000, :50], val_label[:1000, :50], test_label[:1000, :50]
        print("[DEBUG MODE] Using small subset of data")

    print(f"Val Feature File: {args.val_feature_path}, Test Feature File: {args.test_feature_path}")
    print(f"Val ene File: {args.val_ene_path}, Test ene File: {args.test_ene_path}")
    print(f"Val Feature Shape: {val_feature.shape}, Test Feature Shape: {test_feature.shape}")
    print(f"Train Label Shape:   {train_label.shape}, Val Label Shape:   {val_label.shape}, Test Label shape:   {test_label.shape}")
    print("=============================================================================\n")

    return train_label, val_feature, val_label, test_feature, test_label


# threshold computation
def find_threshold(val_outputs, val_labels, mode=None, thresholds=None):
    val_scores = torch.sigmoid(val_outputs).detach().cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.detach().cpu().numpy()

    # Micro mode
    if mode == 'micro':
        y_flat = val_labels.flatten()
        s_flat = val_scores.flatten()
        return _compute_single_threshold(y_flat, s_flat)
    # Per-label / Macro mode
    elif mode == 'macro':
        for i in thresholds.keys():
            s = val_scores[:, i]
            thresholds[i] = np.median(s)

        return thresholds
    else:
        raise ValueError('Invalid mode')

def _compute_single_threshold(y, s):
    order = np.argsort(s)
    y_sorted = y[order]
    label_count = np.sum(y_sorted)
    TP = label_count - np.cumsum(y_sorted)
    FP = len(y) - label_count - np.cumsum(y_sorted == 0)
    FN = np.cumsum(y_sorted)
    f1 = (2 * TP) / (2 * TP + FP + FN)
    best_threshold = s[order[np.argmax(f1)]]
    return best_threshold


# Compute metrics
def MMAF(test_outputs, test_truth, threshold, threshold_micro, threshold_macro, output_dir):
    idx = list(threshold.keys())
    test_outputs_nonzero = test_outputs[:, idx]
    test_truth_nonzero = test_truth[:, idx]
    threshold_values = list(threshold.values())

    idx_top50 = list(threshold_macro.keys())
    test_outputs_top50 = test_outputs[:, idx_top50]
    test_truth_top50 = test_truth[:, idx_top50]
    threshold_values_top50 = list(threshold_macro.values())

    predicted_test = (torch.sigmoid(test_outputs_nonzero).data > torch.tensor(threshold_values)).float()
    predicted_test_micro = (torch.sigmoid(test_outputs_nonzero).data > threshold_micro).float()
    predicted_test_macro = (torch.sigmoid(test_outputs_top50).data > torch.tensor(threshold_values_top50)).float()

    micro_auc = roc_auc_score(test_truth_nonzero, test_outputs_nonzero, average='micro')
    macro_auc = roc_auc_score(test_truth_nonzero, test_outputs_nonzero, average='macro')
    macro_auc_top50 = roc_auc_score(test_truth_top50, test_outputs_top50, average='macro')
    micro_f1 = f1_score(test_truth_nonzero, predicted_test_micro, average='micro')
    macro_f1 = f1_score(test_truth_nonzero, predicted_test, average='macro')
    macro_f1_top50 = f1_score(test_truth_top50, predicted_test_macro, average='macro')

    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'a') as f:
        f.write(f"Micro AUC: {micro_auc:.4f}\nMacro AUC: {macro_auc:.4f}\nMacro AUC@50: {macro_auc_top50:.4f}\n")
        f.write(f"Micro F1: {micro_f1:.4f}\nMacro F1: {macro_f1:.4f}\nMacro F1@50: {macro_f1_top50:.4f}\n")
    print(f"✅ Metrics saved to {metrics_path}")


def sampling_test(test_outputs, test_truth, test_label, output_dir, threshold, train_col_sum, val_col_sum, test_col_sum, nonzero_col_idx, neg_to_pos_ratio=10):
    print("\n============================== Sampling-based Evaluation ==============================")
    print(f"Setting negative-to-positive ratio = {neg_to_pos_ratio}")

    precision_, recall_, fpr, tpr = {}, {}, {}, {}
    auprc, auroc, f1 = {}, {}, {}

    all_probs = {}
    all_truth = {}
    all_thresholds = np.array(threshold)

    # per-label evaluation
    for i in tqdm(range(test_label.shape[1]), desc="Per-label sampling evaluation"):
        y_test_subset = test_label[:, i]
        pos_idx = list(np.where(y_test_subset == 1)[0])
        neg_idx = list(np.where(y_test_subset == 0)[0])

        if i not in nonzero_col_idx:
            continue

        test_outputs_i, test_truth_i = [], []

        if len(neg_idx) / len(pos_idx) >= 10:
            rounds, pos_each = 50, 16
        else:
            rounds, pos_each = 200, 64

        # sample
        for _ in range(rounds):
            if len(neg_idx) / len(pos_idx) >= 10 and len(pos_idx) < 1000:
                selected_pos = random.choices(pos_idx, k=pos_each)
            else:
                selected_pos = random.sample(pos_idx, pos_each)
            selected_neg = random.sample(neg_idx, pos_each * neg_to_pos_ratio)
            selected = selected_pos + selected_neg
            test_outputs_i.append(test_outputs[selected])
            test_truth_i.append(test_truth[selected])

        test_outputs_ = torch.cat(test_outputs_i, dim=0)
        test_truth_ = torch.cat(test_truth_i, dim=0)

        # compute metrics
        probs = torch.sigmoid(test_outputs_).cpu().numpy()[:, i]
        truth = test_truth_.cpu().numpy()[:, i]
        preds = (probs > threshold[i]).astype(float)
        all_truth[i] = truth
        all_probs[i] = probs

        precision, recall, _ = precision_recall_curve(truth, probs)
        fpr_i, tpr_i, _ = roc_curve(truth, probs)

        precision_[i], recall_[i] = precision, recall
        fpr[i], tpr[i] = fpr_i, tpr_i
        auprc[i] = auc(recall, precision)
        auroc[i] = roc_auc_score(truth, probs)
        f1[i] = f1_score(truth, preds, average='binary')

    print(f"✅ Sampling evaluation done for {len(auroc)} labels.")

    # save
    np.save(os.path.join(output_dir, "truth.npy"), all_truth)
    np.save(os.path.join(output_dir, "probs.npy"), all_probs)
    np.save(os.path.join(output_dir, "thresholds.npy"), all_thresholds)

    train_col_sum = pd.Series(train_col_sum, name='train_col_sum')
    val_col_sum = pd.Series(val_col_sum, name='val_col_sum')
    test_col_sum = pd.Series(test_col_sum, name='test_col_sum')

    model_auc = pd.DataFrame.from_dict(auroc, orient="index")
    model_prc = pd.DataFrame.from_dict(auprc, orient="index")
    model_f1 = pd.DataFrame.from_dict(f1, orient="index")
    data_concat = pd.concat([train_col_sum, val_col_sum, test_col_sum, model_auc, model_prc, model_f1], axis=1, join="inner")

    data_concat.columns = ["train_col_sum", "val_col_sum", "test_col_sum", "auroc", "auprc", "f1"]
    data_concat.to_csv(os.path.join(output_dir, "metrics.csv"))


    # Segmented statistics
    bins = [(0, 10), (10, 100), (100, 500), (500, np.inf)]
    metrics_results = pd.DataFrame(0, index=range(len(bins)), columns=["bin_label", "auroc", "auprc", "f1", "count"])
    for j, (low, high) in enumerate(bins):
        if high == np.inf:
            bin_label = f"{low}+"
        else:
            bin_label = f"{low}-{high}"
        metrics_results.loc[j, 'bin_label'] = bin_label
        subset = data_concat[(data_concat['train_col_sum'] > low) & (data_concat['train_col_sum'] <= high)]
        metrics_results.loc[j, 'auroc'] = subset['auroc'].mean()
        metrics_results.loc[j, 'auprc'] = subset['auprc'].mean()
        metrics_results.loc[j, 'f1'] = subset['f1'].mean()
        metrics_results.loc[j, 'count'] = len(subset)

    metrics_results.to_csv(os.path.join(output_dir, "mean.csv"))

    np.save(os.path.join(output_dir, 'rareseeker_fpr.npy'), fpr)
    np.save(os.path.join(output_dir, 'rareseeker_tpr.npy'), tpr)
    np.save(os.path.join(output_dir, 'rareseeker_precision.npy'), precision_)
    np.save(os.path.join(output_dir, 'rareseeker_recall.npy'), recall_)
    np.save(os.path.join(output_dir, 'rareseeker_AUROC.npy'), auroc)
    np.save(os.path.join(output_dir, 'rareseeker_AUPRC.npy'), auprc)
    np.save(os.path.join(output_dir, 'rareseeker_f1.npy'), f1)

    print(f"✅ Sampling-based metrics saved to {output_dir}\n")


# main test
def test(train_label, val_feature, val_label, test_feature, test_label,
         model_checkpoint_loc, output_dir, use_cuda=True, hidden_size=150, batch_size=128):
    print("\n============================== Model Testing ==============================")

    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")

    # load model
    net = nn_5layer(test_feature.shape[1], test_label.shape[1], hidden_size)
    checkpoint = torch.load(model_checkpoint_loc, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    net.eval()

    print(f"Model loaded from {model_checkpoint_loc}")
    print(f"Evaluating on device: {device}")

    # load data
    testdata = Dataset(test_feature, test_label)
    valdata = Dataset(val_feature, val_label)
    test_loader = DataLoader(dataset=testdata, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=valdata, batch_size=batch_size, shuffle=False)

    print(f"Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    def infer(loader, infer_type):
        outputs, truths = [], []
        for inputs, labels in tqdm(loader, desc=f"{infer_type} Inferencing"):
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            with torch.no_grad():
                output = net(inputs)
            outputs.append(output.cpu())
            truths.append(labels.cpu())
        return torch.cat(outputs, dim=0), torch.cat(truths, dim=0)

    val_outputs, _ = infer(val_loader, "Val")
    test_outputs, test_truth = infer(test_loader, "Test")
    print("✅ Inference completed.")

    # compute threshold
    train_col_sum = np.sum(train_label, axis=0)
    val_col_sum = np.sum(val_label, axis=0)
    test_col_sum = np.sum(test_label, axis=0)
    nonzero_col_idx = np.intersect1d(np.where(train_col_sum != 0)[0], np.where(test_col_sum != 0)[0])

    threshold = {i: None for i in nonzero_col_idx}
    threshold = find_threshold(val_outputs, val_label, 'macro', threshold)
    threshold_micro = find_threshold(val_outputs[:, nonzero_col_idx], val_label[:, nonzero_col_idx], 'micro', None)
    top_50_cols = np.argsort(train_col_sum)[-50:]
    threshold_macro = {i: threshold[i] for i in top_50_cols}

    print("✅ Thresholds computed. Calculating final metrics...")
    MMAF(test_outputs, test_truth, threshold, threshold_micro, threshold_macro, output_dir)
    sampling_test(test_outputs, test_truth, test_label, output_dir, threshold, train_col_sum, val_col_sum, test_col_sum, nonzero_col_idx, neg_to_pos_ratio=10)


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s" if h > 0 else (f"{m}m {s}s" if m > 0 else f"{s}s")


def main(args):
    save_dir = args.save_dir
    model_checkpoint_loc = os.path.join(save_dir, "best_classifier.pth.tar")
    if args.debug == 1:
        output_dir = os.path.join(save_dir, "debug")
    else:
        output_dir = os.path.join(save_dir, args.test_name)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    train_label, val_feature, val_label, test_feature, test_label = load_data(args)
    test(train_label, val_feature, val_label, test_feature, test_label,
         model_checkpoint_loc, output_dir, use_cuda=True, hidden_size=args.hidden_size, batch_size=args.batch_size)


if __name__ == "__main__":
    start_time = time.time()
    args = args_argument()
    print(args)
    main(args)
    total_time = time.time() - start_time
    print(f"\nTotal Time Used: {format_time(total_time)}")