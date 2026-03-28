import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from disease_prediction.model import nn_5layer
from utils.data import Dataset
from utils.model_saving import ModelSaving
import random


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(1234)


def args_argument():
    parser = argparse.ArgumentParser(description="Multi-Disease Prediction Training Script")
    parser.add_argument('-s', '--save_dir', required=True, help='Output directory')
    parser.add_argument('--data_type', type=str, choices=['binary', 'binary_ene', 'ene'], required=True, help="Type of input features.")
    parser.add_argument('--train_feature_path', type=str, required=True, help="Path to train feature .npy")
    parser.add_argument('--val_feature_path', type=str, required=True, help="Path to val feature .npy")
    parser.add_argument('--train_ene_path', type=str, default=None, help="Path to train ene feature .npy")
    parser.add_argument('--val_ene_path', type=str, default=None, help="Path to val ene feature .npy")
    parser.add_argument('--train_label_path', type=str, required=True, help="Path to train label .npy")
    parser.add_argument('--val_label_path', type=str, required=True, help="Path to val label .npy")
    parser.add_argument('-hs', '--hidden_size', type=int, default=150, help='Hidden layer size (default: 150)')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Learning rate (default: 1e-4)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='Weight decay (default: 0)')
    parser.add_argument('--debug', type=int, default=0, help='Debug mode (use small dataset if set to 1)')
    return parser.parse_args()


# Load data
def load_data(args):
    print("\n============================== Data Loading ==============================")
    print(f"→ Data type: {args.data_type}")

    if args.data_type == 'binary':
        train_feature = np.load(args.train_feature_path, allow_pickle=True)
        val_feature = np.load(args.val_feature_path, allow_pickle=True)
    elif args.data_type == 'ene':
        train_feature = np.load(args.train_ene_path, allow_pickle=True)
        val_feature = np.load(args.val_ene_path, allow_pickle=True)
    else:
        raise ValueError('Invalid data type')

    train_label = np.load(args.train_label_path)
    val_label = np.load(args.val_label_path)

    # debug mode
    if args.debug == 1:
        train_feature, train_label = train_feature[:1000, :50], train_label[:1000, :50]
        val_feature, val_label = val_feature[:1000, :50], val_label[:1000, :50]
        print("[DEBUG MODE] Using small subset of data")

    print(f"Train Feature File: {args.train_feature_path}, Val Feature File: {args.val_feature_path}")
    print(f"Train ene File: {args.train_ene_path}, Val ene File: {args.val_ene_path}")
    print(f"Train Feature Shape: {train_feature.shape}, Val Feature Shape: {val_feature.shape}")
    print(f"Train Label Shape:   {train_label.shape}, Val Label shape:   {val_label.shape}")
    print("=============================================================================\n")

    return train_feature, train_label, val_feature, val_label

# Train
def train(train_feature, train_label, val_feature, val_label,
          use_cuda=True, hidden_size=150, batch_size=128,
          learning_rate=1e-4, weight_decay=0.0, save_dir=''):

    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net = nn_5layer(train_feature.shape[1], train_label.shape[1], hidden_size)
    net.initialize()
    net = net.to(device)
    print(f"Model: {net.__class__.__name__}")

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_data = Dataset(train_feature, train_label)
    val_data = Dataset(val_feature, val_label)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    n_epochs = 200
    early_stop = ModelSaving(patience=5, verbose=True)
    val_best = np.inf
    train_losses_epoch, val_losses_epoch = [], []

    os.makedirs(save_dir, exist_ok=True)
    print("\n============================== Training Start ==============================")

    # Start training
    for epoch in range(n_epochs):
        net.train()
        batch_train_losses = []

        for batch_idx, (train_inputs, train_labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        ):
            train_inputs = train_inputs.float().to(device)
            train_labels = train_labels.float().to(device)

            optimizer.zero_grad()
            outputs = net(train_inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.item())

        avg_train_loss = np.mean(batch_train_losses)
        print(f"[Epoch {epoch+1:03d}] 🔹 Train Loss: {avg_train_loss:.4f}")

        # Eval
        net.eval()
        batch_val_losses = []
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.float().to(device)
                val_labels = val_labels.float().to(device)
                val_outputs = net(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                batch_val_losses.append(val_loss.item())

        avg_val_loss = np.mean(batch_val_losses)
        print(f"[Epoch {epoch+1:03d}] 🔸 Val Loss:   {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < val_best:
            val_best = avg_val_loss
            best_model_path = os.path.join(save_dir, 'best_classifier.pth.tar')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, best_model_path)
            print(f"✅ [Best Model Saved] Epoch {epoch + 1:03d} | Val Loss ↓ {val_best:.4f}")

        # Early stopping
        early_stop(avg_val_loss)
        if early_stop.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        train_losses_epoch.append(avg_train_loss)
        val_losses_epoch.append(avg_val_loss)

    print("\n============================== ✅ Training Finished ==============================")
    print(f"Best Validation Loss: {val_best:.4f}")


    plt.figure()
    plt.plot(train_losses_epoch, '-o', label='Train')
    plt.plot(val_losses_epoch, '-x', label='Validation')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(save_dir, 'train_loss.png'))
    plt.close()

    print(f"Loss curve saved to {os.path.join(save_dir, 'train_loss.png')}")


def main(args):
    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created save directory: {save_dir}")

    hidden_size = args.hidden_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay

    print("\n============================== Training Configuration ==============================")
    print(f"Hidden Size: {hidden_size} | Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate} | Weight Decay: {weight_decay}")
    print("=============================================================================\n")

    train_feature, train_label, val_feature, val_label = load_data(args)
    train(train_feature, train_label, val_feature, val_label,
          use_cuda=True, hidden_size=hidden_size, batch_size=batch_size,
          learning_rate=learning_rate, weight_decay=weight_decay,
          save_dir=save_dir)


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s" if h > 0 else (f"{m}m {s}s" if m > 0 else f"{s}s")


if __name__ == "__main__":
    start_time = time.time()
    args = args_argument()
    print(args)
    main(args)
    total_time = time.time() - start_time
    print(f"\nTotal Time Used: {format_time(total_time)}")