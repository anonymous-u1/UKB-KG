import argparse
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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
    """Dataset for multi-label disease prediction."""

    def __init__(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Feature and label sample counts do not match: "
                f"x={x.shape[0]}, y={y.shape[0]}"
            )

        self.feature_size = x.shape[1]
        self.label_size = y.shape[1]
        self.x_data = torch.as_tensor(x, dtype=torch.float32)
        self.y_data = torch.as_tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


class EarlyStopping:
    """Stop training if validation loss does not improve for consecutive epochs."""

    def __init__(self, patience=5, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        # Non-finite validation loss indicates numerical instability.
        if not np.isfinite(val_loss):
            raise ValueError(
                f"Validation loss is not finite: {val_loss}. "
                f"Training may have encountered numerical instability."
            )

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

            if self.verbose:
                print(
                    f"Validation loss did not improve: "
                    f"{self.counter}/{self.patience}"
                )

            if self.counter >= self.patience:
                self.early_stop = True


def args_argument():
    parser = argparse.ArgumentParser(
        description="Multi-Disease Prediction Training Script"
    )

    parser.add_argument(
        "-s", "--save_dir", required=True, help="Output directory"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["binary", "ene"],
        required=True,
        help="Input feature type: binary or ene"
    )

    parser.add_argument(
        "--train_feature_path",
        type=str,
        default=None,
        help="Path to train binary feature .npy"
    )
    parser.add_argument(
        "--val_feature_path",
        type=str,
        default=None,
        help="Path to validation binary feature .npy"
    )
    parser.add_argument(
        "--train_ene_path",
        type=str,
        default=None,
        help="Path to train KGE embedding feature .npy"
    )
    parser.add_argument(
        "--val_ene_path",
        type=str,
        default=None,
        help="Path to validation KGE embedding feature .npy"
    )
    parser.add_argument(
        "--train_label_path",
        type=str,
        required=True,
        help="Path to train label .npy"
    )
    parser.add_argument(
        "--val_label_path",
        type=str,
        required=True,
        help="Path to validation label .npy"
    )

    parser.add_argument(
        "-hs", "--hidden_size", type=int, default=150,
        help="Hidden layer size (default: 150)"
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=128,
        help="Batch size (default: 128)"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "-wd", "--weight_decay", type=float, default=0.0,
        help="Weight decay (default: 0)"
    )
    parser.add_argument(
        "--debug", type=int, default=0, help="Debug mode"
    )

    return parser.parse_args()


def check_finite(name, array, chunk_size=5000):
    """Check whether a NumPy array contains NaN, +Inf, or -Inf."""

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


def validate_data(train_feature, train_label, val_feature, val_label):
    """
    Check dimensions, sample counts, feature/label consistency, and finite values.
    """

    arrays = {
        "train_feature": train_feature,
        "train_label": train_label,
        "val_feature": val_feature,
        "val_label": val_label
    }

    for name, array in arrays.items():
        if array.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D array, but got shape {array.shape}"
            )

    if train_feature.shape[0] != train_label.shape[0]:
        raise ValueError(
            "Train feature/label sample count mismatch: "
            f"{train_feature.shape[0]} vs {train_label.shape[0]}"
        )

    if val_feature.shape[0] != val_label.shape[0]:
        raise ValueError(
            "Validation feature/label sample count mismatch: "
            f"{val_feature.shape[0]} vs {val_label.shape[0]}"
        )

    if train_feature.shape[1] != val_feature.shape[1]:
        raise ValueError(
            "Train/validation feature dimension mismatch: "
            f"{train_feature.shape[1]} vs {val_feature.shape[1]}"
        )

    if train_label.shape[1] != val_label.shape[1]:
        raise ValueError(
            "Train/validation label dimension mismatch: "
            f"{train_label.shape[1]} vs {val_label.shape[1]}"
        )

    print("[INFO] Shape consistency check passed.")

    for name, array in arrays.items():
        check_finite(name, array)


def load_data(args):
    print("\n============================== Data Loading ==============================")
    print(f"→ Data type: {args.data_type}")

    if args.data_type == "binary":
        if args.train_feature_path is None or args.val_feature_path is None:
            raise ValueError(
                "For data_type='binary', --train_feature_path and "
                "--val_feature_path are required."
            )

        train_feature = np.load(args.train_feature_path)
        val_feature = np.load(args.val_feature_path)

        feature_train_path = args.train_feature_path
        feature_val_path = args.val_feature_path

    elif args.data_type == "ene":
        if args.train_ene_path is None or args.val_ene_path is None:
            raise ValueError(
                "For data_type='ene', --train_ene_path and "
                "--val_ene_path are required."
            )

        train_feature = np.load(args.train_ene_path)
        val_feature = np.load(args.val_ene_path)

        feature_train_path = args.train_ene_path
        feature_val_path = args.val_ene_path

    else:
        raise ValueError(f"Invalid data type: {args.data_type}")

    train_label = np.load(args.train_label_path)
    val_label = np.load(args.val_label_path)

    validate_data(train_feature, train_label, val_feature, val_label)

    if args.debug == 1:
        train_feature = train_feature[:1000, :50]
        train_label = train_label[:1000, :50]

        val_feature = val_feature[:1000, :50]
        val_label = val_label[:1000, :50]

        print("[DEBUG MODE] Using small subset of data.")

    print(f"Train Feature File: {feature_train_path}")
    print(f"Val Feature File: {feature_val_path}")
    print(f"Train Label File: {args.train_label_path}")
    print(f"Val Label File: {args.val_label_path}")
    print(f"Train Feature Shape: {train_feature.shape}")
    print(f"Val Feature Shape: {val_feature.shape}")
    print(f"Train Label Shape: {train_label.shape}")
    print(f"Val Label Shape: {val_label.shape}")
    print("========================================\n=====================================")

    return train_feature, train_label, val_feature, val_label


def train(
    train_feature,
    train_label,
    val_feature,
    val_label,
    use_cuda=True,
    hidden_size=150,
    batch_size=128,
    learning_rate=1e-4,
    weight_decay=0.0,
    save_dir=""
):
    device = torch.device(
        "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    net = NN5Layer(
        feature_num=train_feature.shape[1],
        label_num=train_label.shape[1],
        hidden_size=hidden_size
    )

    net.initialize()
    net = net.to(device)

    print(f"Model: {net.__class__.__name__}")

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    criterion = (torch.nn.BCEWithLogitsLoss())

    train_data = Dataset(train_feature, train_label)
    val_data = Dataset(val_feature, val_label)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    n_epochs = 200
    early_stopping = EarlyStopping(patience=5, verbose=True)

    val_best = np.inf
    train_losses_epoch = []
    val_losses_epoch = []

    os.makedirs(save_dir, exist_ok=True)

    print(
        "\n============================== "
        "Training Start =============================="
    )

    for epoch in range(n_epochs):
        net.train()

        train_loss_sum = 0.0
        train_sample_count = 0

        for train_inputs, train_labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{n_epochs}"
        ):
            train_inputs = train_inputs.to(device)
            train_labels = train_labels.to(device)

            optimizer.zero_grad()
            outputs = net(train_inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            # Accumulate a sample-weighted epoch loss.
            current_batch_size = train_inputs.size(0)
            train_loss_sum += loss.item() * current_batch_size
            train_sample_count += current_batch_size

        avg_train_loss = train_loss_sum / train_sample_count

        print(
            f"[Epoch {epoch + 1:03d}] 🔹 "
            f"Train Loss: {avg_train_loss:.4f}"
        )

        net.eval()

        val_loss_sum = 0.0
        val_sample_count = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                val_outputs = net(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                current_batch_size = val_inputs.size(0)
                val_loss_sum += val_loss.item() * current_batch_size
                val_sample_count += current_batch_size

        avg_val_loss = val_loss_sum / val_sample_count

        print(
            f"[Epoch {epoch + 1:03d}] 🔸 "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        train_losses_epoch.append(avg_train_loss)
        val_losses_epoch.append(avg_val_loss)

        early_stopping(avg_val_loss)

        if avg_val_loss < val_best:
            val_best = avg_val_loss

            best_model_path = os.path.join(
                save_dir,
                "best_classifier.pth.tar"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss
                },
                best_model_path
            )

            print(
                f"✅ [Best Model Saved] Epoch {epoch + 1:03d} | "
                f"Val Loss ↓ {val_best:.4f}"
            )

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(
        "\n============================== "
        "✅ Training Finished =============================="
    )
    print(f"Best Validation Loss: {val_best:.4f}")

    plt.figure()
    plt.plot(train_losses_epoch, "-o", label="Train")
    plt.plot(val_losses_epoch, "-x", label="Validation")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Training and Validation Loss")

    loss_curve_path = os.path.join(save_dir, "train_loss.png")
    plt.savefig(loss_curve_path)
    plt.close()

    print(f"Loss curve saved to {loss_curve_path}")


def main(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    print(
        "\n============================== "
        "Training Configuration =============================="
    )
    print(f"Data Type: {args.data_type}")
    print(f"Hidden Size: {args.hidden_size} | Batch Size: {args.batch_size}")
    print(
        f"Learning Rate: {args.learning_rate} | "
        f"Weight Decay: {args.weight_decay}"
    )
    print("========================================\n=====================================")

    train_feature, train_label, val_feature, val_label = load_data(args)

    train(
        train_feature,
        train_label,
        val_feature,
        val_label,
        use_cuda=True,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_dir=save_dir
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


if __name__ == "__main__":
    start_time = time.time()

    args = args_argument()
    print(args)

    main(args)

    total_time = time.time() - start_time
    print(f"\nTotal Time Used: {format_time(total_time)}")