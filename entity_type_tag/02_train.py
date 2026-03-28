import os
import argparse
import random
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


LABEL_Y_MAP: Dict[str, int] = {
    "ACTI": 0,
    "ANAT": 1,
    "BASE": 2,
    "CHEM": 3,
    "DISO": 4,
    "GENE": 5,
    "MEAS": 6,
    "MISC": 7,
    "PHYS": 8,
    "PROC": 9,
}
ID2LABEL = {v: k for k, v in LABEL_Y_MAP.items()}


def read_data_file(path: str) -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", maxsplit=1)
            if len(parts) != 2:
                # skip malformed lines
                continue
            lab, txt = parts
            if lab not in LABEL_Y_MAP:
                raise ValueError(f"Unknown label '{lab}' at line {line_no}")
            texts.append(txt.strip())
            labels.append(LABEL_Y_MAP[lab])
    if len(texts) == 0:
        raise RuntimeError(f"No valid samples found in {path}")
    return texts, labels


class EntityTypingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,  # dynamic padding handled by collator
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()} # input_ids, attention_mask
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_f1": f1_score(labels, preds, average="micro"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval"])

    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="model/biobert_v1.0_pubmed_pmc")
    parser.add_argument("--save_dir", type=str, default="entity_type_tag/save")
    parser.add_argument("--max_length", type=int, default=64)

    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--num_train_epochs", type=float, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["steps", "epoch"])
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["steps", "epoch"])
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=2)

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # -------- Load data --------
    if args.mode == "train":
        x_train, y_train = read_data_file(args.train_data_path)
        x_eval, y_eval = read_data_file(args.eval_data_path)
    else:
        x_eval, y_eval = read_data_file(args.test_data_path)

    # -------- Tokenizer --------
    tokenizer = AutoTokenizer.from_pretrained(args.save_dir if args.mode == "eval" else args.model_name, use_fast=True)

    eval_ds = EntityTypingDataset(x_eval, y_eval, tokenizer, args.max_length)
    train_ds = None
    if args.mode == "train":
        train_ds = EntityTypingDataset(x_train, y_train, tokenizer, args.max_length if args.mode == "train" else None)

    # -------- Model (FULL FINETUNE) --------
    config = AutoConfig.from_pretrained(
        args.save_dir if args.mode == "eval" else args.model_name,
        num_labels=10,
        id2label=ID2LABEL,
        label2id=LABEL_Y_MAP,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.save_dir if args.mode == "eval" else args.model_name,
        config=config,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -------- Trainer args --------
    if args.mode == "train":
        training_args = TrainingArguments(
            output_dir=args.save_dir,
            overwrite_output_dir=True,

            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,

            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,

            evaluation_strategy=args.eval_strategy,
            save_strategy=args.save_strategy,
            save_total_limit=args.save_total_limit,

            logging_steps=args.logging_steps,
            report_to="wandb",

            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,

            fp16=args.fp16,
            bf16=args.bf16,

            dataloader_num_workers=2,
            remove_unused_columns=False,
        )
    else:
        training_args = TrainingArguments(
            output_dir=os.path.join(args.save_dir, "eval"),
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            fp16=args.fp16,
            bf16=args.bf16,
            report_to="none",
        )

    callbacks = []
    if args.early_stopping and args.mode == "train":
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # -------- Train --------
    if args.mode == "train":
        trainer.train()
        trainer.save_model(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)

        metrics = trainer.evaluate()
        with open(os.path.join(args.save_dir, "train_metrics.txt"), "w", encoding="utf-8") as f:
            print("Final eval metrics:", metrics)

    # -------- Evaluation --------
    if args.mode == "eval":
        metrics = trainer.evaluate()
        preds = trainer.predict(eval_ds)
        logits = preds.predictions
        y_pred = np.argmax(logits, axis=-1)
        with open(os.path.join(args.save_dir, "eval/eval_metrics.txt"), "w", encoding="utf-8") as f:
            f.write("\n=== Overall metrics ===\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\n=== Per-class metrics ===\n")
            f.write(classification_report(y_eval, y_pred, target_names=[ID2LABEL[i] for i in range(10)], digits=4))
            f.write("\n=== Confusion matrix ===\n")
            cm = confusion_matrix(y_eval, y_pred)
            f.write("\n".join(" ".join(map(str, row)) for row in cm))


if __name__ == "__main__":
    main()
