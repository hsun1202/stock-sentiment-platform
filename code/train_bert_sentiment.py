# -*- coding: utf-8 -*-
"""
BERT情緒分類模型微調
============================================================

目的：
    用 prepare_bert_dataset.py 產出的弱監督資料（推/噓當標籤），
    微調繁體中文預訓練BERT (ckiplab/bert-base-chinese)，
    取代目前的權重詞典法，處理否定詞、上下文語意這類詞典法
    處理不好的情況。

使用方式：
    python train_bert_sentiment.py --epochs 2 --batch-size 32
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    n_classes = logits.shape[-1]
    average = "binary" if n_classes == 2 else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=average)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="微調BERT做PTT留言情緒分類")
    parser.add_argument("--data-dir", type=str, default="bert_dataset")
    parser.add_argument("--model-name", type=str, default="ckiplab/bert-base-chinese")
    parser.add_argument("--out-dir", type=str, default="bert_sentiment_model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num-labels", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 使用裝置：{device} ===")
    if device == "cuda":
        print(torch.cuda.get_device_name(0))

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    print(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    train_ds = CommentDataset(train_df["text"].astype(str).tolist(), train_df["label"].tolist(), tokenizer, args.max_length)
    val_ds = CommentDataset(val_df["text"].astype(str).tolist(), val_df["label"].tolist(), tokenizer, args.max_length)
    test_ds = CommentDataset(test_df["text"].astype(str).tolist(), test_df["label"].tolist(), tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=200,
        fp16=(device == "cuda"),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    print("\n=== 測試集評估 ===")
    test_metrics = trainer.evaluate(test_ds)
    print(test_metrics)

    final_path = os.path.join(args.out_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n[完成] 模型已存到 {final_path}")
