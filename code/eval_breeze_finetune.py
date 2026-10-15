# -*- coding: utf-8 -*-
"""
比較Breeze-7B微調前後，在held-out val集上的準確率/Macro-F1
============================================================
val.csv是完全沒被拿去訓練的60筆人工標籤，用同一批資料、同一套
prompt格式，分別跑「原版zero-shot」跟「LoRA微調後」，才能公平
比較微調到底有沒有用。

使用方式：
    python eval_breeze_finetune.py
"""

import os
import argparse
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "breeze_infer_267",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "breeze_infer_267.py")
)
_bi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bi)
MODEL = _bi.MODEL
build_prompt = _bi.build_prompt
classify_batch = _bi.classify_batch


def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=["positive", "negative", "neutral"])
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc*100:.2f}%   Macro-F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=["positive", "negative", "neutral"], digits=4))
    return acc, macro_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-csv", type=str, default="breeze_finetune_data/val.csv")
    parser.add_argument("--adapter-path", type=str, default="breeze_lora_adapter/final")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    val_df = pd.read_csv(args.val_csv)
    texts = val_df["content"].fillna("").astype(str).tolist()
    y_true = val_df["human_label"].tolist()
    print(f"Val集共 {len(texts)} 筆")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL)

    print("=== 載入原版Breeze-7B（zero-shot基準）===")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map="auto")
    base_model.eval()
    y_pred_zeroshot = classify_batch(texts, tok, base_model, batch_size=args.batch_size)
    report("Zero-shot（微調前）", y_true, y_pred_zeroshot)

    print("\n=== 套上LoRA adapter（微調後）===")
    ft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    ft_model.eval()
    y_pred_finetuned = classify_batch(texts, tok, ft_model, batch_size=args.batch_size)
    report("LoRA微調後", y_true, y_pred_finetuned)

    out_df = val_df.copy()
    out_df["pred_zeroshot"] = y_pred_zeroshot
    out_df["pred_finetuned"] = y_pred_finetuned
    out_path = "breeze_finetune_eval_detail.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n逐筆結果已存到 {out_path}")
