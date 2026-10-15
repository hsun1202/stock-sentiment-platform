# -*- coding: utf-8 -*-
"""
用跟Breeze-7B一樣的prompt設計（同樣的system prompt、few-shot範例），
在Taiwan-LLM上跑一次392筆人工標籤，直接比較Accuracy/Macro-F1，
不用fine-tune、單純看兩個模型zero-shot的表現差異。
"""
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score

from breeze_infer_267 import SYSTEM, FEWSHOT, classify_batch

MODEL = "yentinglin/Taiwan-LLM-7B-v2.1-chat"

print("=== 載入 Taiwan-LLM（4-bit量化）===", flush=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map="auto")
model.eval()
print("=== 模型載入完成 ===", flush=True)

df = pd.read_csv("all_human_labels.csv")
print(f"測試筆數: {len(df)}", flush=True)

texts = df["content"].fillna("").astype(str).tolist()
labels_pred = classify_batch(texts, tok, model, batch_size=4)

df["predicted"] = labels_pred
df.to_csv("taiwan_llm_predictions.csv", index=False, encoding="utf-8-sig")

acc = accuracy_score(df["human_label"], df["predicted"])
f1 = f1_score(df["human_label"], df["predicted"], average="macro", zero_division=0)

print(f"\n=== Taiwan-LLM 結果 ===", flush=True)
print(f"Accuracy: {acc*100:.2f}%", flush=True)
print(f"Macro-F1: {f1:.4f}", flush=True)
print(f"\n=== 對照 Breeze-7B（已知結果）===", flush=True)
print(f"Accuracy: 68.11%", flush=True)
print(f"Macro-F1: 0.5549", flush=True)

print("\n預測分布:", flush=True)
print(df["predicted"].value_counts(), flush=True)
print("\n真實分布:", flush=True)
print(df["human_label"].value_counts(), flush=True)
