# -*- coding: utf-8 -*-
"""
用Breeze-7B-Instruct對留言做情緒分類（強化版prompt：黑話/成語說明 + few-shot鎖輸出格式）
============================================================

背景：
    第一版prompt測試時，Breeze雖然讀對了轉折語意（詞典法/BERT都
    失敗的case），但兩個地方不理想：
        1. 不認得PTT黑話「歐印」、看不懂反諷成語「利多出盡」
        2. 沒有照指示輸出英文單字，而是回中文同義詞（看空/看多/中立）
    這版加入黑話對照表跟few-shot範例，強迫輸出格式一致，
    並保留穩健的中文詞映射當備援（就算LLM沒完全照格式回答也能解析）。

使用方式：
    python breeze_infer.py --input annotation_result.csv --out breeze_preds.csv
"""

import os
import re
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd

MODEL = "MediaTek-Research/Breeze-7B-Instruct-v1_0"

SYSTEM = """你是判斷台股PTT股票板留言情緒的專家。判斷留言對「該檔股票股價方向」的傾向：positive(看多，認為會漲)、negative(看空，認為會跌)、neutral(中立，無明確方向/純聊天/看不懂)。

PTT股版黑話對照（務必套用）：
- 看多黑話：歐印、梭哈、all in、買爆、漲停、噴出、起飛、抄底、財富自由 -> positive
- 看空黑話：塊陶、畢業、丸子、跌停、斷頭、咕嚕、套牢、住套房 -> negative
- 反諷成語（字面跟真實語意相反，要特別注意）：「利多出盡」表面是利多但實際是看空訊號 -> negative；「利空出盡」表面是利空但實際是看多訊號 -> positive
- 「出貨」在股版通常指倒貨看空，但如果上下文是講貨櫃/物流/實體貨物出貨，跟股價無關，判neutral

只能回答以下三個英文單字之一，不要有其他文字、不要標點、不要中文：positive 或 negative 或 neutral"""

FEWSHOT = [
    ("歐印", "positive"),
    ("利多出盡", "negative"),
    ("塊陶 完了", "negative"),
    ("今天天氣不錯", "neutral"),
]

LABEL_MAP_FALLBACK = {
    "positive": "positive", "看多": "positive", "看好": "positive", "正": "positive", "正面": "positive",
    "negative": "negative", "看空": "negative", "看壞": "negative", "負": "negative", "負面": "negative",
    "neutral": "neutral", "中立": "neutral", "中性": "neutral",
}


def parse_label(text):
    text = text.strip().lower()
    for key, val in LABEL_MAP_FALLBACK.items():
        if key.lower() in text:
            return val
    return "neutral"  # 解析不出來時，安全預設為中立


@torch.no_grad()
def classify(text, tok, model):
    messages = [{"role": "system", "content": SYSTEM}]
    for ex_text, ex_label in FEWSHOT:
        messages.append({"role": "user", "content": ex_text})
        messages.append({"role": "assistant", "content": ex_label})
    messages.append({"role": "user", "content": text})

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_label(reply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用Breeze-7B對留言做情緒分類")
    parser.add_argument("--input", type=str, required=True, help="要分類的CSV，需有content欄位")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map="auto")
    model.eval()

    df = pd.read_csv(args.input)
    preds = []
    for i, text in enumerate(df["content"].astype(str).tolist(), 1):
        preds.append(classify(text, tok, model))
        if i % 20 == 0:
            print(f"{i}/{len(df)}")
    df["breeze_label"] = preds
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[完成] 已輸出至 {args.out}")
