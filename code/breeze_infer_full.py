# -*- coding: utf-8 -*-
"""
用Breeze-7B對30檔股票全部留言做情緒分類（批次生成版，含斷點續傳）
============================================================

目的：
    跟infer_bert_sentiment.py架構一樣，但情緒判斷模型換成
    Breeze-7B-Instruct（本機4-bit量化跑，免費、實測macro-F1
    贏過詞典法跟BERT）。輸出格式跟merge_datasets.py相容，可以
    直接接後續資料整合流程。

    用batch生成（不是像test腳本那樣一則一則跑）大幅提升吞吐量。
    每檔股票處理完就存檔，中斷後重跑會跳過已完成的股票
    （斷點續傳，比照batch_crawler.py的設計）。

使用方式：
    python breeze_infer_full.py --out-dir sentiment_output_breeze
"""

import os
import json
import time
import argparse
import importlib.util
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

PROGRESS_FILE = "breeze_progress.json"


def write_progress(out_dir, stock_idx, total_stocks, code, name, done, total, started_at):
    payload = {
        "stock_idx": stock_idx, "total_stocks": total_stocks,
        "stock_code": code, "stock_name": name,
        "comments_done": done, "comments_total": total,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_at": started_at,
    }
    path = os.path.join(out_dir, PROGRESS_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # 原子寫入，避免讀到寫一半的檔案

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
    return "neutral"


def build_prompt(tok, text):
    messages = [{"role": "system", "content": SYSTEM}]
    for ex_text, ex_label in FEWSHOT:
        messages.append({"role": "user", "content": ex_text})
        messages.append({"role": "assistant", "content": ex_label})
    messages.append({"role": "user", "content": text[:200]})  # 截斷過長留言，控制prompt長度
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def classify_batch(texts, tok, model, batch_size=16, max_new_tokens=6):
    all_labels = []
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        prompts = [build_prompt(tok, t) for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        for j in range(len(chunk)):
            reply = tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            all_labels.append(parse_label(reply))
    return all_labels


def load_stock_list():
    spec = importlib.util.spec_from_file_location(
        "stock_volume_scanner",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_volume_scanner.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.STOCK_LIST


def aggregate_by_article(df, date_col):
    def compute_factor(group):
        pos = (group["text_sentiment"] == "positive").sum()
        neg = (group["text_sentiment"] == "negative").sum()
        total = len(group)
        factor = (pos - neg) / total if total > 0 else 0.0
        return pd.Series({
            "total_comments": total,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": total - pos - neg,
            "text_sentiment_factor": round(factor, 4),
        })

    grouped = df.groupby(["stock_code", "link", date_col], as_index=False).apply(
        compute_factor, include_groups=False
    )
    grouped = grouped.rename(columns={date_col: "resolved_date"})
    return grouped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用Breeze-7B對全部留言做情緒推論")
    parser.add_argument("--batch-dir", type=str, default="batch_output")
    parser.add_argument("--out-dir", type=str, default="sentiment_output_breeze")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=2000,
                         help="單一股票內每處理這麼多則就記錄一次進度、存一次中途進度檔")
    parser.add_argument("--only-codes", type=str, default=None,
                         help="只處理指定股票代碼（逗號分隔），用於跟其他環境（如Colab）分工")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    STOCK_LIST = load_stock_list()
    if args.only_codes:
        wanted = set(args.only_codes.split(","))
        STOCK_LIST = [s for s in STOCK_LIST if s[0] in wanted]
        print(f"=== 只處理指定股票：{[s[0] for s in STOCK_LIST]} ===")

    print("=== 載入 Breeze-7B（4-bit量化）===")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb_config, device_map="auto")
    model.eval()
    print("=== 模型載入完成，開始批次處理 ===\n")

    run_started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    for i, (code, name, industry) in enumerate(STOCK_LIST, 1):
        comments_path = os.path.join(args.batch_dir, f"{code}_{name}_comments.csv")
        out_path = os.path.join(args.out_dir, f"{code}_{name}_sentiment.csv")
        partial_path = os.path.join(args.out_dir, f"{code}_{name}.partial.csv")

        if os.path.exists(out_path):
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}（{industry}）已存在，跳過")
            continue
        if not os.path.exists(comments_path):
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}：找不到留言檔案，跳過")
            continue

        df = pd.read_csv(comments_path)
        date_col = "resolved_date" if "resolved_date" in df.columns else "board_date"
        texts = df["content"].fillna("").astype(str).tolist()

        start_idx = 0
        labels = []
        if os.path.exists(partial_path):
            partial = pd.read_csv(partial_path)
            labels = partial["text_sentiment"].tolist()
            start_idx = len(labels)
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}：接續上次進度，已完成 {start_idx}/{len(texts)}")
        else:
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}（{industry}）：{len(texts)} 則留言，開始分類...")

        while start_idx < len(texts):
            end_idx = min(start_idx + args.chunk_size, len(texts))
            chunk_labels = classify_batch(texts[start_idx:end_idx], tok, model, batch_size=args.batch_size)
            labels.extend(chunk_labels)
            pd.DataFrame({"text_sentiment": labels}).to_csv(partial_path, index=False)
            start_idx = end_idx
            write_progress(args.out_dir, i, len(STOCK_LIST), code, name, start_idx, len(texts), run_started_at)
            print(f"    進度 {start_idx}/{len(texts)}")

        df["text_sentiment"] = labels
        article_summary = aggregate_by_article(df, date_col)
        article_summary.to_csv(out_path, index=False, encoding="utf-8-sig")
        os.remove(partial_path)
        print(f"    [完成] {len(article_summary)} 篇文章的情緒因子 -> {out_path}")

    progress_path = os.path.join(args.out_dir, PROGRESS_FILE)
    if os.path.exists(progress_path):
        os.remove(progress_path)
    print("\n[完成] 全部股票已用Breeze-7B重新推論情緒")
