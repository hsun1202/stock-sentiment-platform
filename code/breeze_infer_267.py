# -*- coding: utf-8 -*-
"""
breeze_infer_full.py 的267檔股票版本，邏輯完全一樣（含斷點續傳：每2000則
存一次 .partial.csv、整檔股票做完才寫最終 _sentiment.csv 並刪掉暫存檔、
中斷後重跑會自動接續進度、已完成的股票會直接跳過），只改了兩個地方：
  1. 股票清單改讀 stock_list_267.py 的 STOCK_LIST_267（267檔）
  2. aggregate_by_article 分組欄位從 "link" 改成 "article_id"
     （match_ptt_articles_267.py 輸出的comments.csv欄位是article_id，
     不是原本batch_crawler.py輸出的link欄位）

使用方式：
    python breeze_infer_267.py --batch-dir ptt_output_267 --out-dir sentiment_output_267

隨時可以Ctrl+C中斷，重跑同一個指令會自動從中斷點接續，不會重算已完成的部分。
"""

import os
import json
import time
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from stock_list_267 import STOCK_LIST_267

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

    grouped = df.groupby(["stock_code", "article_id", date_col], as_index=False).apply(
        compute_factor, include_groups=False
    )
    grouped = grouped.rename(columns={date_col: "resolved_date"})
    return grouped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用Breeze-7B對267檔股票的全部留言做情緒推論（含斷點續傳）")
    parser.add_argument("--batch-dir", type=str, default="ptt_output_267")
    parser.add_argument("--out-dir", type=str, default="sentiment_output_267")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=2000,
                         help="單一股票內每處理這麼多則就記錄一次進度、存一次中途進度檔")
    parser.add_argument("--only-codes", type=str, default=None,
                         help="只處理指定股票代碼（逗號分隔），用於跟其他環境分工")
    parser.add_argument("--start-date", type=str, default=None, help="只處理此日期(含)之後的留言，格式YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="只處理此日期(含)之前的留言，格式YYYY-MM-DD")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    STOCK_LIST = STOCK_LIST_267
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
        date_col = "resolved_date"
        if args.start_date or args.end_date:
            dates = pd.to_datetime(df[date_col])
            mask = pd.Series(True, index=df.index)
            if args.start_date:
                mask &= dates >= pd.Timestamp(args.start_date)
            if args.end_date:
                mask &= dates <= pd.Timestamp(args.end_date)
            df = df[mask].reset_index(drop=True)
        if len(df) == 0:
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}：篩選日期範圍後沒有留言，跳過")
            continue
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
    print("\n[完成] 267檔股票已用Breeze-7B重新推論情緒")
