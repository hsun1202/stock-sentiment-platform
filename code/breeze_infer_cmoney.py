# -*- coding: utf-8 -*-
"""
用Breeze-7B對267檔股票的CMoney文章+留言做情緒推論（比照breeze_infer_267.py）
============================================================

跟PTT版本(breeze_infer_267.py)的差異：
    CMoney的資料是「文章」+「文章底下的留言」兩層，不是PTT那種
    單純的推文列表。這裡把每篇文章自己的內文，當作跟它底下的留言
    平起平坐的一則「待分類文字」（文章本身也代表一種情緒表態），
    全部一起丟給Breeze-7B分類，再依「同一篇文章」聚合成
    positive/negative/neutral計數。

    為了跟PTT版本方法論一致、方便日後直接比較兩個資料源的結論，
    這裡沿用「一篇文章的所有留言（不分留言實際發布日）都算在文章
    發文當天」的做法，跟breeze_infer_267.py處理PTT留言的方式相同
    （PTT留言本來也沒有獨立於文章之外的日期概念）。

    情緒分類用的system prompt、few-shot範例、模型都跟PTT版本完全
    一致，才能公平比較兩個資料源的情緒因子。

輸出格式：
    {代碼}_{名稱}_sentiment.csv，欄位跟breeze_infer_267.py輸出的
    完全一樣（stock_code, article_id, resolved_date, total_comments,
    positive_count, negative_count, neutral_count, text_sentiment_factor），
    可以直接沿用merge_datasets_267.py的邏輯合併進技術指標。

使用方式：
    python breeze_infer_cmoney.py --start-date 2020-01-01 --end-date 2024-12-31

隨時可以Ctrl+C中斷，重跑同一個指令會自動從中斷點接續，不會重算已完成的部分。
用check_breeze_progress.py --out-dir sentiment_output_cmoney 查進度（共用同一套進度檔格式）。
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
    os.replace(tmp, path)


MODEL = "MediaTek-Research/Breeze-7B-Instruct-v1_0"

SYSTEM = """你是判斷台股PTT股票板留言情緒的專家。判斷留言對「該檔股票股價方向」的傾向：positive(看多，認為會漲)、negative(看空，認為會跌)、neutral(中立，無明確方向/純聊天/看不懂)。

PTT股版黑話對照（務必套用）：
- 看多黑話：歐印、梭哈、all in、買爆、漲停、噴出、起飛、抄底、財富自由 -> positive
- 看空黑話：塊陶、畢業、丸子、跌停、斷頭、咕嚕、套牢、住套房 -> negative
- 股市用語規則（字面跟真實語意相反，要特別注意）：「利多出盡」表面是利多但實際是看空訊號 -> negative；「利空出盡」表面是利空但實際是看多訊號 -> positive
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
    messages.append({"role": "user", "content": text[:200]})
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


def load_texts_for_stock(code, name, articles_dir, comments_df_by_article):
    """
    把一檔股票的「文章本身」+「文章底下所有留言」攤平成一份清單，
    每一列都是一則待分類文字，並記錄它屬於哪個article_id、哪一天。
    """
    path = os.path.join(articles_dir, f"{code}_{name}.csv")
    if not os.path.exists(path):
        return None
    try:
        articles = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        # 這檔股票本來就沒有（或全部被廣告規則清掉）任何文章，
        # 檔案是空的（甚至可能連header都沒有），當作沒有資料處理，
        # 不要讓單一檔股票沒資料就讓整支程式當掉。
        return None
    if len(articles) == 0:
        return None
    rows = []
    for _, art in articles.iterrows():
        aid = art["article_id"]
        art_date = art["createDate"]
        # 即使讀csv時指定dtype=str，內容真的是空白的儲存格pandas還是會
        # 存成NaN（float型別），不是空字串，直接呼叫.strip()會噴
        # AttributeError，一定要先轉成字串才安全。
        raw_text = art.get("text")
        art_text = "" if pd.isna(raw_text) else str(raw_text).strip()
        if art_text:
            rows.append({"article_id": aid, "resolved_date": art_date, "content": art_text})
        for c_text in comments_df_by_article.get(aid, []):
            c_text = "" if pd.isna(c_text) else str(c_text).strip()
            if c_text:
                rows.append({"article_id": aid, "resolved_date": art_date, "content": c_text})
    if not rows:
        return None
    return pd.DataFrame(rows)


def aggregate_by_article(df):
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

    grouped = df.groupby(["article_id", "resolved_date"], as_index=False).apply(
        compute_factor, include_groups=False
    )
    return grouped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用Breeze-7B對CMoney 267檔股票資料做情緒推論（含斷點續傳）")
    parser.add_argument("--articles-dir", type=str, default="cmoney_267_clean")
    parser.add_argument("--comments-path", type=str, default="cmoney_267_comments/all_comments_clean.csv")
    parser.add_argument("--out-dir", type=str, default="sentiment_output_cmoney")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--only-codes", type=str, default=None)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    STOCK_LIST = STOCK_LIST_267
    if args.only_codes:
        wanted = set(args.only_codes.split(","))
        STOCK_LIST = [s for s in STOCK_LIST if s[0] in wanted]
        print(f"=== 只處理指定股票：{[s[0] for s in STOCK_LIST]} ===")

    print("=== 讀取留言資料，依article_id分組（一次性讀取，避免每檔股票重複讀大檔） ===")
    comments_raw = pd.read_csv(args.comments_path, dtype=str)
    comments_df_by_article = comments_raw.groupby("article_id")["text"].apply(list).to_dict()
    print(f"留言共 {len(comments_raw)} 則，涵蓋 {len(comments_df_by_article)} 篇文章\n")
    del comments_raw

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
        out_path = os.path.join(args.out_dir, f"{code}_{name}_sentiment.csv")
        partial_path = os.path.join(args.out_dir, f"{code}_{name}.partial.csv")

        if os.path.exists(out_path):
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}（{industry}）已存在，跳過")
            continue

        df = load_texts_for_stock(code, name, args.articles_dir, comments_df_by_article)
        if df is None or len(df) == 0:
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}：沒有可分類的文字，跳過")
            continue

        if args.start_date or args.end_date:
            dates = pd.to_datetime(df["resolved_date"])
            mask = pd.Series(True, index=df.index)
            if args.start_date:
                mask &= dates >= pd.Timestamp(args.start_date)
            if args.end_date:
                mask &= dates <= pd.Timestamp(args.end_date)
            df = df[mask].reset_index(drop=True)
        if len(df) == 0:
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}：篩選日期範圍後沒有資料，跳過")
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
            print(f"[{i}/{len(STOCK_LIST)}] {code} {name}（{industry}）：{len(texts)} 則文字，開始分類...")

        while start_idx < len(texts):
            end_idx = min(start_idx + args.chunk_size, len(texts))
            chunk_labels = classify_batch(texts[start_idx:end_idx], tok, model, batch_size=args.batch_size)
            labels.extend(chunk_labels)
            pd.DataFrame({"text_sentiment": labels}).to_csv(partial_path, index=False)
            start_idx = end_idx
            write_progress(args.out_dir, i, len(STOCK_LIST), code, name, start_idx, len(texts), run_started_at)
            print(f"    進度 {start_idx}/{len(texts)}")

        df["text_sentiment"] = labels
        article_summary = aggregate_by_article(df)
        article_summary.insert(0, "stock_code", code)
        article_summary.to_csv(out_path, index=False, encoding="utf-8-sig")
        os.remove(partial_path)
        print(f"    [完成] {len(article_summary)} 篇文章的情緒因子 -> {out_path}")

    progress_path = os.path.join(args.out_dir, PROGRESS_FILE)
    if os.path.exists(progress_path):
        os.remove(progress_path)
    print("\n[完成] CMoney 267檔股票已用Breeze-7B完成情緒推論")
