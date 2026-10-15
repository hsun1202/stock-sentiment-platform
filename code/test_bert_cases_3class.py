# -*- coding: utf-8 -*-
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = {0: "negative", 1: "neutral", 2: "positive"}

model_dir = sys.argv[1] if len(sys.argv) > 1 else "bert_sentiment_model_3class/final"
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

tests = [
    ("我覺得不看好這檔", "negative"),
    ("大家都不樂觀", "negative"),
    ("這檔我很看好", "positive"),
    ("跌停 塊陶", "negative"),
    ("有聽說電動車多花幾千鎂一櫃要趕出貨", "爭議"),
    ("上一波已經出貨了 ETF現在撿太扯了吧", "negative"),
    ("利多出盡", "negative"),
    ("利空出盡", "positive"),
    ("歐印", "positive"),
    ("推", "neutral?"),
    ("這波應該要獲利了結了吧", "negative"),
    ("感覺會繼續噴出", "positive"),
    ("業績優於預期 但是股價卻不漲", "negative(轉折)"),
    ("外資一直倒貨 完全不想上車", "negative"),
    ("今天量縮 觀望中", "neutral"),
    ("明天記得開盤", "neutral"),
    ("有人知道除息日嗎", "neutral"),
]

out_path = "bert3_case_test_result.txt"
with open(out_path, "w", encoding="utf-8") as f:
    with torch.no_grad():
        for t, truth in tests:
            enc = tok(t, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1)[0]
            pred_idx = int(torch.argmax(probs))
            label = LABELS[pred_idx]
            line = f"{t!r:35s} 應為={truth:14s} -> {label:9s} (neg={probs[0]:.2f} neu={probs[1]:.2f} pos={probs[2]:.2f})"
            print(line)
            f.write(line + "\n")
print(f"\n[done] written to {out_path}")
