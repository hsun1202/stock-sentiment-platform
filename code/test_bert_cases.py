# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
model_dir = sys.argv[1] if len(sys.argv) > 1 else "bert_sentiment_model/final"
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

tests = [
    "我覺得不看好這檔",
    "大家都不樂觀",
    "這檔我很看好",
    "跌停 塊陶",
    "有聽說電動車多花幾千鎂一櫃要趕出貨",
    "上一波已經出貨了 ETF現在撿太扯了吧",
    "利多出盡",
    "利空出盡",
    "歐印",
    "推",
    "這波應該要獲利了結了吧",
    "感覺會繼續噴出",
    "業績優於預期 但是股價卻不漲",
    "外資一直倒貨 完全不想上車",
]

out_path = "bert_case_test_result.txt"
with open(out_path, "w", encoding="utf-8") as f:
    with torch.no_grad():
        for t in tests:
            enc = tok(t, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1)[0]
            label = "positive" if probs[1] > probs[0] else "negative"
            line = f"{t!r:35s} -> {label}  P(pos)={probs[1]:.3f}"
            print(line)
            f.write(line + "\n")
print(f"\n[done] written to {out_path}")
