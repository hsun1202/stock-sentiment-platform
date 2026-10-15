# -*- coding: utf-8 -*-
"""將Claude逐則閱讀後產生的標籤填回claude_label_batch.csv"""
import pandas as pd

labels = [
"neutral","neutral","neutral","neutral","neutral","neutral","positive","neutral","neutral","negative",
"neutral","negative","positive","neutral","neutral","neutral","positive","negative","neutral","positive",
"negative","positive","neutral","neutral","negative","neutral","neutral","neutral","neutral","positive",
"positive","negative","neutral","neutral","neutral","neutral","negative","neutral","neutral","neutral",
"neutral","positive","neutral","positive","negative","neutral","neutral","positive","negative","negative",
"positive","negative","neutral","neutral","neutral","positive","negative","neutral","neutral","neutral",
"neutral","neutral","neutral","neutral","negative","neutral","negative","negative","negative","neutral",
"positive","neutral","negative","negative","negative","neutral","neutral","positive","negative","negative",
"negative","negative","neutral","positive","neutral","positive","positive","neutral","negative","neutral",
"neutral","neutral","neutral","positive","neutral","neutral","positive","neutral","positive","positive",

"neutral","neutral","negative","positive","neutral","positive","positive","neutral","negative","neutral",
"negative","neutral","neutral","neutral","positive","neutral","negative","neutral","positive","neutral",
"neutral","neutral","positive","negative","neutral","neutral","neutral","positive","neutral","negative",
"neutral","negative","negative","neutral","negative","negative","neutral","positive","positive","positive",
"neutral","neutral","negative","positive","neutral","negative","positive","neutral","neutral","negative",
"negative","neutral","negative","negative","neutral","positive","positive","positive","neutral","positive",
"neutral","neutral","neutral","positive","negative","negative","positive","negative","positive","neutral",
"neutral","negative","neutral","neutral","positive","negative","neutral","neutral","negative","neutral",
"positive","neutral","positive","neutral","neutral","negative","negative","positive","negative","neutral",
"neutral","negative","neutral","negative","negative","neutral","positive","negative","negative","neutral",

"neutral","negative","neutral","negative","positive","negative","neutral","negative","neutral","neutral",
"negative","neutral","negative","neutral","positive","positive","negative","neutral","neutral","neutral",
"neutral","negative","neutral","neutral","positive","negative","neutral","neutral","negative","neutral",
"neutral","negative","neutral","neutral","neutral","negative","neutral","neutral","negative","neutral",
"positive","positive","neutral","positive","neutral","negative","neutral","positive","negative","neutral",
"negative","positive","positive","negative","positive","neutral","negative","neutral","neutral","neutral",
"neutral","positive","positive","negative","positive","neutral","negative","neutral","neutral","neutral",
"neutral","negative","positive","negative","neutral","positive","neutral","neutral","neutral","neutral",
"positive","negative","positive","neutral","neutral","neutral","neutral","negative","positive","positive",
"neutral","negative","neutral","neutral","neutral","neutral","negative","positive","negative","negative",
]

assert len(labels) == 300, f"應該要有300筆，實際{len(labels)}筆"

df = pd.read_csv("claude_label_batch.csv")
assert len(df) == len(labels), f"筆數對不上：csv={len(df)}, labels={len(labels)}"
df["claude_label"] = labels
df.to_csv("claude_label_batch.csv", index=False, encoding="utf-8-sig")
print("已寫入", len(df), "筆Claude標籤")
print(df["claude_label"].value_counts())
