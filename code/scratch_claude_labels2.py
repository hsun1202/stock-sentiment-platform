# -*- coding: utf-8 -*-
"""將Claude逐則閱讀後產生的第二批800筆標籤填回claude_label_batch2.csv"""
import pandas as pd

labels_A = [
"neutral","neutral","positive","negative","neutral","negative","neutral","neutral","negative","neutral",
"negative","positive","positive","neutral","negative","neutral","neutral","positive","positive","neutral",
"negative","positive","neutral","neutral","negative","neutral","neutral","neutral","neutral","positive",
"positive","negative","neutral","neutral","neutral","neutral","negative","neutral","neutral","neutral",
"neutral","positive","neutral","positive","negative","neutral","neutral","positive","negative","negative",
"positive","negative","neutral","neutral","neutral","positive","negative","neutral","neutral","neutral",
"neutral","neutral","neutral","neutral","negative","neutral","negative","negative","negative","neutral",
"positive","neutral","negative","negative","negative","neutral","neutral","positive","negative","negative",
"negative","negative","neutral","positive","neutral","positive","positive","neutral","negative","neutral",
"neutral","neutral","neutral","positive","neutral","neutral","positive","neutral","positive","positive",
]

labels_B = [
"negative","neutral","neutral","negative","negative","positive","negative","neutral","neutral","neutral",
"negative","neutral","neutral","neutral","positive","neutral","negative","neutral","positive","neutral",
"neutral","neutral","positive","negative","neutral","neutral","neutral","positive","neutral","negative",
"neutral","negative","negative","neutral","positive","negative","neutral","negative","neutral","negative",
"negative","neutral","negative","positive","positive","negative","neutral","neutral","neutral","neutral",
"neutral","negative","negative","negative","negative","positive","neutral","negative","neutral","neutral",
"negative","negative","negative","positive","neutral","neutral","neutral","negative","neutral","positive",
"negative","positive","neutral","negative","neutral","negative","negative","negative","neutral","neutral",
"negative","negative","neutral","negative","neutral","positive","negative","neutral","positive","neutral",
"negative","neutral","neutral","negative","neutral","positive","negative","neutral","positive","neutral",
]

labels_C = [
"neutral","positive","negative","positive","positive","negative","positive","negative","negative","negative",
"positive","positive","negative","negative","positive","negative","negative","negative","neutral","neutral",
"negative","positive","neutral","negative","neutral","neutral","negative","negative","negative","positive",
"neutral","negative","positive","negative","neutral","neutral","positive","positive","negative","neutral",
"negative","neutral","neutral","positive","negative","negative","neutral","neutral","positive","positive",
"negative","neutral","neutral","neutral","negative","negative","positive","neutral","negative","positive",
"neutral","positive","neutral","neutral","negative","negative","positive","neutral","neutral","neutral",
"negative","negative","positive","positive","neutral","positive","neutral","neutral","neutral","positive",
"neutral","negative","neutral","neutral","neutral","negative","positive","neutral","positive","neutral",
"neutral","neutral","neutral","neutral","neutral","negative","positive","neutral","positive","neutral",
]

labels_D = [
"neutral","negative","neutral","neutral","neutral","neutral","negative","negative","negative","negative",
"negative","positive","positive","neutral","negative","negative","neutral","neutral","neutral","positive",
"positive","neutral","neutral","neutral","neutral","neutral","negative","neutral","neutral","neutral",
"neutral","neutral","neutral","negative","neutral","negative","neutral","negative","neutral","negative",
"positive","negative","negative","positive","neutral","neutral","positive","neutral","neutral","negative",
"neutral","negative","negative","negative","neutral","negative","negative","negative","neutral","positive",
"neutral","neutral","neutral","negative","neutral","negative","negative","neutral","neutral","negative",
"neutral","positive","neutral","positive","neutral","positive","neutral","neutral","neutral","positive",
"neutral","neutral","neutral","negative","neutral","negative","negative","neutral","neutral","negative",
"neutral","neutral","neutral","positive","neutral","positive","neutral","negative","negative","neutral",
]

labels_E = [
"positive","neutral","positive","positive","positive","negative","neutral","negative","neutral","negative",
"neutral","positive","positive","negative","neutral","neutral","neutral","negative","positive","neutral",
"positive","positive","neutral","positive","neutral","neutral","positive","neutral","neutral","neutral",
"neutral","neutral","neutral","positive","neutral","neutral","positive","positive","negative","neutral",
"positive","neutral","negative","neutral","negative","neutral","neutral","neutral","neutral","neutral",
"neutral","negative","positive","negative","positive","positive","neutral","negative","negative","negative",
"neutral","neutral","negative","neutral","negative","neutral","positive","neutral","negative","neutral",
"positive","neutral","neutral","neutral","neutral","neutral","negative","neutral","neutral","negative",
"negative","negative","neutral","neutral","positive","neutral","neutral","neutral","positive","neutral",
"negative","neutral","neutral","positive","positive","neutral","negative","negative","positive","neutral",
]

labels_F = [
"positive","neutral","negative","negative","neutral","negative","negative","neutral","neutral","neutral",
"positive","neutral","neutral","positive","negative","neutral","negative","neutral","negative","neutral",
"positive","negative","neutral","neutral","negative","negative","negative","neutral","neutral","negative",
"neutral","negative","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral",
"neutral","positive","neutral","negative","positive","neutral","neutral","neutral","neutral","neutral",
"negative","positive","positive","positive","positive","neutral","neutral","neutral","neutral","neutral",
"neutral","positive","neutral","neutral","negative","neutral","neutral","neutral","neutral","neutral",
"neutral","positive","negative","neutral","negative","positive","negative","negative","negative","positive",
"neutral","neutral","positive","negative","negative","negative","neutral","negative","neutral","negative",
"negative","neutral","neutral","positive","neutral","neutral","neutral","positive","positive","neutral",
]

labels_G = [
"negative","neutral","negative","neutral","neutral","neutral","neutral","neutral","negative","neutral",
"neutral","positive","negative","neutral","neutral","negative","neutral","neutral","neutral","positive",
"neutral","negative","positive","positive","negative","negative","positive","neutral","negative","neutral",
"negative","neutral","positive","neutral","positive","negative","negative","neutral","neutral","negative",
"positive","neutral","neutral","neutral","negative","neutral","neutral","neutral","neutral","neutral",
"positive","neutral","negative","neutral","negative","negative","positive","positive","positive","positive",
"neutral","negative","neutral","neutral","neutral","neutral","negative","negative","neutral","positive",
"negative","neutral","positive","neutral","neutral","positive","neutral","negative","negative","neutral",
"positive","negative","negative","negative","negative","positive","neutral","neutral","positive","negative",
"neutral","neutral","positive","neutral","neutral","negative","negative","neutral","negative","neutral",
]

labels_H = [
"positive","neutral","neutral","negative","negative","neutral","neutral","negative","neutral","neutral",
"negative","neutral","neutral","neutral","neutral","positive","neutral","positive","positive","neutral",
"neutral","negative","negative","positive","positive","neutral","neutral","neutral","negative","neutral",
"neutral","neutral","negative","neutral","neutral","positive","neutral","neutral","neutral","neutral",
"negative","neutral","negative","positive","negative","neutral","negative","negative","neutral","positive",
"positive","neutral","neutral","negative","negative","positive","neutral","negative","positive","neutral",
"neutral","negative","neutral","neutral","neutral","positive","neutral","negative","negative","positive",
"neutral","negative","neutral","neutral","neutral","neutral","positive","positive","neutral","neutral",
"neutral","neutral","negative","positive","positive","neutral","neutral","neutral","neutral","neutral",
"negative","positive","negative","neutral","negative","negative","negative","neutral","neutral","negative",
]

labels = labels_A + labels_B + labels_C + labels_D + labels_E + labels_F + labels_G + labels_H

assert len(labels) == 800, f"應該要有800筆，實際{len(labels)}筆"
for name, block in [("A",labels_A),("B",labels_B),("C",labels_C),("D",labels_D),
                     ("E",labels_E),("F",labels_F),("G",labels_G),("H",labels_H)]:
    assert len(block) == 100, f"block {name} 應該100筆，實際{len(block)}筆"

df = pd.read_csv("claude_label_batch2.csv")
assert len(df) == len(labels), f"筆數對不上：csv={len(df)}, labels={len(labels)}"
df["claude_label"] = labels
df.to_csv("claude_label_batch2.csv", index=False, encoding="utf-8-sig")
print("已寫入", len(df), "筆Claude標籤")
print(df["claude_label"].value_counts())
