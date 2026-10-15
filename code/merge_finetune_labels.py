# -*- coding: utf-8 -*-
"""
合併「392筆人工標籤」+「300筆Claude標籤」，供微調用
============================================================
兩種標籤來源準確度不完全相同（人工是黃金標準；Claude標籤先前的
Kappa分析顯示跟人工只有中等一致性），所以保留source欄位，之後
如果想只用人工標籤驗證、或分析Claude標籤有沒有拖累微調效果，
都能回頭用source欄位切分。

使用方式：
    python merge_finetune_labels.py
"""

import pandas as pd

human = pd.read_csv("all_human_labels.csv")[["content", "human_label"]].rename(columns={"human_label": "label"})
human["source"] = "human"

claude = pd.read_csv("claude_label_batch.csv")[["content", "claude_label"]].rename(columns={"claude_label": "label"})
claude["source"] = "claude"

combined = pd.concat([human, claude], ignore_index=True)
combined["content"] = combined["content"].astype(str).str.strip()
before = len(combined)
combined = combined.drop_duplicates(subset=["content"])
print(f"合併前：{before}，去重後：{len(combined)}")
print(combined["source"].value_counts())
print(combined["label"].value_counts())

combined.to_csv("combined_labels_for_finetune.csv", index=False, encoding="utf-8-sig")
print("\n已輸出 combined_labels_for_finetune.csv")
