# -*- coding: utf-8 -*-
"""把 2025 年 PTT 個股情緒（ptt_2025_stock_labels.csv）併進 final_dataset_v3_2025 → final_dataset_v3_2025_ptt/all_stocks_v3.csv
（2015-2024 的 PTT 欄位不動；2025 原本幾乎為 0，改成新分類結果；欄位定義同 build_dataset_v3.py：pre＝前一收盤後~今天開盤前、intra＝盤中）
python make_dataset_2025_with_ptt.py"""
import os

import pandas as pd

PCOLS = ["ptt_pre_pos", "ptt_pre_neg", "ptt_pre_n", "ptt_intra_pos", "ptt_intra_neg", "ptt_intra_n"]
base = pd.read_csv("final_dataset_v3_2025/all_stocks_v3.csv", dtype={"stock_code": str})
lab = pd.read_csv("ptt_2025_stock_labels.csv", dtype={"stock_code": str})
lab["Date"] = pd.to_datetime(lab.Date).dt.strftime("%Y-%m-%d")
agg = lab.assign(pos=lab.text_sentiment.eq("positive").astype(int), neg=lab.text_sentiment.eq("negative").astype(int)) \
    .groupby(["stock_code", "Date", "part"]).agg(n=("pos", "size"), pos=("pos", "sum"), neg=("neg", "sum")).unstack("part").fillna(0)
agg.columns = [f"ptt_{p}_{k}" for k, p in agg.columns]
agg = agg.reset_index()
is25 = base.Date.str[:4] == "2025"
before = base.loc[is25, PCOLS].sum().to_dict()
base.loc[is25, PCOLS] = 0
m = base[["stock_code", "Date"]].merge(agg, on=["stock_code", "Date"], how="left")
for c in PCOLS:
    base.loc[is25, c] = m.loc[is25, c].fillna(0).values
os.makedirs("final_dataset_v3_2025_ptt", exist_ok=True)
base.to_csv("final_dataset_v3_2025_ptt/all_stocks_v3.csv", index=False, encoding="utf-8-sig")
print("2025 PTT 欄位（原→新）：", {c: (int(before[c]), int(base.loc[is25, c].sum())) for c in PCOLS})
print("2025 有留言的股票日：", int(((base.loc[is25, "ptt_pre_n"] + base.loc[is25, "ptt_intra_n"]) > 0).sum()))
