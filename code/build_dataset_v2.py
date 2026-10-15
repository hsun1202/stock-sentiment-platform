# -*- coding: utf-8 -*-
"""
建立v2資料集（修正版）：
  1. 用還原股價 prices_adj/ 重算16項技術指標（不做全期間min-max縮放，避免測試期資訊洩漏）
  2. 標記異常價格日：還原後單日|報酬|>10.5%且非上市前5個交易日（減資、停牌復牌、資料錯誤）
  3. PTT情緒：非交易日(週末/假日)的留言歸到「下一個交易日」，不再丟棄
輸出：
  technical_output_267_adj/{code}_{name}_technical.csv
  final_dataset_v2/all_stocks_v2.csv（2020-2024，每檔每交易日一列）
交易日曆取自期交所PCR日期（含yfinance缺的2021-04-06）。

python build_dataset_v2.py
"""
import os
import io
import glob
import numpy as np
import pandas as pd

from stock_list_267 import STOCK_LIST_267
from stock_indicators import calculate_technical_indicators

S, E = "2020-01-01", "2024-12-31"
TECH_DIR, OUT_DIR = "technical_output_267_adj", "final_dataset_v2"
os.makedirs(TECH_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

pcr = pd.concat([pd.read_csv(io.StringIO(open(f, encoding="utf-8").read()), index_col=False).iloc[:, :1]
                 for f in sorted(glob.glob("chip_raw/pcr/*.csv"))])
CAL = pd.DatetimeIndex(sorted(set(pd.to_datetime(pcr.iloc[:, 0]))))
CAL_S = CAL.strftime("%Y-%m-%d")

# ---------- PTT情緒：歸到「當天或下一個交易日」 ----------
sent = pd.concat([pd.read_csv(f, dtype={"stock_code": str})
                  for f in glob.glob("sentiment_output_267_stitched_2020_2024/*_sentiment.csv")], ignore_index=True)
d = pd.to_datetime(sent.resolved_date)
pos = CAL.searchsorted(d, side="left")
sent = sent[pos < len(CAL)].copy()
sent["trade_date"] = CAL_S[pos[pos < len(CAL)]]
sent["moved"] = sent.trade_date != sent.resolved_date
sent["moved_comments"] = sent.total_comments * sent.moved
daily = sent.groupby(["stock_code", "trade_date"], as_index=False).agg(
    article_count=("article_id", "count"), total_comments=("total_comments", "sum"),
    positive_count=("positive_count", "sum"), negative_count=("negative_count", "sum"),
    neutral_count=("neutral_count", "sum"), moved_comments=("moved_comments", "sum"),
)
daily["sentiment_factor"] = ((daily.positive_count - daily.negative_count) / daily.total_comments).where(daily.total_comments > 0, 0.0)
print(f"PTT情緒: 留言 {int(sent.total_comments.sum())} 則，其中從非交易日移到下一個交易日 {int(sent.loc[sent.moved, 'total_comments'].sum())} 則", flush=True)

# ---------- 逐檔：技術指標 + 異常標記 + 合併 ----------
frames, rep = [], []
for code, name, industry in STOCK_LIST_267:
    f = glob.glob(f"prices_adj/{code}_*.csv")
    if not f:
        rep.append((code, name, "無還原股價"))
        continue
    p = pd.read_csv(f[0], dtype={"Date": str}).set_index("Date")
    p.index = pd.to_datetime(p.index)
    tech = calculate_technical_indicators(p[["Open", "High", "Low", "Close", "Volume"]])
    tech["close_unadj"] = p.close_unadj
    tech["ret1"] = tech.Close.pct_change()
    nth = np.arange(1, len(tech) + 1)
    tech["abnormal_price"] = ((tech.ret1.abs() > 0.105) & (nth > 5)).astype(int)
    tech.index.name = "Date"
    tech.to_csv(os.path.join(TECH_DIR, f"{code}_{name}_technical.csv"), encoding="utf-8-sig")

    w = tech.loc[S:E].reset_index()
    w["Date"] = w.Date.dt.strftime("%Y-%m-%d")
    if w.empty:
        rep.append((code, name, "2020-2024無資料"))
        continue
    s = daily[daily.stock_code == code].rename(columns={"trade_date": "Date"}).drop(columns="stock_code")
    m = w.merge(s, on="Date", how="left")
    fill = ["article_count", "total_comments", "positive_count", "negative_count", "neutral_count",
            "moved_comments", "sentiment_factor"]
    m[fill] = m[fill].fillna(0.0)
    m.insert(0, "stock_code", code)
    m.insert(1, "stock_name", name)
    m.insert(2, "industry", industry)
    frames.append(m)

out = pd.concat(frames, ignore_index=True)
out.to_csv(os.path.join(OUT_DIR, "all_stocks_v2.csv"), index=False, encoding="utf-8-sig")

kept = int(out.total_comments.sum())
print(f"股票 {out.stock_code.nunique()} 檔，列數 {len(out)}，日期 {out.Date.min()}~{out.Date.max()}")
print(f"保留留言 {kept} / {int(sent.total_comments.sum())}（舊版只保留 1,509,283）")
print(f"異常價格日 {int(out.abnormal_price.sum())} 列；(股票,日期)重複 {int(out.duplicated(['stock_code', 'Date']).sum())}")
nan = out.select_dtypes("number").isna().sum()
print("NaN欄位:", nan[nan > 0].to_dict())
print("略過:", rep)
