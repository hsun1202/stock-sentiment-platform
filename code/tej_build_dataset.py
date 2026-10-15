# -*- coding: utf-8 -*-
"""
把資料集的「股價」換成 TEJ（調整股價(日)-除權息調整），其餘欄位（PTT、籌碼）完全不動 → 穩健性檢查用
  價格欄位：Open/High/Low/Close（TEJ 還原價；無成交日的開高低以收盤價補）、Volume（千股×1000）
  技術指標：用 TEJ 價格、同一支 calculate_technical_indicators 重算（TEJ 自 2014-01-02 起，暖機期比原本長）
  close_unadj：TEJ 真正的未調整收盤價；ret1、abnormal_price 依原規則重算
  列集合：與原資料集相同的（股票, 日期），TEJ 沒有的列捨棄
輸出：final_dataset_v3_tej/（PTT=原 Breeze）以及下列各版本只換 ptt_* 欄位的 TEJ 版：
        _tejft（微調BERT）、_tejp20（PTT 只留2020-24）、_tejcm2/_tejcmc/_tejcmbb2（CMoney 清理後）
      tej_build_dataset.log
python tej_build_dataset.py
"""
import os

import numpy as np
import pandas as pd

from stock_indicators import calculate_technical_indicators

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


base = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str})
codes = sorted(base.stock_code.unique())
px = pd.read_pickle("tej_raw/tej_price_daily.pkl")
px = px[px.code.isin(codes)].sort_values(["code", "date"])
chip = pd.read_pickle("tej_raw/tej_chip_daily.pkl")[["code", "date", "未調整收盤價(元)"]]
px = px.merge(chip, on=["code", "date"], how="left")
say(f"原資料集 {len(base):,} 列、{len(codes)} 檔；TEJ 價格涵蓋其中 {px.code.nunique()} 檔、{len(px):,} 列（2014-01-02 起）")

frames = []
for code, p in px.groupby("code"):
    p = p.set_index("date")
    d = pd.DataFrame({"Open": p["開盤價(元)"], "High": p["最高價(元)"], "Low": p["最低價(元)"], "Close": p["收盤價(元)"],
                      "Volume": p["成交量(千股)"].fillna(0) * 1000.0})
    for c in ("Open", "High", "Low"):
        d[c] = d[c].fillna(d.Close)
    tech = calculate_technical_indicators(d)
    tech["close_unadj"] = p["未調整收盤價(元)"]
    tech["ret1"] = tech.Close.pct_change()
    tech["tej_ret"] = p["報酬率％"] / 100
    nth = np.arange(1, len(tech) + 1)
    limit = np.where(tech.index < "2015-06-01", 0.07, 0.10) + 0.005
    tech["abnormal_price"] = ((tech.ret1.abs() > limit) & (nth > 5)).astype(int)
    tech["mktcap"] = p["市值(百萬元)"]
    tech["limit_flag"] = p["漲跌停"]
    w = tech.reset_index().rename(columns={"date": "Date"})
    w["Date"] = w.Date.dt.strftime("%Y-%m-%d")
    w.insert(0, "stock_code", code)
    frames.append(w)
T = pd.concat(frames, ignore_index=True)
say(f"自算報酬與 TEJ 報酬率欄位：相關 {T.ret1.corr(T.tej_ret):.5f}、差>0.2pp 的比例 {((T.ret1 - T.tej_ret).abs() > 0.002).mean():.4%}")

PRICE = [c for c in base.columns[4:28]]          # Open ... abnormal_price
assert PRICE[0] == "Open" and PRICE[-1] == "abnormal_price", PRICE
keep = base.drop(columns=PRICE)
new = keep.merge(T[["stock_code", "Date"] + PRICE + ["mktcap", "limit_flag"]], on=["stock_code", "Date"], how="inner")
new = new[list(base.columns) + ["mktcap", "limit_flag"]].sort_values(["stock_code", "Date"]).reset_index(drop=True)
say(f"TEJ 版資料集 {len(new):,} 列（原 {len(base):,}，捨棄 {len(base) - len(new):,} 列 TEJ 沒有的）；異常價格日 {int(new.abnormal_price.sum())}（原 {int(base.abnormal_price.sum())}）")
yr = new.Date.str[:4]
say("每年列數：" + " ".join(f"{y}:{n:,}" for y, n in yr.value_counts().sort_index().items()))
os.makedirs("final_dataset_v3_tej", exist_ok=True)
new.to_csv("final_dataset_v3_tej/all_stocks_v3.csv", index=False, encoding="utf-8-sig")

PTT = [c for c in base.columns if c.startswith("ptt_")]
for src, dst in [("ft", "tejft"), ("p20", "tejp20"), ("cm2", "tejcm2"), ("cmc", "tejcmc"), ("cmbb2", "tejcmbb2")]:
    f = f"final_dataset_v3_{src}/all_stocks_v3.csv"
    if not os.path.exists(f):
        say(f"（略過 {src}：找不到）")
        continue
    v = pd.read_csv(f, dtype={"stock_code": str}, usecols=["stock_code", "Date"] + PTT)
    x = new.drop(columns=PTT).merge(v, on=["stock_code", "Date"], how="left")
    assert len(x) == len(new) and x[PTT].notna().all().all()
    os.makedirs(f"final_dataset_v3_{dst}", exist_ok=True)
    x[new.columns].to_csv(f"final_dataset_v3_{dst}/all_stocks_v3.csv", index=False, encoding="utf-8-sig")
    say(f"final_dataset_v3_{dst} 完成")
open("tej_build_dataset.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
