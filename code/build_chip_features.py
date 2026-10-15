# -*- coding: utf-8 -*-
"""
整理籌碼資料並併入v2資料集：
  1. 補抓交易日曆(期交所PCR日期)中還沒下載的三大法人/融資融券(主要是2021-04-06)
  2. 解析 chip_raw/t86(股)、chip_raw/margin(張) -> chip_daily.csv（每檔每天一列）
  3. 併入 final_dataset_v2/all_stocks_v2.csv -> final_dataset_v2/all_stocks_v2_chip.csv
     三大法人當天沒紀錄 = 當天無法人交易 -> 0；融資融券沒紀錄 = 該股不能信用交易 -> NaN並標記
  4. 一致性檢查：今日「前日餘額」應等於前一交易日「今日餘額」
python build_chip_features.py
"""
import io
import os
import glob
import json
import time
import requests
import numpy as np
import pandas as pd

from stock_list_267 import STOCK_LIST_267

CODES = {c for c, _, _ in STOCK_LIST_267}
H = {"User-Agent": "Mozilla/5.0"}
URL = {
    "t86": "https://www.twse.com.tw/rwd/zh/fund/T86?date={d}&selectType=ALLBUT0999&response=json",
    "margin": "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={d}&selectType=ALL&response=json",
}

pcr = pd.concat([pd.read_csv(io.StringIO(open(f, encoding="utf-8").read()), index_col=False).iloc[:, :7]
                 for f in sorted(glob.glob("chip_raw/pcr/*.csv"))])
pcr.columns = ["Date", "put_vol", "call_vol", "pc_vol_ratio", "put_oi", "call_oi", "pc_oi_ratio"]
pcr["Date"] = pd.to_datetime(pcr.Date).dt.strftime("%Y-%m-%d")
pcr = pcr.drop_duplicates("Date").sort_values("Date")
CAL = pcr.Date.tolist()

for k, u in URL.items():
    for d in CAL:
        f = f"chip_raw/{k}/{d.replace('-', '')}.json"
        if not os.path.exists(f):
            j = requests.get(u.format(d=d.replace("-", "")), headers=H, timeout=30).json()
            json.dump(j, open(f, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"補抓 {k} {d}: stat={j.get('stat')}", flush=True)
            time.sleep(4)


def n(x):
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return np.nan


rows = []
for d in CAL:
    dd = d.replace("-", "")
    t = json.load(open(f"chip_raw/t86/{dd}.json", encoding="utf-8"))
    for r in t.get("data") or []:
        c = str(r[0]).strip()
        if c in CODES:
            rows.append({"stock_code": c, "Date": d, "foreign_net": n(r[4]) + n(r[7]), "trust_net": n(r[10]),
                         "dealer_net": n(r[11]), "inst_net": n(r[18]), "_t86": 1})
t86 = pd.DataFrame(rows)

rows = []
for d in CAL:
    j = json.load(open(f"chip_raw/margin/{d.replace('-', '')}.json", encoding="utf-8"))
    tb = [t for t in (j.get("tables") or []) if t.get("data") and len(t["data"]) > 10]
    for r in (tb[0]["data"] if tb else []):
        c = str(r[0]).strip()
        if c in CODES:
            rows.append({"stock_code": c, "Date": d, "margin_buy": n(r[2]), "margin_sell": n(r[3]),
                         "margin_bal_prev": n(r[5]), "margin_bal": n(r[6]), "short_buy": n(r[8]),
                         "short_sell": n(r[9]), "short_bal_prev": n(r[11]), "short_bal": n(r[12]),
                         "credit_note": str(r[15]).strip(), "_margin": 1})
mg = pd.DataFrame(rows)

chip = t86.merge(mg, on=["stock_code", "Date"], how="outer").merge(pcr[["Date", "pc_vol_ratio", "pc_oi_ratio"]], on="Date", how="left")
chip.to_csv("chip_daily.csv", index=False, encoding="utf-8-sig")
print(f"chip_daily: {len(chip)} 列、{chip.stock_code.nunique()} 檔、{chip.Date.min()}~{chip.Date.max()}")

# ---------- 併入v2 ----------
v2 = pd.read_csv("final_dataset_v2/all_stocks_v2.csv", dtype={"stock_code": str})
m = v2.merge(chip.drop(columns=["pc_vol_ratio", "pc_oi_ratio"]), on=["stock_code", "Date"], how="left")
m = m.merge(pcr[["Date", "pc_vol_ratio", "pc_oi_ratio"]], on="Date", how="left")
for c in ("foreign_net", "trust_net", "dealer_net", "inst_net"):
    m[c] = m[c].fillna(0.0)
m["_t86"] = m["_t86"].fillna(0).astype(int)
m["margin_eligible"] = m["_margin"].fillna(0).astype(int)
vol = m.Volume.replace(0, np.nan)
for c in ("foreign", "trust", "dealer", "inst"):
    m[f"{c}_ratio"] = (m[f"{c}_net"] / vol).fillna(0.0)
m["margin_chg"] = ((m.margin_bal - m.margin_bal_prev) / m.margin_bal_prev.replace(0, np.nan))
m["short_chg"] = ((m.short_bal - m.short_bal_prev) / m.short_bal_prev.replace(0, np.nan))
m["short_margin_ratio"] = m.short_bal / m.margin_bal.replace(0, np.nan)
m = m.drop(columns=["_margin"])
m.to_csv("final_dataset_v2/all_stocks_v2_chip.csv", index=False, encoding="utf-8-sig")

# ---------- 檢查 ----------
print(f"\n併入後 {len(m)} 列（v2 {len(v2)} 列）；PCR缺值 {int(m.pc_vol_ratio.isna().sum())}")
traded = m.Volume > 0
print(f"有成交的股票日中，有三大法人紀錄 {m.loc[traded, '_t86'].mean():.1%}；可信用交易(有融資融券紀錄) {m.loc[traded, 'margin_eligible'].mean():.1%}")
print(f"不能信用交易的股票日 {int((m.margin_eligible == 0).sum())} 列，涉及 {m.loc[m.margin_eligible == 0, 'stock_code'].nunique()} 檔")
over = (m.inst_ratio.abs() > 1).mean()
print(f"|三大法人買賣超/成交量|>1 的比例 {over:.2%}（T86含零股/盤後/鉅額，成交量來源Yahoo，少量超過可接受）")
mm = m[m.margin_eligible == 1].sort_values(["stock_code", "Date"])
prev_ok = mm.groupby("stock_code").margin_bal.shift(1)
mism = ((mm.margin_bal_prev - prev_ok).abs() > 0.5) & prev_ok.notna()
print(f"融資『前日餘額』≠前一交易日『今日餘額』: {int(mism.sum())} / {int(prev_ok.notna().sum())}")
if mism.any():
    print(mm.loc[mism, ["stock_code", "Date", "margin_bal_prev"]].assign(prev_day_bal=prev_ok[mism]).head(10).to_string(index=False))
print("\n數值分布：")
print(m[["foreign_ratio", "trust_ratio", "inst_ratio", "margin_chg", "short_chg", "short_margin_ratio", "pc_vol_ratio", "pc_oi_ratio"]]
      .describe(percentiles=[.01, .5, .99]).round(4).T.to_string())
