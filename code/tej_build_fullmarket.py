# -*- coding: utf-8 -*-
"""
TEJ 新檢驗 (3)：全市場資料集（四碼普通股、含已下市），欄位格式與 final_dataset_v3_tej 相同，既有檢驗程式可直接讀。
  股價/技術指標：tej_raw/tej_price_daily.pkl（還原價），作法同 tej_build_dataset.py
  籌碼：tej_raw/tej_chip_daily.pkl → 法人買賣超（張×1000＝股）、融資融券餘額（張），衍生欄位定義同 build_dataset_v3.py
  PTT 欄位一律 0（全市場沒有 PTT 配對，這個資料集只用來檢驗技術面、融資融券、法人、散戶成交比重）
  另加：inst_share / credit_share / daytrade_share（法人成交比重、信用交易比重、資券互抵比例，%）、in265（是否在原 265 檔內）
輸出：final_dataset_v3_tejall/all_stocks_v3.csv、tej_build_fullmarket.log
python tej_build_fullmarket.py
"""
import os

import numpy as np
import pandas as pd

from stock_indicators import calculate_technical_indicators

LOG = []
OUTDIR = os.environ.get("OUTDIR", "final_dataset_v3_tejall")


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


ref = pd.read_csv("final_dataset_v3_tej/all_stocks_v3.csv", dtype={"stock_code": str})
px = pd.read_pickle("tej_raw/tej_price_daily.pkl")
px = px[px.code.str.match(r"^[1-9]\d{3}$")].sort_values(["code", "date"])
chip = pd.read_pickle("tej_raw/tej_chip_daily.pkl").drop_duplicates(["code", "date"])
chip = chip[chip.code.isin(px.code.unique())]
px = px.merge(chip.drop(columns=["name"]), on=["code", "date"], how="left")
say(f"TEJ 四碼普通股 {px.code.nunique()} 檔、{len(px):,} 列（{px.date.min().date()}~{px.date.max().date()}）")

frames = []
for code, p in px.groupby("code"):
    p = p.set_index("date")
    if len(p) < 80:
        continue
    d = pd.DataFrame({"Open": p["開盤價(元)"], "High": p["最高價(元)"], "Low": p["最低價(元)"], "Close": p["收盤價(元)"],
                      "Volume": p["成交量(千股)"].fillna(0) * 1000.0})
    for c in ("Open", "High", "Low"):
        d[c] = d[c].fillna(d.Close)
    t = calculate_technical_indicators(d)
    t["close_unadj"] = p["未調整收盤價(元)"]
    t["ret1"] = t.Close.pct_change()
    nth = np.arange(1, len(t) + 1)
    limit = np.where(t.index < "2015-06-01", 0.07, 0.10) + 0.005
    t["abnormal_price"] = ((t.ret1.abs() > limit) & (nth > 5)).astype(int)
    for c in ("foreign", "trust", "dealer"):
        t[f"{c}_net"] = (p[{"foreign": "外資買賣超(張)", "trust": "投信買賣超(張)", "dealer": "自營買賣超(張)"}[c]] * 1000.0).fillna(0.0)
    t["inst_net"] = t.foreign_net + t.trust_net + t.dealer_net
    t["margin_bal"], t["short_bal"] = p["融資餘額(張)"], p["融券餘額(張)"]
    t["margin_bal_prev"], t["short_bal_prev"] = t.margin_bal.shift(1), t.short_bal.shift(1)
    t["mktcap"], t["limit_flag"] = p["市值(百萬元)"], p["漲跌停"]
    t["inst_share"], t["credit_share"], t["daytrade_share"] = p["法人成交比重"], p["信用交易比重"], p["資券互抵比例"]
    t["margin_use"] = p["融資使用率"]
    w = t.loc["2015-01-01":os.environ.get("END", "2024-12-31")].reset_index().rename(columns={"date": "Date"})
    if w.empty:
        continue
    w["Date"] = w.Date.dt.strftime("%Y-%m-%d")
    w.insert(0, "stock_code", code)
    w.insert(1, "stock_name", p["name"].iloc[-1])
    frames.append(w)
out = pd.concat(frames, ignore_index=True)
vol = out.Volume.replace(0, np.nan)
for c in ("foreign", "trust", "dealer", "inst"):
    out[f"{c}_ratio"] = (out[f"{c}_net"] / vol).fillna(0.0)
out["margin_chg"] = (out.margin_bal - out.margin_bal_prev) / out.margin_bal_prev.replace(0, np.nan)
out["short_chg"] = (out.short_bal - out.short_bal_prev) / out.short_bal_prev.replace(0, np.nan)
out["short_margin_ratio"] = out.short_bal / out.margin_bal.replace(0, np.nan)
for c in [c for c in ref.columns if c.startswith("ptt_")]:
    out[c] = 0
out["in265"] = out.stock_code.isin(ref.stock_code.unique()).astype(int)

# 可否信用交易：TEJ 沒有直接的旗標，用原資料集（交易所融資融券表）對照挑定義
RC = ["margin_eligible", "margin_chg", "short_chg", "short_margin_ratio", "foreign_ratio", "Close"]
m = out.merge(ref[["stock_code", "Date"] + RC].rename(columns={c: c + "_ref" for c in RC}), on=["stock_code", "Date"], how="inner")
cands = {"融資餘額有值": m.margin_bal.notna(), "融資或融券餘額>0": (m.margin_bal.fillna(0) > 0) | (m.short_bal.fillna(0) > 0),
         "融資使用率有值": m.margin_use.notna(), "融資餘額>0": m.margin_bal.fillna(0) > 0}
say(f"與原 265 檔資料集重疊 {len(m):,} 列；可信用交易旗標各候選定義與原旗標一致率：")
best, bestv = None, -1
for k, v in cands.items():
    a = (v.astype(int) == m.margin_eligible_ref).mean()
    say(f"  {k}: {a:.4%}（候選為 1 的比例 {v.mean():.2%}；原 {m.margin_eligible_ref.mean():.2%}）")
    if a > bestv:
        best, bestv = k, a
say(f"→ 採用「{best}」")
out["margin_eligible"] = {"融資餘額有值": out.margin_bal.notna(), "融資或融券餘額>0": (out.margin_bal.fillna(0) > 0) | (out.short_bal.fillna(0) > 0),
                          "融資使用率有值": out.margin_use.notna(), "融資餘額>0": out.margin_bal.fillna(0) > 0}[best].astype(int)
for c in ("margin_chg", "short_chg", "short_margin_ratio", "foreign_ratio", "Close"):
    ok = m[c].notna() & m[c + "_ref"].notna() & np.isfinite(m[c]) & np.isfinite(m[c + "_ref"])
    say(f"  {c}: 與原資料集相關 {m.loc[ok, c].corr(m.loc[ok, c + '_ref']):.5f}（{int(ok.sum()):,} 列；差異中位數 {(m.loc[ok, c] - m.loc[ok, c + '_ref']).abs().median():.2e}）")

out = out.drop(columns=["margin_use"]).sort_values(["stock_code", "Date"]).reset_index(drop=True)
yr = out.Date.str[:4]
say(f"全市場資料集 {len(out):,} 列、{out.stock_code.nunique()} 檔（其中原 265 檔內 {out[out.in265 == 1].stock_code.nunique()} 檔）；可信用交易 {out.margin_eligible.mean():.1%}")
say("每年檔數：" + " ".join(f"{y}:{n}" for y, n in out.groupby(yr).stock_code.nunique().items()))
last = out.groupby("stock_code").Date.max()
say(f"2024-12 以前就沒有資料的（下市等）{int((last < '2024-12-01').sum())} 檔")
os.makedirs(OUTDIR, exist_ok=True)
out.to_csv(OUTDIR + "/all_stocks_v3.csv", index=False, encoding="utf-8-sig")
open(OUTDIR + "_build.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
