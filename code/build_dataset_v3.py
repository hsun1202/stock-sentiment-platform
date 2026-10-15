# -*- coding: utf-8 -*-
"""
建立 v3 資料集：2015-01 ~ 2024-12 十年（v2 只有 2020-2024；v2 的程式與輸出保留不動）
============================================================
  1. 技術面：還原股價 prices_adj_2014/（fix_prices_2015.py，含Yahoo缺漏日修補、刪颱風假資料）重算16項指標
     （2014-06 起算當暖機，不做全期間縮放）
  2. 異常價格日：單日|報酬| 超過「漲跌幅限制+0.5%」且非上市前5個交易日
     漲跌幅限制 2015-06-01 以前 7%，之後 10%（v2 只有 10% 規則）
  3. PTT 情緒（sentiment_output_267_2015_2024/*_comment_labels.csv，新分類器、清理後、新比對規則）：
     依「留言自己的時間」分到交易日 t 的兩個時段（lag_timestamp_test.py 已證明舊的文章日期歸屬會偷看答案）：
        ptt_pre_*   ：前一交易日收盤(13:30)後 ~ t 開盤(09:00)前（含假日）
        ptt_intra_* ：t 的 09:00 ~ 13:30
        → 「t 收盤前可得」= pre + intra；「t+1 開盤前可得」= 再加上 t+1 的 pre
     另存舊做法 ptt_art_*（文章日期，非交易日歸下一交易日）只供對照，建模不要用
     各時段存 n(留言數)、pos、neg；time_anomaly 的留言排除
  4. 籌碼：chip_raw/t86（股）、chip_raw/margin（張），2015-2024；三大法人依「欄位名稱」解析
     （2017-12 以前 T86 只有 16 欄、外資沒拆外陸資/外資自營商；之後 19 欄，依位置讀會錯）
     PCR：chip_raw/pcr
輸出：technical_output_267_adj_2014/、final_dataset_v3/all_stocks_v3.csv、chip_daily_v3.csv
python build_dataset_v3.py
"""
import io
import os
import glob
import json
import numpy as np
import pandas as pd

from stock_list_267 import STOCK_LIST_267
from stock_indicators import calculate_technical_indicators

S, E = "2015-01-01", "2024-12-31"
TECH_DIR, OUT_DIR = "technical_output_267_adj_2014", "final_dataset_v3"
os.makedirs(TECH_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
CODES = {c for c, _, _ in STOCK_LIST_267}

# ---------- 交易日曆：2014-06~2019 證交所官方 FMTQIK；2020~ 沿用 v2（期交所PCR日期，含2021-04-06） ----------
cal1 = pd.to_datetime(pd.read_csv("chip_raw/trading_days_2014_2019.csv", dtype=str)["date"], format="%Y%m%d")
pcr = pd.concat([pd.read_csv(io.StringIO(open(f, encoding="utf-8").read()), index_col=False).iloc[:, :7]
                 for f in sorted(glob.glob("chip_raw/pcr/*.csv"))])
pcr.columns = ["Date", "put_vol", "call_vol", "pc_vol_ratio", "put_oi", "call_oi", "pc_oi_ratio"]
pcr["Date"] = pd.to_datetime(pcr.Date)
pcr = pcr.drop_duplicates("Date").sort_values("Date")
cal2 = pcr.loc[pcr.Date >= "2020-01-01", "Date"]
CAL = pd.DatetimeIndex(sorted(set(cal1) | set(cal2)))
CAL_S = CAL.strftime("%Y-%m-%d")
pcr["Date"] = pcr.Date.dt.strftime("%Y-%m-%d")
print(f"交易日曆 {CAL.min().date()}~{CAL.max().date()}，{len(CAL)} 天；2015-2024 {((CAL >= S) & (CAL <= E)).sum()} 天")

# ---------- PTT：依留言時間分到交易日的盤前/盤中時段 ----------
close_dt = CAL.values + np.timedelta64(13 * 3600 + 30 * 60, "s")
open_dt = CAL.values + np.timedelta64(9 * 3600, "s")
parts = []
for f in glob.glob("sentiment_output_267_2015_2024/*_comment_labels.csv"):
    parts.append(pd.read_csv(f, dtype=str, keep_default_na=False,
                             usecols=["stock_code", "resolved_date", "comment_time", "time_anomaly", "text_sentiment"]))
cm = pd.concat(parts, ignore_index=True)
n0 = len(cm)
cm = cm[(cm.comment_time != "") & (cm.time_anomaly != "1")].copy()
cm["pos"] = (cm.text_sentiment == "positive").astype(int)
cm["neg"] = (cm.text_sentiment == "negative").astype(int)
ct = pd.to_datetime(cm.comment_time).values
i = np.searchsorted(close_dt, ct, side="left")
ok = i < len(CAL)
cm, ct, i = cm[ok].copy(), ct[ok], i[ok]
cm["Date"] = CAL_S[i]
cm["part"] = np.where(ct < open_dt[i], "pre", "intra")
j = np.searchsorted(CAL.values, pd.to_datetime(cm.resolved_date).values, side="left")
cm["Date_art"] = np.where(j < len(CAL), CAL_S[np.minimum(j, len(CAL) - 1)], "")
print(f"PTT 留言 {n0:,} 則，有時間且非異常 {len(cm):,} 則；盤前 {(cm.part == 'pre').mean():.1%}、盤中 {(cm.part == 'intra').mean():.1%}")


def ptt_agg(x, key, prefix):
    a = x.groupby(["stock_code", key]).agg(n=("pos", "size"), pos=("pos", "sum"), neg=("neg", "sum")).reset_index()
    return a.rename(columns={key: "Date", "n": f"{prefix}_n", "pos": f"{prefix}_pos", "neg": f"{prefix}_neg"})


ptt = ptt_agg(cm[cm.part == "pre"], "Date", "ptt_pre").merge(
    ptt_agg(cm[cm.part == "intra"], "Date", "ptt_intra"), on=["stock_code", "Date"], how="outer").merge(
    ptt_agg(cm[cm.Date_art != ""], "Date_art", "ptt_art"), on=["stock_code", "Date"], how="outer")
PTT_COLS = [c for c in ptt.columns if c.startswith("ptt_")]


# ---------- 籌碼 ----------
def num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return np.nan


def t86_row_parser(fields):
    """依欄位名稱找位置。外資 = 所有以「外」開頭的買賣超欄位加總（新版拆成外陸資＋外資自營商，舊版只有外資一欄）。"""
    idx = {f: k for k, f in enumerate(fields)}
    foreign = [k for f, k in idx.items() if f.startswith("外") and "買賣超" in f]
    trust = idx["投信買賣超股數"]
    dealer = idx["自營商買賣超股數"]
    inst = idx["三大法人買賣超股數"]
    return lambda r: {"foreign_net": sum(num(r[k]) for k in foreign), "trust_net": num(r[trust]),
                      "dealer_net": num(r[dealer]), "inst_net": num(r[inst])}


chip_days = [d for d in CAL_S if S <= d <= E]
rows, missing, layouts = [], [], {}
for d in chip_days:
    f = f"chip_raw/t86/{d.replace('-', '')}.json"
    if not os.path.exists(f):
        missing.append(("t86", d))
        continue
    t = json.load(open(f, encoding="utf-8"))
    if not t.get("data"):
        missing.append(("t86_empty", d))
        continue
    parse = t86_row_parser(t["fields"])
    layouts[len(t["fields"])] = layouts.get(len(t["fields"]), 0) + 1
    for r in t["data"]:
        c = str(r[0]).strip()
        if c in CODES:
            rows.append({"stock_code": c, "Date": d, **parse(r), "_t86": 1})
t86 = pd.DataFrame(rows)
print(f"T86 欄位格式（欄數:天數）{layouts}")

rows = []
for d in chip_days:
    f = f"chip_raw/margin/{d.replace('-', '')}.json"
    if not os.path.exists(f):
        missing.append(("margin", d))
        continue
    j = json.load(open(f, encoding="utf-8"))
    tb = [t for t in (j.get("tables") or []) if t.get("data") and len(t["data"]) > 10]
    if not tb:
        missing.append(("margin_empty", d))
        continue
    fl = tb[0]["fields"]
    assert fl[:7] == ["代號", "名稱", "買進", "賣出", "現金償還", "前日餘額", "今日餘額"] and fl[8:10] == ["買進", "賣出"], (d, fl)
    for r in tb[0]["data"]:
        c = str(r[0]).strip()
        if c in CODES:
            rows.append({"stock_code": c, "Date": d, "margin_buy": num(r[2]), "margin_sell": num(r[3]),
                         "margin_bal_prev": num(r[5]), "margin_bal": num(r[6]), "short_buy": num(r[8]),
                         "short_sell": num(r[9]), "short_bal_prev": num(r[11]), "short_bal": num(r[12]),
                         "credit_note": str(r[15]).strip(), "_margin": 1})
mg = pd.DataFrame(rows)
print(f"籌碼缺檔/空檔：{len(missing)} 筆 {missing[:10]}")
chip = t86.merge(mg, on=["stock_code", "Date"], how="outer")
chip.to_csv("chip_daily_v3.csv", index=False, encoding="utf-8-sig")

# ---------- 逐檔：技術指標 + 異常標記 + 合併 ----------
frames, rep = [], []
for code, name, industry in STOCK_LIST_267:
    f = glob.glob(f"prices_adj_2014/{code}_*.csv")
    if not f:
        rep.append((code, name, "無還原股價"))
        continue
    p = pd.read_csv(f[0], dtype={"Date": str}).set_index("Date")
    p.index = pd.to_datetime(p.index)
    tech = calculate_technical_indicators(p[["Open", "High", "Low", "Close", "Volume"]])
    tech["close_unadj"] = p.close_unadj
    tech["ret1"] = tech.Close.pct_change()
    nth = np.arange(1, len(tech) + 1)
    limit = np.where(tech.index < "2015-06-01", 0.07, 0.10) + 0.005
    tech["abnormal_price"] = ((tech.ret1.abs() > limit) & (nth > 5)).astype(int)
    tech.index.name = "Date"
    tech.to_csv(os.path.join(TECH_DIR, f"{code}_{name}_technical.csv"), encoding="utf-8-sig")
    w = tech.loc[S:E].reset_index()
    if w.empty:
        rep.append((code, name, "2015-2024無資料"))
        continue
    w["Date"] = w.Date.dt.strftime("%Y-%m-%d")
    w.insert(0, "stock_code", code)
    w.insert(1, "stock_name", name)
    w.insert(2, "industry", industry)
    frames.append(w)
out = pd.concat(frames, ignore_index=True)
bad_days = sorted(set(out.Date) - set(CAL_S))
print(f"股價日期不在交易日曆裡的：{bad_days[:10]}（應為空）")

out = out.merge(ptt, on=["stock_code", "Date"], how="left")
out[PTT_COLS] = out[PTT_COLS].fillna(0).astype(int)
out = out.merge(chip, on=["stock_code", "Date"], how="left").merge(
    pcr[["Date", "pc_vol_ratio", "pc_oi_ratio"]], on="Date", how="left")
for c in ("foreign_net", "trust_net", "dealer_net", "inst_net"):
    out[c] = out[c].fillna(0.0)
out["_t86"] = out["_t86"].fillna(0).astype(int)
out["margin_eligible"] = out["_margin"].fillna(0).astype(int)
vol = out.Volume.replace(0, np.nan)
for c in ("foreign", "trust", "dealer", "inst"):
    out[f"{c}_ratio"] = (out[f"{c}_net"] / vol).fillna(0.0)
out["margin_chg"] = (out.margin_bal - out.margin_bal_prev) / out.margin_bal_prev.replace(0, np.nan)
out["short_chg"] = (out.short_bal - out.short_bal_prev) / out.short_bal_prev.replace(0, np.nan)
out["short_margin_ratio"] = out.short_bal / out.margin_bal.replace(0, np.nan)
out = out.drop(columns=["_margin"])
out.to_csv(os.path.join(OUT_DIR, "all_stocks_v3.csv"), index=False, encoding="utf-8-sig")

# ---------- 檢查 ----------
print(f"\n股票 {out.stock_code.nunique()} 檔，列數 {len(out):,}，日期 {out.Date.min()}~{out.Date.max()}")
print(f"(股票,日期)重複 {int(out.duplicated(['stock_code', 'Date']).sum())}；異常價格日 {int(out.abnormal_price.sum())} 列")
yr = out.Date.str[:4]
traded = out.Volume > 0
print("各年：列數 / PTT留言(盤前+盤中) / 有法人紀錄比例(有成交日) / 可信用交易比例 / PCR缺值")
for y in sorted(yr.unique()):
    m = yr == y
    print(f"  {y}: {m.sum():,} / {int((out.ptt_pre_n + out.ptt_intra_n)[m].sum()):,} / "
          f"{out.loc[m & traded, '_t86'].mean():.1%} / {out.loc[m & traded, 'margin_eligible'].mean():.1%} / "
          f"{int(out.pc_vol_ratio[m].isna().sum())}")
mm = out[out.margin_eligible == 1].sort_values(["stock_code", "Date"])
prev_ok = mm.groupby("stock_code").margin_bal.shift(1)
mism = ((mm.margin_bal_prev - prev_ok).abs() > 0.5) & prev_ok.notna()
print(f"融資『前日餘額』≠前一交易日『今日餘額』: {int(mism.sum())} / {int(prev_ok.notna().sum())}")
print("略過:", rep)
