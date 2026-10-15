# -*- coding: utf-8 -*-
"""專案資料總檢查：筆數、日期範圍、缺漏、重複、異常值、資料集之間的一致性。
python data_audit.py > data_audit_report.txt"""
import os
import glob
import json
import numpy as np
import pandas as pd

from stock_list_267 import STOCK_LIST_267

S, E = "2020-01-01", "2024-12-31"
NAME = {c: n for c, n, _ in STOCK_LIST_267}


def hdr(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}", flush=True)


# ---------- 交易日曆：期交所PCR日期(含2021-04-06) ----------
import io
pcr = pd.concat([pd.read_csv(io.StringIO(open(f, encoding="utf-8").read()), index_col=False).iloc[:, :7]
                 for f in sorted(glob.glob("chip_raw/pcr/*.csv"))])
pcr.columns = ["date", "put_vol", "call_vol", "pc_vol_ratio", "put_oi", "call_oi", "pc_oi_ratio"]
CAL = sorted(set(pd.to_datetime(pcr.date).dt.strftime("%Y-%m-%d")))

# ---------- 1. 原始股價 ----------
hdr("1. 原始股價 stocks_267/stocks/{code}/price")
issues = []
price = {}
for code in NAME:
    fs = sorted(glob.glob(f"stocks_267/stocks/{code}/price/*.csv"))
    if not fs:
        issues.append((code, "無股價檔"))
        continue
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    dup = d.duplicated("date").sum()
    d = d.drop_duplicates("date").sort_values("date")
    w = d[(d.date >= S) & (d.date <= E)].copy()
    price[code] = w
    rec = {"code": code, "dup": dup, "n": len(w)}
    if len(w) == 0:
        rec["note"] = "2020-2024無資料"
        issues.append(rec)
        continue
    rec["first"], rec["last"] = w.date.min(), w.date.max()
    cal_in = [x for x in CAL if rec["first"] <= x <= rec["last"]]
    rec["missing_days"] = len(set(cal_in) - set(w.date))
    rec["extra_days"] = len(set(w.date) - set(CAL))
    rec["nan"] = int(w[["open", "high", "low", "close", "volume"]].isna().sum().sum())
    rec["nonpos"] = int((w[["open", "high", "low", "close"]] <= 0).sum().sum())
    tol = 1e-6
    rec["ohlc_bad"] = int(((w.low > w[["open", "close"]].min(axis=1) + tol) |
                           (w.high < w[["open", "close"]].max(axis=1) - tol)).sum())
    rec["zero_vol"] = int((w.volume == 0).sum())
    r = w.close.pct_change()
    rec["ret_gt11"] = int((r.abs() > 0.11).sum())
    rec["max_abs_ret"] = float(r.abs().max()) if len(r) > 1 else 0
    issues.append(rec)
P = pd.DataFrame([x for x in issues if isinstance(x, dict)])
print(f"267檔中有股價檔: {len(price)}；2020-2024有資料: {(P.n > 0).sum()}")
print(f"2020-2024 總股票日: {int(P.n.sum())}")
print(f"首日晚於2020-01-02(期間內才上市/才有資料): {(P['first'] > '2020-01-02').sum()} 檔")
print(f"末日早於2024-12-31(期間內下市/停止): {(P['last'] < '2024-12-31').sum()} 檔 -> {P.loc[P['last'] < '2024-12-31', 'code'].tolist()}")
for col, desc in [("dup", "重複日期(已去重前)"), ("missing_days", "上市期間內缺交易日"), ("extra_days", "非交易日卻有資料"),
                  ("nan", "NaN值"), ("nonpos", "價格<=0"), ("ohlc_bad", "OHLC矛盾(高<開收 或 低>開收)"),
                  ("zero_vol", "成交量=0天數"), ("ret_gt11", "單日漲跌>11%(超過漲跌停,疑除權/減資/資料錯)")]:
    s = P[col].fillna(0)
    print(f"  {desc}: 有問題 {int((s > 0).sum())} 檔，合計 {int(s.sum())}", end="")
    top = P.loc[s > 0].sort_values(col, ascending=False).head(5)
    print("  例:", [(c, NAME[c], int(v)) for c, v in zip(top.code, top[col])] if len(top) else "")
big = P.sort_values("max_abs_ret", ascending=False).head(8)
print("  單日最大漲跌幅前8:", [(c, NAME[c], f"{v:.1%}") for c, v in zip(big.code, big.max_abs_ret)])
P.to_csv("data_audit_price_detail.csv", index=False, encoding="utf-8-sig")

# ---------- 2. 大盤 ----------
hdr("2. 大盤 taiex_2020_2024.csv (yfinance)")
tx = pd.read_csv("taiex_2020_2024.csv", header=[0, 1], index_col=0)
td = set(pd.to_datetime(tx.index).strftime("%Y-%m-%d"))
print(f"筆數 {len(tx)}；缺交易日: {sorted(set(CAL) - td)}；多出: {sorted(td - set(CAL))}")
print(f"NaN: {int(tx.isna().sum().sum())}；成交量=0天數: {int((tx.xs('Volume', axis=1, level=0).iloc[:, 0] == 0).sum())}")

# ---------- 3. 技術指標 ----------
hdr("3. 技術指標 technical_output_267")
tech_issue = []
for code, name in NAME.items():
    fs = glob.glob(f"technical_output_267/{code}_*_technical.csv")
    if not fs:
        tech_issue.append((code, "無檔"))
        continue
    t = pd.read_csv(fs[0])
    t["Date"] = pd.to_datetime(t["Date"]).dt.strftime("%Y-%m-%d")
    w = t[(t.Date >= S) & (t.Date <= E)]
    num = w.select_dtypes("number")
    n_nan = int(num.isna().sum().sum())
    n_inf = int(np.isinf(num).sum().sum())
    if code in price and len(price[code]):
        m = w.merge(price[code][["date", "close"]], left_on="Date", right_on="date", how="inner")
        diff = float((m.Close - m.close).abs().max()) if len(m) else 0
        rowdiff = len(w) - len(price[code])
    else:
        diff, rowdiff = 0, len(w)
    if n_nan or n_inf or diff > 1e-6 or rowdiff:
        tech_issue.append((code, name, f"NaN={n_nan}", f"inf={n_inf}", f"收盤差={diff:.4g}", f"列數差={rowdiff}"))
print(f"有問題的檔數: {len(tech_issue)}")
for x in tech_issue[:15]:
    print("  ", x)

# ---------- 4. PTT比對到267檔(2020-2024) ----------
hdr("4. PTT比對留言 ptt_output_267_stitched_2020_2024")
fr = []
for f in glob.glob("ptt_output_267_stitched_2020_2024/*_comments.csv"):
    fr.append(pd.read_csv(f, dtype={"stock_code": str}))
pc = pd.concat(fr, ignore_index=True)
print(f"檔數 {len(fr)}；留言 {len(pc)}；股票 {pc.stock_code.nunique()}；日期 {pc.resolved_date.min()}~{pc.resolved_date.max()}")
print(f"空白留言: {int((pc.content.fillna('').astype(str).str.strip() == '').sum())}；"
      f"完全重複列: {int(pc.duplicated().sum())}")
pc["wd"] = pd.to_datetime(pc.resolved_date).dt.dayofweek
print(f"週末(六日)的留言: {int((pc.wd >= 5).sum())} ({(pc.wd >= 5).mean():.1%})；"
      f"非交易日(含週末與國定假日)的留言: {int((~pc.resolved_date.isin(CAL)).sum())} ({(~pc.resolved_date.isin(CAL)).mean():.1%})")
no_ptt = sorted(set(NAME) - set(pc.stock_code))
print(f"267檔中完全沒有PTT留言的: {len(no_ptt)} 檔 -> {[(c, NAME[c]) for c in no_ptt]}")

# ---------- 5. 情緒分類輸出 ----------
hdr("5. 情緒分類 sentiment_output_267_stitched_2020_2024")
so = pd.concat([pd.read_csv(f, dtype={"stock_code": str}) for f in glob.glob("sentiment_output_267_stitched_2020_2024/*_sentiment.csv")])
print(f"文章列數 {len(so)}；留言合計 {int(so.total_comments.sum())}（比對檔 {len(pc)}，差 {int(so.total_comments.sum()) - len(pc)}）")
bad_sum = (so.positive_count + so.negative_count + so.neutral_count != so.total_comments).sum()
recompute = ((so.positive_count - so.negative_count) / so.total_comments).round(4)
print(f"正+負+中≠總數: {int(bad_sum)}；情緒因子超出[-1,1]: {int(((so.text_sentiment_factor < -1) | (so.text_sentiment_factor > 1)).sum())}；"
      f"情緒因子與重算不符: {int(((recompute - so.text_sentiment_factor).abs() > 1e-3).sum())}")
print(f"(股票,文章,日期)重複: {int(so.duplicated(['stock_code', 'article_id', 'resolved_date']).sum())}；"
      f"同一文章跨多個日期: {int(so.groupby(['stock_code', 'article_id']).resolved_date.nunique().gt(1).sum())}")

# ---------- 6. 最終合併資料集 ----------
hdr("6. 最終合併 final_dataset_267_stitched/all_stocks_final_dataset_267.csv")
fd = pd.read_csv("final_dataset_267_stitched/all_stocks_final_dataset_267.csv", dtype={"stock_code": str})
print(f"列數 {len(fd)}；股票 {fd.stock_code.nunique()}；日期 {fd.Date.min()}~{fd.Date.max()}")
print(f"(股票,日期)重複: {int(fd.duplicated(['stock_code', 'Date']).sum())}")
nan = fd.isna().sum()
print(f"有NaN的欄位: {nan[nan > 0].to_dict()}")
missing_stocks = sorted(set(NAME) - set(fd.stock_code))
print(f"267檔中不在最終資料集的: {[(c, NAME[c]) for c in missing_stocks]}")
in_final = int(fd.total_comments.sum())
print(f"最終資料集保留的留言數 {in_final}，情緒輸出總數 {int(so.total_comments.sum())} -> "
      f"合併時遺失 {int(so.total_comments.sum()) - in_final} ({1 - in_final / so.total_comments.sum():.1%})")
lost = so[~so.resolved_date.isin(CAL)]
print(f"  其中發在非交易日(週末/假日)而被丟掉: {int(lost.total_comments.sum())}")
sc = [c for c in fd.columns if c.endswith("_scaled")]
print(f"_scaled欄位 {len(sc)} 個，值域: min {fd[sc].min().min():.3f} max {fd[sc].max().max():.3f}（是以2011-2026全期間算的，含測試期資訊）")

# ---------- 7. 全市場每日情緒 ----------
hdr("7. 全市場情緒 market_daily_sentiment.csv")
md = pd.read_csv("market_daily_sentiment.csv")
print(f"天數 {len(md)}（含非交易日 {int((~md.resolved_date.isin(CAL)).sum())} 天）；"
      f"日期 {md.resolved_date.min()}~{md.resolved_date.max()}；留言合計 {int(md.total_comments.sum())}")

# ---------- 8. 標註資料 ----------
hdr("8. 標註資料 claude_all_labels.csv")
lb = pd.read_csv("claude_all_labels.csv")
print(f"筆數 {len(lb)}；分布 {lb.claude_label.value_counts().to_dict()}；"
      f"內容重複 {int(lb.content.duplicated().sum())}；空白 {int((lb.content.fillna('').astype(str).str.strip() == '').sum())}")

# ---------- 9. 籌碼資料(下載中) ----------
hdr("9. 籌碼資料 chip_raw")
p2 = pcr.copy(); p2["date"] = pd.to_datetime(p2.date)
print(f"PCR: {p2.date.nunique()} 天 {p2.date.min().date()}~{p2.date.max().date()}；NaN {int(p2.isna().sum().sum())}")
for k in ("t86", "margin"):
    fs = sorted(glob.glob(f"chip_raw/{k}/*.json"))
    stats, rows, cover = {}, [], []
    for f in fs:
        j = json.load(open(f, encoding="utf-8"))
        stats[j.get("stat")] = stats.get(j.get("stat"), 0) + 1
        if k == "t86":
            data = j.get("data") or []
        else:
            tb = [t for t in (j.get("tables") or []) if t.get("data") and len(t["data"]) > 10]
            data = tb[0]["data"] if tb else []
        rows.append(len(data))
        cover.append(len({str(r[0]).strip() for r in data} & set(NAME)))
    if fs:
        print(f"{k}: 已下載 {len(fs)}/{len(CAL)} 天 ({os.path.basename(fs[0])[:8]}~{os.path.basename(fs[-1])[:8]})；"
              f"stat分布 {stats}；每日列數 {min(rows)}~{max(rows)}；267檔覆蓋 {min(cover)}~{max(cover)} 檔")
    else:
        print(f"{k}: 尚未下載")

# ---------- 10. CMoney(已知問題) ----------
hdr("10. CMoney（已知問題，僅摘要）")
cov = pd.read_csv("cmoney_coverage_windows.csv")
print(f"有2020-2024文章的股票: {len(cov)}/267；最晚到2024-12-31的: {(cov['max'] == '2024-12-31').sum()}；"
      f"最早在2020上半年的: {(cov['min'] <= '2020-06-30').sum()}")
print("已知：留言日期被套用文章日期(17.1%錯置，最長約4.7年)；留言爬到2026-09，未套用日期篩選")

# ---------- 11. 舊版資料(已被取代) ----------
hdr("11. 舊版資料（已被新版取代，僅列出）")
for d, why in [("ptt_output_267", "組員提供的PTT比對，舊版"), ("ptt_output_267_stitched", "拼接版2022-2024，已擴大為2020-2024"),
               ("sentiment_output_267_2022_2024", "舊分類器(有缺陷)"), ("final_dataset_267", "舊分類器+2022-2024")]:
    print(f"  {d}: {why}")
print("\n[完成]")
