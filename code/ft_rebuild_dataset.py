# -*- coding: utf-8 -*-
"""由 sentiment_output_267_ft/ 的逐則標籤重建 final_dataset_v3_ft/all_stocks_v3.csv（只換 ptt_* 欄位）。
先用其中的 breeze 欄重算，確認與原資料集一致（流程驗證），再換成微調標籤。 python ft_rebuild_dataset.py"""
import glob, io, os, sys
import numpy as np, pandas as pd
parts = [pd.read_csv(f, dtype=str, keep_default_na=False, usecols=["stock_code", "resolved_date", "comment_time", "time_anomaly", "text_sentiment", "breeze"])
         for f in glob.glob("sentiment_output_267_ft/*_comment_labels.csv")]
cm = pd.concat(parts, ignore_index=True)
print("留言", len(cm))
cal1 = pd.to_datetime(pd.read_csv("chip_raw/trading_days_2014_2019.csv", dtype=str)["date"], format="%Y%m%d")
pcr = pd.concat([pd.read_csv(io.StringIO(open(f, encoding="utf-8").read()), index_col=False).iloc[:, :7] for f in sorted(glob.glob("chip_raw/pcr/*.csv"))])
pcr.columns = ["Date", "put_vol", "call_vol", "pc_vol_ratio", "put_oi", "call_oi", "pc_oi_ratio"]
pcr["Date"] = pd.to_datetime(pcr.Date)
cal2 = pcr.drop_duplicates("Date").sort_values("Date").loc[lambda x: x.Date >= "2020-01-01", "Date"]
CAL = pd.DatetimeIndex(sorted(set(cal1) | set(cal2)))
CAL = CAL[CAL <= "2024-12-31"]  # 原資料集建立時日曆只到 2024 年底（之後才下載 2025 的 PCR）
CAL_S = CAL.strftime("%Y-%m-%d")
close_dt = CAL.values + np.timedelta64(13 * 3600 + 30 * 60, "s"); open_dt = CAL.values + np.timedelta64(9 * 3600, "s")
def build_ptt(c, col):
    c = c[(c.comment_time != "") & (c.time_anomaly != "1")].copy()
    c["pos"] = (c[col] == "positive").astype(int); c["neg"] = (c[col] == "negative").astype(int)
    ct = pd.to_datetime(c.comment_time).values
    i = np.searchsorted(close_dt, ct, side="left"); ok = i < len(CAL)
    c, ct, i = c[ok].copy(), ct[ok], i[ok]
    c["Date"] = CAL_S[i]; c["part"] = np.where(ct < open_dt[i], "pre", "intra")
    j = np.searchsorted(CAL.values, pd.to_datetime(c.resolved_date).values, side="left")
    c["Date_art"] = np.where(j < len(CAL), CAL_S[np.minimum(j, len(CAL) - 1)], "")
    def agg(x, key, prefix):
        a = x.groupby(["stock_code", key]).agg(n=("pos", "size"), pos=("pos", "sum"), neg=("neg", "sum")).reset_index()
        return a.rename(columns={key: "Date", "n": f"{prefix}_n", "pos": f"{prefix}_pos", "neg": f"{prefix}_neg"})
    return agg(c[c.part == "pre"], "Date", "ptt_pre").merge(agg(c[c.part == "intra"], "Date", "ptt_intra"), on=["stock_code", "Date"], how="outer").merge(
        agg(c[c.Date_art != ""], "Date_art", "ptt_art"), on=["stock_code", "Date"], how="outer")
base = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str})
PTT = [c for c in base.columns if c.startswith("ptt_")]
def swap(col):
    new = base.drop(columns=PTT).merge(build_ptt(cm, col), on=["stock_code", "Date"], how="left")
    assert len(new) == len(base)
    new[PTT] = new[PTT].fillna(0).astype(int)
    return new[base.columns]
chk = swap("breeze")
ok = True
for c in PTT:
    d = (chk[c].values != base[c].fillna(0).astype(int).values)
    print(f"  {c}: 不同 {d.sum()} 列｜原總和 {int(base[c].sum()):,}、重算 {int(chk[c].sum()):,}")
    ok &= d.sum() == 0
print("用 Breeze 標籤重算與原資料集完全相同：", ok)
if not ok and "--force" not in sys.argv:
    bad = (chk[PTT].values != base[PTT].fillna(0).astype(int).values).any(1)
    print(base.loc[bad, ["stock_code", "Date"] + PTT].head(8).to_string()); print(chk.loc[bad, ["stock_code", "Date"] + PTT].head(8).to_string())
    print(base.loc[bad].Date.str[:4].value_counts().sort_index().to_dict(), base.loc[bad].stock_code.nunique())
    sys.exit(1)
new = swap("text_sentiment")
os.makedirs("final_dataset_v3_ft", exist_ok=True)
new.to_csv("final_dataset_v3_ft/all_stocks_v3.csv", index=False, encoding="utf-8-sig")
for c in ("pre", "intra"):
    print(f"ptt_{c}：正 {int(base[f'ptt_{c}_pos'].sum()):,} → {int(new[f'ptt_{c}_pos'].sum()):,}｜負 {int(base[f'ptt_{c}_neg'].sum()):,} → {int(new[f'ptt_{c}_neg'].sum()):,}｜則數 {int(new[f'ptt_{c}_n'].sum()):,}")
