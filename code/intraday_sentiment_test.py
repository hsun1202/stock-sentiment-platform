# -*- coding: utf-8 -*-
"""
盤中即時散戶情緒（仿 Renault 2017, JBF）：開盤前半小時的情緒變化，能不能預測收盤前半小時的大盤報酬？
============================================================
資料：盤中/盤後閒聊留言（market_sentiment/chat_labels.csv，每日抽樣≤300則已分類）＋證交所每分鐘加權指數（taiex_intraday_1min.csv）
每個交易日 d：
  S_night  ＝ 前一交易日 13:30 ~ d 09:00 的留言淨情緒 (正−負)/(正+負)
  S_first  ＝ d 09:00 ~ 09:30 的留言淨情緒
  ΔS       ＝ S_first − S_night（Renault：前半小時的情緒「變化」）
  att_first＝ 09:00~09:30 全部留言數（不抽樣）相對過去20日平均的 log 比（異常熱度）
報酬（log）：r_first ＝ 前一日收盤 → d 09:30（含隔夜跳空，同 Gao et al. 2018 的第一個半小時）
            r_mid   ＝ 12:30 → 13:00（倒數第二個半小時）
            r_last  ＝ 13:00 → 13:30（收盤前半小時，預測目標）
            r_rest  ＝ 09:30 → 13:30（開盤半小時後到收盤，另一個預測目標）
迴歸：r_last（或 r_rest）~ ΔS(z) + r_first + r_mid（+ S_night），Newey-West(5)；另檢驗 |r_last| ~ att_first（波動）
輸出：intraday_sentiment_test.log
python intraday_sentiment_test.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

D = "market_sentiment"
ix = pd.read_csv(f"{D}/taiex_intraday_1min.csv", dtype={"Date": str, "time": str})
piv = ix.pivot_table(index="Date", columns="time", values="taiex")
piv.index = pd.to_datetime(piv.index)
need = ["09:00:00", "09:30:00", "12:30:00", "13:00:00", "13:30:00"]
piv = piv[[c for c in need if c in piv.columns]].dropna()
tx = pd.read_csv(f"{D}/taiex_official.csv", parse_dates=["Date"]).set_index("Date")
df = piv.join(tx.taiex.rename("close_off"), how="inner")
df["prev_close"] = df.close_off.shift(1)
df["r_first"] = np.log(df["09:30:00"] / df.prev_close)
df["r_mid"] = np.log(df["13:00:00"] / df["12:30:00"])
df["r_last"] = np.log(df["13:30:00"] / df["13:00:00"])
df["r_rest"] = np.log(df["13:30:00"] / df["09:30:00"])

lab = pd.read_csv(f"{D}/chat_labels.csv", dtype=str, keep_default_na=False)
t = pd.to_datetime(lab.comment_time)
lab["s"] = np.select([lab.text_sentiment == "positive", lab.text_sentiment == "negative"], [1, -1], 0)
day = t.dt.normalize()
hm = t.dt.hour * 100 + t.dt.minute
CAL = df.index.values
# 09:00 前（含前一晚）→ 歸到「下一個 ≥ 該日」的交易日；13:30 後 → 歸到下一個交易日
nxt = np.searchsorted(CAL, (day + pd.to_timedelta(np.where(hm >= 1330, 1, 0), unit="D")).values, side="left")
lab["d"] = pd.NaT
okn = nxt < len(CAL)
lab.loc[okn, "d"] = CAL[nxt[okn]]
lab["win"] = np.where((hm >= 900) & (hm < 930) & (day.values == lab.d.values), "first",
                      np.where((hm >= 1330) | (hm < 900) | (day.values != lab.d.values), "night", "other"))
dir_ = lab[lab.s != 0]
agg = dir_.groupby(["d", "win"]).s.agg(["mean", "count"]).unstack()
df["S_first"] = agg[("mean", "first")]
df["S_night"] = agg[("mean", "night")]
df["n_first_lab"] = agg[("count", "first")]
import os  # noqa: E402
if os.path.exists(f"{D}/first30_labels.csv"):
    # 開盤半小時改用大樣本分類（intraday_classify_capped.py：每日≤150則，早期742天全量），降低 S_first 的抽樣雜訊
    f30 = pd.read_csv(f"{D}/first30_labels.csv", dtype=str, keep_default_na=False)
    f30["s"] = np.select([f30.text_sentiment == "positive", f30.text_sentiment == "negative"], [1, -1], 0)
    f30["d"] = pd.to_datetime(f30.comment_time.str[:10])
    g30 = f30[f30.s != 0].groupby("d").s.agg(["mean", "count"])
    df["S_first"] = g30["mean"].reindex(df.index)
    df["n_first_lab"] = g30["count"].reindex(df.index)
    print("（開盤半小時情緒改用 first30_labels.csv 大樣本）", flush=True)
df["dS"] = df.S_first - df.S_night

# 開盤熱度（全部留言，不抽樣）
vol = pd.read_csv(f"{D}/chat_daily_volume.csv", parse_dates=["Date"])  # 只有每日總量；開盤半小時數量另外算
arts = pd.read_csv("ptt_2015_2024/articles.csv", usecols=["article_id", "title"], dtype=str, keep_default_na=False)
chat = set(arts.loc[arts.title.str.contains(r"盤[中後]\s*閒聊", regex=True), "article_id"])
cnt = []
for ch in pd.read_csv("ptt_2015_2024/comments_clean.csv", chunksize=2_000_000, dtype=str, keep_default_na=False,
                      usecols=["article_id", "comment_time", "time_anomaly"]):
    ch = ch[ch.article_id.isin(chat) & (ch.comment_time != "") & (ch.time_anomaly != "1")]
    tt = pd.to_datetime(ch.comment_time)
    h2 = tt.dt.hour * 100 + tt.dt.minute
    cnt.append(tt[(h2 >= 900) & (h2 < 930)].dt.normalize().value_counts())
first_n = pd.concat(cnt).groupby(level=0).sum()
df["n_first"] = first_n.reindex(df.index).fillna(0)
ln = np.log1p(df.n_first)
df["att_first"] = ln - ln.shift(1).rolling(20, min_periods=10).mean()
df = df.loc["2015-02-01":"2024-12-31"]
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def z(s):
    return (s - s.mean()) / s.std()


log(f"交易日 {len(df):,}；開盤半小時已分類表態留言 中位數 {df.n_first_lab.median():.0f} 則/日；開盤半小時全部留言 中位數 {df.n_first.median():.0f} 則/日")
log(f"相關：S_first 與 r_first {df.S_first.corr(df.r_first):+.2f}（情緒反應開盤漲跌）；ΔS 與 r_first {df.dS.corr(df.r_first):+.2f}")
for y, nm in (("r_last", "收盤前半小時 13:00→13:30"), ("r_rest", "09:30→收盤")):
    log(f"\n==== 預測 {nm} ====")
    for xs, lab_ in ((["r_first", "r_mid"] if y == "r_last" else ["r_first"], "只有報酬（盤中動能基準）"),
                     (["dS_z", "r_first"] + (["r_mid"] if y == "r_last" else []), "+ΔS 開盤情緒變化"),
                     (["S_first_z", "S_night_z", "r_first"] + (["r_mid"] if y == "r_last" else []), "+開盤情緒/前晚情緒（分開）")):
        d = df.assign(dS_z=z(df.dS), S_first_z=z(df.S_first), S_night_z=z(df.S_night))[[y] + xs].dropna()
        m = sm.OLS(d[y], sm.add_constant(d[xs])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        log(f"  [{lab_}] n={len(d):,} R²={m.rsquared:.4f}｜" + "｜".join(f"{c} {m.params[c]*1e4:+.2f}bp(t={m.tvalues[c]:+.2f})" for c in xs))
log("\n==== 開盤熱度 → 收盤前波動 ====")
d = df[["r_last", "att_first", "r_first"]].dropna().assign(abs_last=lambda x: x.r_last.abs(), abs_first=lambda x: x.r_first.abs())
m = sm.OLS(d.abs_last, sm.add_constant(d[["att_first", "abs_first"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
log(f"  |r_last| ~ att_first + |r_first|：att_first {m.params['att_first']*1e4:+.2f}bp(t={m.tvalues['att_first']:+.2f})，|r_first| t={m.tvalues['abs_first']:+.2f}")
open("intraday_sentiment_test.log", "w", encoding="utf-8").write("\n".join(lines) + "\n")

# ---- 穩健性：分期間、只看留言夠多的日子 ----
log("\n==== 穩健性（預測收盤前半小時；ΔS 係數）====")
base = df.assign(dS_z=z(df.dS))
for nm, sel in (("2015-2019", base.index < "2020-01-01"), ("2020-2024", base.index >= "2020-01-01"),
                ("開盤表態留言≥20則的日子", base.n_first_lab >= 20), ("開盤表態留言≥30則的日子", base.n_first_lab >= 30)):
    d = base.loc[sel, ["r_last", "dS_z", "r_first", "r_mid"]].dropna()
    m = sm.OLS(d.r_last, sm.add_constant(d[["dS_z", "r_first", "r_mid"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    log(f"  [{nm}] n={len(d):,}：ΔS {m.params['dS_z']*1e4:+.2f}bp(t={m.tvalues['dS_z']:+.2f})")
q = pd.qcut(base.dS, 5, labels=[1, 2, 3, 4, 5])
log("  五分位（開盤情緒變化 1=轉悲觀 … 5=轉樂觀）收盤前半小時平均報酬(bp)：" +
    " ".join(f"{v*1e4:+.1f}" for v in base.groupby(q, observed=True).r_last.mean()))
log(f"  （參考：收盤前半小時報酬標準差 {df.r_last.std()*1e4:.1f}bp）")
open("intraday_sentiment_test.log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
