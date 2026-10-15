# -*- coding: utf-8 -*-
"""
市場散戶「細分情緒」每日指數（盤中/盤後閒聊，emotion/chat_emotion.csv，每日150則，Breeze 八類機率）
============================================================
指數（交易日 t：前一交易日收盤後 ~ t 收盤前的留言；各類機率的當日平均）：
  恐慌 p_慌、貪婪 p_貪、FOMO p_追、絕望 p_絕、反串 p_諷、看多 p_多、看空 p_空
  貪婪恐慌差 = (貪+追+多) − (慌+絕+空)
(1) 合理性：與當日/過去5日大盤報酬的相關；大事件期間各指數的 z 分數
(2) 預測：未來加權指數 1/5/20 日報酬 ~ 指數(5日平均, 擴張視窗z) + 過去5/20日報酬，Newey-West
(3) 極端值：指數最高 10% 的日子 vs 其他日子，之後 5/20 日大盤平均報酬
輸出：emotion_market_analysis.log、emotion/daily_emotion_index.csv
python emotion_market_analysis.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

em = pd.read_csv("emotion/chat_emotion.csv", parse_dates=["Date"])
P = {"恐慌": "p_慌", "貪婪": "p_貪", "FOMO": "p_追", "絕望": "p_絕", "反串": "p_諷", "看多": "p_多", "看空": "p_空"}
day = em.groupby("Date")[list(P.values())].mean()
day.columns = list(P.keys())
day["貪婪恐慌差"] = (day.貪婪 + day.FOMO + day.看多) - (day.恐慌 + day.絕望 + day.看空)
tx = pd.read_csv("market_sentiment/taiex_official.csv", parse_dates=["Date"]).set_index("Date")
df = tx.join(day, how="inner")
lr = np.log(df.taiex)
df["r0"] = lr - lr.shift(1)
df["past5"], df["past20"] = lr - lr.shift(5), lr - lr.shift(20)
for h in (1, 5, 20):
    df[f"f{h}"] = lr.shift(-h) - lr
IDX = list(P.keys()) + ["貪婪恐慌差"]
for c in IDX:
    s = df[c].rolling(5, min_periods=3).mean()
    df[c + "_z"] = (s - s.expanding(250).mean()) / s.expanding(250).std()
    df[c + "_raw_z"] = (df[c] - df[c].mean()) / df[c].std()
df.to_csv("emotion/daily_emotion_index.csv", encoding="utf-8-sig")
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


log(f"交易日 {len(df):,}（{df.index.min().date()}~{df.index.max().date()}）")
log("\n==== (1) 合理性：與當天大盤報酬 / 過去5日報酬 的相關 ====")
log("  " + "｜".join(f"{c} {df[c].corr(df.r0):+.2f}/{df[c].corr(df.past5):+.2f}" for c in IDX))
EVENTS = {"2015-08 股災": ("2015-08-17", "2015-08-26"), "2018-10 十月股災": ("2018-10-08", "2018-10-26"),
          "2020-03 疫情崩盤": ("2020-03-09", "2020-03-20"), "2021-05 本土疫情大跌": ("2021-05-10", "2021-05-18"),
          "2021-06~07 航海王多頭": ("2021-06-01", "2021-07-09"), "2022 空頭(9~10月)": ("2022-09-01", "2022-10-25"),
          "2024-06~07 AI 多頭高點": ("2024-06-17", "2024-07-11"), "2024-08-05 大跌": ("2024-08-02", "2024-08-07")}
log("\n==== 大事件期間的指數（全期 z 分數；>0 代表高於平常）====")
log("  " + " ".join(f"{c:>6s}" for c in IDX))
for nm, (a, b) in EVENTS.items():
    w = df.loc[a:b]
    log(f"  {nm:18s} " + " ".join(f"{w[c + '_raw_z'].mean():+6.2f}" for c in IDX) + f"｜期間大盤 {np.exp(w.r0.sum())-1:+.1%}")
log("\n==== (2) 預測未來大盤：係數＝指數每1σ（5日平均、只用過去資料標準化），控制過去5/20日報酬，Newey-West ====")
for c in IDX:
    out = []
    for h in (1, 5, 20):
        d = df[[f"f{h}", c + "_z", "past5", "past20"]].dropna()
        m = sm.OLS(d[f"f{h}"], sm.add_constant(d[[c + "_z", "past5", "past20"]])).fit(cov_type="HAC", cov_kwds={"maxlags": max(h, 5)})
        out.append(f"{h}日 {m.params[c + '_z']:+.2%}(t={m.tvalues[c + '_z']:+.2f})")
    log(f"  {c:6s} " + "｜".join(out))
log("\n==== (3) 極端值：指數最高10%的日子 vs 其他日子，之後大盤平均報酬 ====")
for c in IDX:
    d = df[[c + "_z", "f5", "f20"]].dropna()
    hi = d[c + "_z"] >= d[c + "_z"].quantile(0.9)
    lo = d[c + "_z"] <= d[c + "_z"].quantile(0.1)
    log(f"  {c:6s} 最高10%：5日 {d.f5[hi].mean():+.2%}、20日 {d.f20[hi].mean():+.2%}｜最低10%：5日 {d.f5[lo].mean():+.2%}、20日 {d.f20[lo].mean():+.2%}"
        f"｜其他：5日 {d.f5[~hi & ~lo].mean():+.2%}、20日 {d.f20[~hi & ~lo].mean():+.2%}")
open("emotion_market_analysis.log", "w", encoding="utf-8").write("\n".join(lines) + "\n")

log("\n==== (4) 極端值檢定：未來報酬 ~ 極端日虛擬變數 + 過去5/20日報酬，Newey-West（考慮重疊期間）====")
for c, side in (("貪婪恐慌差", "lo"), ("貪婪恐慌差", "hi"), ("恐慌", "hi"), ("FOMO", "hi"), ("FOMO", "lo"), ("絕望", "hi"), ("反串", "hi")):
    d = df[[c + "_z", "f5", "f20", "past5", "past20"]].dropna().copy()
    q = d[c + "_z"].quantile(0.1 if side == "lo" else 0.9)
    d["ext"] = (d[c + "_z"] <= q) if side == "lo" else (d[c + "_z"] >= q)
    d["ext"] = d["ext"].astype(float)
    res = []
    for h in (5, 20):
        m = sm.OLS(d[f"f{h}"], sm.add_constant(d[["ext", "past5", "past20"]])).fit(cov_type="HAC", cov_kwds={"maxlags": h})
        res.append(f"{h}日 {m.params['ext']:+.2%}(t={m.tvalues['ext']:+.2f})")
    log(f"  {c}{'最低' if side == 'lo' else '最高'}10% 相對其他日子：" + "｜".join(res))
log(f"\n（多重檢定提醒：本檔共做了約 {len(IDX)*3 + 14} 個檢定，5% 水準下純巧合預期約 {0.05*(len(IDX)*3+14):.1f} 個顯著）")
open("emotion_market_analysis.log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
