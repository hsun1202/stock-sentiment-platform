# -*- coding: utf-8 -*-
"""
情緒當「開關」：市場情緒高或低的時候，技術面訊號的準確度一不一樣？（真正的「整合技術面與情緒」：情緒不是加進去的變數，而是決定技術面可不可信的狀態）
文獻：Stambaugh, Yu & Yuan (2012) 異象在高情緒之後較強；Antoniou, Doukas & Subrahmanyam (2013) 動能只在樂觀期有效。
資料：daytrade_preds/B_{Y}.pkl（技術面 XGBoost 對隔天開盤→收盤上漲機率的樣本外預測，2018~2024，265 檔）；
      emotion/daily_emotion_index.csv（PTT 閒聊串每日細分情緒：看多、看空、恐慌…）；final_dataset_v3（融資餘額、報酬）。
狀態變數（都只用 t 日收盤前已知的資訊；用「過去 250 個交易日」的分位數分高/中/低三組，不偷看未來）：
  S1 說的-多空：過去 5 天平均（看多 − 看空）
  S2 說的-恐慌：過去 5 天平均 恐慌
  S3 做的-融資：265 檔融資餘額合計的過去 20 天變化率
技術面訊號好壞（t+1 日）：每日橫斷面 Rank IC（預測機率 vs 隔天開→收報酬），以及預測最高 20% − 最低 20% 的報酬差。

事先寫定的假說（2026-10-09，跑之前寫下；主要檢定 3 個，Bonferroni 門檻 p<0.0167，雙尾）：
  R1／R2／R3：以 S1／S2／S3 分組，高組日子與低組日子的「技術面每日 Rank IC」平均不同（Welch t，逐日）。
  探索性：高−低報酬差的差異、月頻三個選股訊號（低波動、避開暴漲股 MAX、券資比）在高/低情緒月份之後的 IC（月數少，只當參考）、分前後期。
python sentiment_regime_test.py
"""
import pickle

import numpy as np
import pandas as pd
from scipy import stats

LOG = []
ALPHA = 0.05 / 3


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


raw = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                  usecols=["stock_code", "Date", "Open", "Close", "close_unadj", "Volume", "abnormal_price", "margin_bal", "short_margin_ratio"])
raw = raw.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = raw.groupby("stock_code")
raw["r_next"] = (g.Close.shift(-1) / g.Open.shift(-1) - 1) * 1e4
rows = []
for Y in range(2018, 2025):
    o = pickle.load(open(f"daytrade_preds/B_{Y}.pkl", "rb"))
    rows.append(pd.DataFrame({"i": o["idx"], "p": o["P"]["技術面"]}))
pr = pd.concat(rows)
pr["Date"] = raw.Date.values[pr.i]
pr["r"] = raw.r_next.values[pr.i]
pr = pr.dropna(subset=["r"])
pr["rp"], pr["rr"] = pr.groupby("Date").p.rank(), pr.groupby("Date").r.rank()
ic = pr.groupby("Date").apply(lambda z: z.rp.corr(z.rr))
q = pr.groupby("Date").p.rank(pct=True)
ls = pr[q > 0.8].groupby("Date").r.mean() - pr[q <= 0.2].groupby("Date").r.mean()
D = pd.DataFrame({"ic": ic, "ls": ls})
say(f"[技術面訊號] {len(D):,} 個交易日（{D.index.min().date()}~{D.index.max().date()}）；每日 Rank IC 平均 {D.ic.mean():.4f}；最高20%−最低20% 平均 {D.ls.mean():+.1f} bp/天")

em = pd.read_csv("emotion/daily_emotion_index.csv", encoding="utf-8-sig", parse_dates=["Date"]).set_index("Date").sort_index()
mb = raw.groupby("Date").margin_bal.sum()
S = pd.DataFrame(index=sorted(raw.Date.unique()))
S["S1 說的-多空(看多−看空)"] = (em["看多"] - em["看空"]).reindex(S.index).rolling(5, min_periods=3).mean()
S["S2 說的-恐慌"] = em["恐慌"].reindex(S.index).rolling(5, min_periods=3).mean()
S["S3 做的-融資餘額20日變化"] = mb.reindex(S.index).pct_change(20)


def tercile(s):
    """用過去 250 天（不含今天）的分位數，把今天分成 0低/1中/2高"""
    lo = s.rolling(250, min_periods=120).quantile(1 / 3).shift(1)
    hi = s.rolling(250, min_periods=120).quantile(2 / 3).shift(1)
    return pd.Series(np.where(s > hi, 2, np.where(s < lo, 0, 1)), index=s.index).where(s.notna() & lo.notna())


say("\n===== 日頻：依 t 日的市場情緒狀態分組，看 t+1 日技術面訊號的好壞 =====")
res = {}
for name in S.columns:
    st = tercile(S[name]).reindex(D.index)
    d = D.assign(st=st).dropna(subset=["st", "ic"])
    grp = d.groupby("st")
    hi, lo = d[d.st == 2], d[d.st == 0]
    t, p = stats.ttest_ind(hi.ic, lo.ic, equal_var=False)
    t2, p2 = stats.ttest_ind(hi.ls.dropna(), lo.ls.dropna(), equal_var=False)
    res[name] = (hi.ic.mean() - lo.ic.mean(), t, p)
    say(f"  {name}")
    say("    Rank IC：" + "｜".join(f"{['低', '中', '高'][int(k)]}組 {v.ic.mean():.4f}（{len(v)} 天）" for k, v in grp)
        + f"｜高−低 {hi.ic.mean() - lo.ic.mean():+.4f}（t={t:+.2f}，p={p:.4f}）→ {'通過' if p < ALPHA else '未通過'}")
    say("    最高20%−最低20%（bp/天）：" + "｜".join(f"{['低', '中', '高'][int(k)]}組 {v.ls.mean():+.1f}" for k, v in grp)
        + f"｜高−低 {hi.ls.mean() - lo.ls.mean():+.1f}（t={t2:+.2f}）")
    for a, b in [("2018", "2020"), ("2021", "2024")]:
        x = d[(d.index >= a) & (d.index <= b + "-12-31")]
        h, l = x[x.st == 2].ic, x[x.st == 0].ic
        say(f"      {a}~{b}：高 {h.mean():.4f}／低 {l.mean():.4f}｜差 {h.mean() - l.mean():+.4f}（t={stats.ttest_ind(h, l, equal_var=False)[0]:+.2f}）")

# ---------------- 月頻（探索性）：三個選股訊號在高/低情緒月份之後 ----------------
raw["ym"] = raw.Date.dt.to_period("M")
abn = raw.abnormal_price.fillna(0) != 0
raw["r"] = g.Close.pct_change().where(~(abn | abn.groupby(raw.stock_code).shift(1).fillna(False)))
last = raw[(~abn) & raw.Close.notna()].drop_duplicates(["stock_code", "ym"], keep="last").set_index(["stock_code", "ym"])[["Close", "short_margin_ratio"]]
agg = raw.groupby(["stock_code", "ym"]).agg(vol=("r", "std"), mx=("r", "max"), n=("r", "count"))
P = last.join(agg).reset_index().sort_values(["stock_code", "ym"]).reset_index(drop=True)
P["mi"] = P.ym.dt.year * 12 + P.ym.dt.month
gp = P.groupby("stock_code")
P["ret_next"] = (gp.Close.shift(-1) / P.Close - 1).where(gp.mi.shift(-1) == P.mi + 1)
P.loc[P.n < 10, ["vol", "mx"]] = np.nan
P["smr"] = P.short_margin_ratio.clip(0, 1)
MS = pd.DataFrame({"S1 說的-多空": (em["看多"] - em["看空"]).groupby(em.index.to_period("M")).mean(),
                   "S2 說的-恐慌": em["恐慌"].groupby(em.index.to_period("M")).mean()})
mbm = mb.groupby(mb.index.to_period("M")).last()
MS["S3 做的-融資餘額月變化"] = mbm.pct_change().reindex(MS.index)
say("\n===== 月頻（探索性，月數少）：選股訊號的月 Rank IC（訊號越高→下月報酬越低，所以 IC 為負＝有效），依當月市場情緒高/低分 =====")
for col, nm in [("vol", "波動度"), ("mx", "最大單日漲幅"), ("smr", "券資比")]:
    icm = pd.Series({ym: stats.spearmanr(z[col], z.ret_next)[0] for ym, z in P[P[col].notna() & P.ret_next.notna()].groupby("ym") if len(z) >= 30})
    line = f"  {nm}（全期 IC {icm.mean():+.4f}，{len(icm)} 個月）"
    for sname in MS.columns:
        s = MS[sname].reindex(icm.index)
        med = s.expanding(min_periods=24).median().shift(1)
        hi, lo = icm[s > med], icm[s <= med]
        line += f"｜{sname}：高 {hi.mean():+.3f}（{len(hi)}）／低 {lo.mean():+.3f}（{len(lo)}），差 t={stats.ttest_ind(hi, lo, equal_var=False)[0]:+.2f}"
    say(line)
say("\n主要檢定彙總：" + "；".join(f"{k.split()[0]} 高−低 {v[0]:+.4f}（p={v[2]:.4f}，{'通過' if v[2] < ALPHA else '未通過'}）" for k, v in res.items()))
open("sentiment_regime_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
