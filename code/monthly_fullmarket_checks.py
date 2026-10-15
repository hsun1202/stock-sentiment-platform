# -*- coding: utf-8 -*-
"""
月頻全市場選股（monthly_fullmarket_strategy.py）的第二輪檢查——2026-10-12，跑之前寫定，全部照實報：
  (1) 月底收漲停買不到：把月底當天收漲停的股票從買進名單剔除（不遞補），看占多少、績效差多少。
  (2) 拿掉 2018 年（那年模型只用 2015~2016 兩年訓練，而且相對基準贏最多）：2019~2025 還贏不贏。
  (3) 分前後段：2018~2021、2022~2025。
  (4) 不用機器學習的透明版本：只用三個從專題一開始就點名的訊號——低波動、避開暴漲股（當月最大單日漲幅低）、券資比低——
      各自在當月排名後取平均，買分數最高的 20%。沒有任何要訓練的參數。看它自己贏不贏、和模型差多少。
全部：成交值最低 20% 不買、扣 0.471%×換股、下市記 −50%（沿用 monthly_fullmarket_preds.pkl 的報酬）。
python monthly_fullmarket_checks.py
"""
import numpy as np
import pandas as pd

LOG = []
C = 0.00471
COL = "p_技術面＋散戶行為★"


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


T = pd.read_pickle("monthly_fullmarket_preds.pkl")
px = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str, "limit_flag": str}, parse_dates=["Date"],
                 usecols=["stock_code", "Date", "Close", "Volume", "limit_flag", "abnormal_price", "short_margin_ratio"])
px = px.sort_values(["stock_code", "Date"])
px["ym"] = px.Date.dt.to_period("M")
g = px.groupby("stock_code")
abn = px.abnormal_price.fillna(0) != 0
px["r"] = g.Close.pct_change().where(~(abn | abn.groupby(px.stock_code).shift(1).fillna(False)))
last = px[px.Close.notna()].drop_duplicates(["stock_code", "ym"], keep="last")[["stock_code", "ym", "limit_flag", "Volume", "short_margin_ratio"]]
agg = px.groupby(["stock_code", "ym"]).agg(vol1m=("r", "std"), maxret=("r", "max"), n_r=("r", "count")).reset_index()
T = T.merge(last, on=["stock_code", "ym"], how="left").merge(agg, on=["stock_code", "ym"], how="left")
T["lim_up"] = (T.limit_flag == "+") | (T.Volume.fillna(0) <= 0)
U = T[T.liq_ok].copy()


def tstat(s):
    s = pd.Series(s).dropna()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def perf(r):
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod()
    return ann, vol, r.mean() * 12 / vol, (eq / eq.cummax() - 1).min()


ew = U.groupby("ym").ret_next.mean()
vw = U.groupby("ym").apply(lambda z: np.average(z.ret_next, weights=z.mktcap.fillna(0) + 1e-9))


def port(score, drop_limit=False):
    months = sorted(U.ym.unique())
    top = U[U.groupby("ym")[score].rank(pct=True) > 0.8]
    if drop_limit:
        top = top[~top.lim_up]
    gross = top.groupby("ym").ret_next.mean()
    hold = top.groupby("ym").stock_code.apply(set)
    turn = pd.Series({m: (1.0 if i == 0 else len(hold[m] - hold[months[i - 1]]) / len(hold[m])) for i, m in enumerate(months)})
    return gross - turn * C, turn, top


def report(label, net, sel=None):
    if sel is not None:
        net = net[sel(net.index)]
    e1, e2 = net - ew.reindex(net.index), net - vw.reindex(net.index)
    a, v, sh, dd = perf(net)
    say(f"  {label:<26s} {len(net)} 個月｜年化 {a:+.1%}、Sharpe {sh:.2f}、最大回落 {dd:.1%}｜相對等權 {e1.mean():+.2%}/月（t={tstat(e1):+.2f}）｜相對市值加權 {e2.mean():+.2%}/月（t={tstat(e2):+.2f}）")


net, turn, top = port(COL)
say("===== (1) 月底收漲停買不到 =====")
say(f"  買進名單中，月底當天收漲停或沒成交的占 {top.lim_up.mean():.2%}（每月約 {top.groupby('ym').lim_up.sum().mean():.1f} 檔／{top.groupby('ym').size().mean():.0f} 檔）；這些股票下個月平均報酬 {top[top.lim_up].ret_next.mean():+.2%}，其餘 {top[~top.lim_up].ret_next.mean():+.2%}")
report("原設定", net)
net_l, _, _ = port(COL, drop_limit=True)
report("剔除月底收漲停", net_l)

say("\n===== (2)(3) 拿掉 2018 年、分前後段（原設定）=====")
report("2019~2025", net, lambda i: i.year >= 2019)
report("2018~2021", net, lambda i: i.year <= 2021)
report("2022~2025", net, lambda i: i.year >= 2022)
report("2024~2025", net, lambda i: i.year >= 2024)

say("\n===== (4) 不用機器學習的透明版本：低波動＋避開暴漲股＋低券資比（三個排名取平均）=====")
U.loc[U.n_r < 10, ["vol1m", "maxret"]] = np.nan
U["smr"] = U.short_margin_ratio.clip(0, 1)
r1 = 1 - U.groupby("ym").vol1m.rank(pct=True)
r2 = 1 - U.groupby("ym").maxret.rank(pct=True)
r3 = (1 - U.groupby("ym").smr.rank(pct=True)).fillna(0.5)          # 沒有融資融券資料的當成中間
U["simple"] = pd.concat([r1, r2, r3], axis=1).mean(axis=1, skipna=False)
U["simple2"] = pd.concat([r1, r2], axis=1).mean(axis=1, skipna=False)
for sc, nm in [("simple", "三個訊號（含券資比）"), ("simple2", "只有低波動＋避開暴漲股")]:
    n_, t_, tp = port(sc)
    say(f"  [{nm}] 每月換股 {t_.iloc[1:].mean():.0%}、每月約 {tp.groupby('ym').size().mean():.0f} 檔")
    report("2018~2025", n_)
    report("2019~2025", n_, lambda i: i.year >= 2019)
    report("2025", n_, lambda i: i.year == 2025)
    yr = ((n_ - ew).groupby(n_.index.year).mean() * 12)
    say("    分年（相對等權，年化）：" + " ".join(f"{y}:{x:+.1%}" for y, x in yr.items()) + f"｜為正 {(yr > 0).sum()}/{len(yr)}")
n3, _, _ = port("simple")
d = (net - n3).dropna()
say(f"  機器學習模型 − 透明版本（三個訊號）：{d.mean():+.2%}/月（t={tstat(d):+.2f}）")
d = (n3 - port("simple2")[0]).dropna()
say(f"  券資比的增量（三個訊號 − 兩個訊號）：{d.mean():+.2%}/月（t={tstat(d):+.2f}）")
open("monthly_fullmarket_checks.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
