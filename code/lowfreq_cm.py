# -*- coding: utf-8 -*-
"""
把交易拉長：週頻、月頻選股，看 CMoney（與 PTT）情緒在「技術面＋散戶行為」之上有沒有增量。事先寫定見 PREREG_cmoney_full.md 第 D 節。
做法同 weekly_fullmarket_strategy.py / monthly_fullmarket_strategy.py；股票＝265 檔（CMoney 有資料）、期間 2020–2024、測試 2022–2024。
python cmoney_full/lowfreq_cm.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
CM_TAG = os.environ.get("CM_TAG", "cm3")
C_DISC, XSEEDS, TEST_YEARS = 0.00471, 3, [2022, 2023, 2024]
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def tstat(s):
    s = pd.Series(s).dropna()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def ic_series(U, col):
    return U.groupby("per").apply(lambda z: spearmanr(z[col], z.ret_next, nan_policy="omit")[0])


# ---------- 日資料 ----------
use = ["stock_code", "Date", "Close", "close_unadj", "Volume", "abnormal_price", "MA60", "RSI14", "mktcap",
       "margin_bal", "short_margin_ratio", "inst_share", "credit_share", "in265"]
df = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=use, parse_dates=["Date"])
df = df[(df.in265 == 1) & (df.Date <= "2024-12-31")].sort_values(["stock_code", "Date"]).reset_index(drop=True)
SC = [f"ptt_{a}_{b}" for a in ("pre", "intra") for b in ("n", "pos", "neg")]
for tag, path in [("cm", f"final_dataset_v3_{CM_TAG}/all_stocks_v3.csv"), ("pt", "final_dataset_v3/all_stocks_v3.csv")]:
    s = pd.read_csv(path, dtype={"stock_code": str}, usecols=["stock_code", "Date"] + SC, parse_dates=["Date"])
    s[f"{tag}_n"] = s.ptt_pre_n + s.ptt_intra_n               # 當天收盤前（前一天收盤後～當天收盤）
    s[f"{tag}_pos"] = s.ptt_pre_pos + s.ptt_intra_pos
    s[f"{tag}_neg"] = s.ptt_pre_neg + s.ptt_intra_neg
    df = df.merge(s[["stock_code", "Date", f"{tag}_n", f"{tag}_pos", f"{tag}_neg"]], on=["stock_code", "Date"], how="left")
g = df.groupby("stock_code")
abn = df.abnormal_price.fillna(0) != 0
okd = ~(abn | abn.groupby(df.stock_code).shift(1).fillna(False))
tr_ = df.Volume > 0
df["r"] = g.Close.pct_change().where(okd)
df["dv"] = (df.close_unadj * df.Volume).where(tr_)
df["vpos"] = df.Volume.where(tr_)
df["rs"] = ((100 - df.inst_share) / 100).where(tr_)
df["cr"] = (df.credit_share / 100).where(tr_)


def roll(col, w, f, mp):
    return g[col].transform(lambda x: getattr(x.rolling(w, min_periods=mp), f)())


df["vol20"], df["max20"], df["dv20"] = roll("r", 20, "std", 10), roll("r", 20, "max", 10), roll("dv", 20, "median", 10)
df["v5"], df["v60"] = roll("vpos", 5, "mean", 3), roll("vpos", 60, "mean", 30)
df["hi250"] = roll("Close", 250, "max", 120)
df["rs20"] = g.rs.transform(lambda x: x.rolling(20, min_periods=10).mean().shift(5))
df["mb5"] = g.margin_bal.shift(5)


def panel(freq):
    d = df.copy()
    d["per"] = d.Date.dt.to_period(freq)
    last = d[d.Close.notna()].drop_duplicates(["stock_code", "per"], keep="last").set_index(["stock_code", "per"])[
        ["Date", "Close", "MA60", "RSI14", "mktcap", "margin_bal", "mb5", "short_margin_ratio", "vol20", "max20", "dv20", "v5", "v60", "hi250", "rs20"]]
    agg = d.groupby(["stock_code", "per"]).agg(rs_p=("rs", "mean"), cr_p=("cr", "mean"),
                                                cm_n=("cm_n", "sum"), cm_pos=("cm_pos", "sum"), cm_neg=("cm_neg", "sum"),
                                                pt_n=("pt_n", "sum"), pt_pos=("pt_pos", "sum"), pt_neg=("pt_neg", "sum"))
    P = last.join(agg).reset_index().sort_values(["stock_code", "per"]).reset_index(drop=True)
    P["pi"] = P.per.apply(lambda p: p.ordinal)

    def lag(s, k):
        return s.groupby(P.stock_code).shift(k).where(P.pi.groupby(P.stock_code).shift(k) == P.pi - k)

    C = P.Close
    if freq == "W-FRI":
        P["m1"], P["m2"], P["m3"], P["m4"] = C / lag(C, 1) - 1, C / lag(C, 4) - 1, C / lag(C, 12) - 1, lag(C, 4) / lag(C, 26) - 1
    elif freq == "M":
        P["m1"], P["m2"], P["m3"], P["m4"] = C / lag(C, 1) - 1, C / lag(C, 3) - 1, C / lag(C, 6) - 1, lag(C, 1) / lag(C, 12) - 1
    else:                                       # 季頻：1、2、4 季動能、4 季−1 季
        P["m1"], P["m2"], P["m3"], P["m4"] = C / lag(C, 1) - 1, C / lag(C, 2) - 1, C / lag(C, 4) - 1, lag(C, 1) / lag(C, 4) - 1
    P["logdv"], P["voltrend"] = np.log(P.dv20), np.log(P.v5 / P.v60)
    P["ma60r"], P["rsi"], P["hi52"] = C / P.MA60 - 1, P.RSI14, C / P.hi250 - 1
    P["smr"] = P.short_margin_ratio.clip(0, 1)
    P["margin_chg5"] = (P.margin_bal / P.mb5.where(P.mb5 > 0) - 1).clip(-0.5, 0.5)
    P["rs_dev"] = P.rs_p - P.rs20
    for t in ("cm", "pt"):
        att = np.log1p(P[f"{t}_n"])
        net = (P[f"{t}_pos"] - P[f"{t}_neg"]) / (P[f"{t}_pos"] + P[f"{t}_neg"]).replace(0, np.nan)
        P[f"{t}_att"], P[f"{t}_net"] = att, net
        P[f"{t}_att_chg"] = att - sum(lag(att, k) for k in range(1, 5)) / 4      # 討論量相對前 4 期
        P[f"{t}_net_chg"] = net - lag(net, 1)
    last_pi, data_end = P.groupby("stock_code").pi.transform("max"), P.pi.max()
    nxt = P.pi.groupby(P.stock_code).shift(-1)
    P["ret_next"] = (C.groupby(P.stock_code).shift(-1) / C - 1).where(nxt == P.pi + 1)
    P.loc[P.ret_next.isna() & (P.pi == last_pi) & (P.pi < data_end - 1), "ret_next"] = -0.5
    gap = P.ret_next.isna() & (P.pi < last_pi)
    P.loc[gap, "ret_next"] = (C.groupby(P.stock_code).shift(-1) / C - 1)[gap]
    P = P[P.Date >= "2020-01-01"].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    P["year"] = P.Date.dt.year
    P["liq_ok"] = P.groupby("per").logdv.rank(pct=True) > 0.2
    P["tgt"] = P.groupby("per").ret_next.rank(pct=True) - 0.5
    return P


BASE = ["m1", "m2", "m3", "m4", "vol20", "max20", "logdv", "voltrend", "ma60r", "rsi", "hi52", "smr", "margin_chg5", "cr_p", "rs_p", "rs_dev"]
CMF = ["cm_att", "cm_net", "cm_att_chg", "cm_net_chg"]
PTF = ["pt_att", "pt_net", "pt_att_chg", "pt_net_chg"]
B0 = "基準（技術面＋散戶行為）"
ARMS = {B0: BASE, "+CMoney": BASE + CMF, "+PTT": BASE + PTF}


def run(freq, label, per_year, use_buffer):
    P = panel(freq)
    valid = P.ret_next.notna() & P.tgt.notna()
    say(f"\n===== {label}：{P.stock_code.nunique()} 檔、{P.per.min()}~{P.per.max()}；每期可買約 {P[P.liq_ok].groupby('per').size().mean():.0f} 檔；"
        f"當期有 CMoney 貼文 {(P.cm_n > 0).mean():.0%}、有 PTT {(P.pt_n > 0).mean():.0%} =====")
    pred = {a: pd.Series(np.nan, index=P.index) for a in ARMS}
    for Y in TEST_YEARS:
        tr, va, te = valid & (P.year <= Y - 2), valid & (P.year == Y - 1), valid & (P.year == Y)
        for arm, cols in ARMS.items():
            ps = []
            for s in range(XSEEDS):
                m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=5, early_stopping_rounds=30, tree_method="hist", n_jobs=8, random_state=s, verbosity=0)
                m.fit(P.loc[tr, cols], P.loc[tr, "tgt"], eval_set=[(P.loc[va, cols], P.loc[va, "tgt"])], verbose=False)
                ps.append(m.predict(P.loc[te, cols]))
            pred[arm][te] = np.mean(ps, axis=0)
    U = P[valid & P.year.isin(TEST_YEARS) & P.liq_ok].copy()
    pers = sorted(U.per.unique())
    bench = U.groupby("per").ret_next.mean()
    vw = U.groupby("per").apply(lambda z: np.average(z.ret_next, weights=z.mktcap.fillna(0) + 1e-9))

    def perf(r):
        ann = (1 + r).prod() ** (per_year / len(r)) - 1
        vol = r.std() * np.sqrt(per_year)
        eq = (1 + r).cumprod()
        return ann, r.mean() * per_year / vol, (eq / eq.cummax() - 1).min()

    def build(buffer):
        held, gross, turn = set(), {}, {}
        G = {w: z for w, z in U.groupby("per")}
        for w in pers:
            z = G[w].sort_values("s", ascending=False)
            n = max(int(len(z) * 0.2), 1)
            top = list(z.stock_code.iloc[:n])
            if buffer:
                keep = held & set(z.stock_code.iloc[:int(len(z) * 0.4)])
                new = keep | set([c for c in top if c not in keep][:max(n - len(keep), 0)])
            else:
                new = set(top)
            turn[w] = 1.0 if not held else len(new - held) / len(new)
            gross[w] = z[z.stock_code.isin(new)].ret_next.mean()
            held = new
        gross, turn = pd.Series(gross), pd.Series(turn)
        return gross - turn * C_DISC, gross, turn

    a, sh, dd = perf(bench)
    say(f"  基準 等權：年化 {a:+.1%}、Sharpe {sh:.2f}、回落 {dd:.1%}｜市值加權：年化 {perf(vw)[0]:+.1%}、Sharpe {perf(vw)[1]:.2f}")
    unit = {"W-FRI": "週", "M": "月", "Q": "季"}[freq]
    res = {}
    for arm in ARMS:
        U["s"] = pred[arm][U.index]
        ic = ic_series(U, "s")
        res[("ic", arm)] = ic
        say(f"  {arm}｜Rank IC {ic.mean():+.4f}（t={tstat(ic):+.1f}）")
        for buf in ([False, True] if use_buffer else [False]):
            net, gross, turn = build(buf)
            ex, exv = net - bench, net - vw
            a, sh, dd = perf(net)
            res[(arm, buf)] = net
            nm = "B 降低換股" if buf else "A 每期換前20%"
            say(f"    {nm}：換股 {turn.iloc[1:].mean():.0%}/{unit}｜毛 {gross.mean():+.2%}、扣成本 {net.mean():+.2%}/{unit} → 年化 {a:+.1%}、Sharpe {sh:.2f}、回落 {dd:.1%}"
                f"｜相對等權 {ex.mean():+.3%}（t={tstat(ex):+.2f}）｜相對市值加權 {exv.mean():+.3%}（t={tstat(exv):+.2f}）")
    out = {}
    for arm in ["+CMoney", "+PTT"]:
        d = (res[(arm, use_buffer)] - res[(B0, use_buffer)]).dropna()
        dic = (res[("ic", arm)] - res[("ic", B0)]).dropna()
        say(f"  {arm} 的增量：扣成本後 {d.mean():+.3%}/{unit}（t={tstat(d):+.2f}，{len(d)} 期）｜ΔIC {dic.mean():+.4f}（t={tstat(dic):+.2f}）")
        out[arm] = d
    parts = []
    for c in CMF + ["pt_att", "pt_att_chg"]:
        s = ic_series(U, c)
        parts.append(f"{c} {s.mean():+.3f}（t={tstat(s):+.1f}）")
    say("  單一變數對下期報酬的 Rank IC：" + "｜".join(parts))
    return out


w = run("W-FRI", "週頻（每週五收盤換股）", 52, True)
m = run("M", "月頻（每月底換股）", 12, False)
q = run("Q", "季頻（每季底換股；測試只有 12 季，只報數字）", 4, False)
d = w["+CMoney"]
say(f"\n→ L1（週頻 +CMoney 扣成本後 − 基準 > 0）：{d.mean():+.3%}/週，t={tstat(d):+.2f} → {'通過' if d.mean() > 0 and tstat(d) > 1.96 else '未通過'}")
d = m["+CMoney"]
say(f"→ L2（月頻 +CMoney 扣成本後 − 基準 > 0）：{d.mean():+.3%}/月，t={tstat(d):+.2f} → {'通過' if d.mean() > 0 and tstat(d) > 1.96 else '未通過'}")
d = q["+CMoney"]
say(f"（季頻，只報數字）+CMoney 扣成本後 − 基準：{d.mean():+.3%}/季，t={tstat(d):+.2f}，{len(d)} 期")
open(f"cmoney_full/lowfreq_cm_{CM_TAG}.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
