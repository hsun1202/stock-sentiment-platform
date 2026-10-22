# -*- coding: utf-8 -*-
"""
財報特徵加進機器學習月頻選股（事先寫定見 PREREG_ai_strategies.md D 節）。
面板與規格沿用 ml_monthly_backtest.py（直接執行它「模型」之前的部分），再接上只用「發布日早於訊號日」的財報特徵。
cd D:/專題claude && python ai_strategies/ml_fund_backtest.py
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

src = open("ai_strategies/ml_monthly_backtest.py", encoding="utf-8").read().split("# ---------- 模型 ----------")[0]
exec(src)                                   # 產生 P（股票×月面板）、FEAT、FD、say、t_、perf、COST、OUT
L.clear()


def code_of(s):
    return s.astype(str).str.split().str[0]


def to_date(s):
    return pd.to_datetime(s.astype(str).str.replace(r"\.0$", "", regex=True).str.replace("/", ""), format="%Y%m%d", errors="coerce")


# ---------- 月營收 ----------
rv = pd.read_csv("tej_raw/rev/20261022103241.csv", encoding="utf-16", sep="\t", on_bad_lines="skip", low_memory=False, na_values=["."])
rv["code"] = code_of(rv["證券代碼"])
rv = rv[rv.code.str.match(r"^[1-9]\d{3}$")]
pe = pd.to_datetime(rv["年月"].astype(str) + "01", format="%Y%m%d")
deadline = pe + pd.offsets.MonthBegin(1) + pd.Timedelta(days=9)        # 沒有發布日的：用法定期限（次月 10 日）
rv["pub"] = to_date(rv["營收發布日"]).fillna(deadline)
clipv = lambda s: s.clip(-100, 500)  # noqa: E731
R = pd.DataFrame({"code": rv.code, "pub": rv.pub,
                  "rev_yoy": clipv(rv["以合併為主單月營收成長率％"].fillna(rv["單月營收成長率％"])),
                  "rev_yoy3": clipv(rv["近3月累計營收成長率"]), "rev_yoy12": clipv(rv["近12月累計營收成長率"]),
                  "rev_mom": clipv(rv["以合併為主單月營收與上月比％"].fillna(rv["單月營收與上月比％"]))})
R["rev_accel"] = R.rev_yoy3 - R.rev_yoy12
R = R.dropna(subset=["pub"])

# ---------- 季報 ----------
fn = pd.read_csv("tej_raw/fin/20261022104526.csv", encoding="utf-16", sep="\t", low_memory=False, na_values=["."])
fn["code"] = code_of(fn["證券代碼"])
fn = fn[fn.code.str.match(r"^[1-9]\d{3}$")]
fn["pub"] = to_date(fn["財報發布日"])
qend = pd.to_datetime(fn["年月"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)
bad = fn.pub < qend
fn = fn[~bad & fn.pub.notna()]
pe_q = fn["當季季底P/E"]
Q = pd.DataFrame({"code": fn.code, "pub": fn.pub, "roe": fn["ROE(A)－稅後"].clip(-100, 100), "gm": fn["營業毛利率"].clip(-100, 100),
                  "debt": fn["負債比率"], "cfo_a": (fn["來自營運之現金流量"] / fn["資產總額"].where(fn["資產總額"] > 0)).clip(-1, 1),
                  "ep": (1 / pe_q.where(pe_q.abs() > 0)).clip(-1, 1)})
FUND = ["rev_yoy", "rev_yoy3", "rev_yoy12", "rev_mom", "rev_accel", "roe", "gm", "debt", "cfo_a", "ep"]
say(f"[財報] 月營收 {len(R):,} 筆、季報 {len(Q):,} 筆（剔除發布日早於季底 {int(bad.sum())} 筆）")

# ---------- 依「發布日早於訊號日」對到每個股票×月 ----------
G = P[["stock_code", "fd"]].rename(columns={"stock_code": "code"}).reset_index()
G = G.sort_values("fd")
for T, cols in [(R, ["rev_yoy", "rev_yoy3", "rev_yoy12", "rev_mom", "rev_accel"]), (Q, ["roe", "gm", "debt", "cfo_a", "ep"])]:
    T = T.sort_values("pub").rename(columns={"pub": "pub_" + cols[0]})
    G = pd.merge_asof(G, T[["code", "pub_" + cols[0]] + cols], left_on="fd", right_on="pub_" + cols[0], by="code",
                      allow_exact_matches=False, direction="backward")
    stale = (G.fd - G["pub_" + cols[0]]).dt.days > (70 if cols[0].startswith("rev") else 200)   # 太舊的不用
    G.loc[stale, cols] = np.nan
G = G.set_index("index")
P = P.join(G[FUND])
P["year"] = P.fd.dt.year
cov = P.loc[P.year >= 2018, FUND].notna().mean()
say("[覆蓋率 2018 後] " + "｜".join(f"{c} {v:.0%}" for c, v in cov.items()))

# ---------- 兩組模型 ----------
ARMS = {"A 技術面＋散戶行為": FEAT, "F 加財報": FEAT + FUND}
for arm, cols in ARMS.items():
    P["s_" + arm] = np.nan
    for Y in range(2018, 2026):
        tr, va, te = (P.year <= Y - 2), (P.year == Y - 1), (P.year == Y)
        ps = []
        for s in range(3):
            m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=5, early_stopping_rounds=30, tree_method="hist", n_jobs=8, random_state=s, verbosity=0)
            m.fit(P.loc[tr, cols], P.loc[tr, "tgt"], eval_set=[(P.loc[va, cols], P.loc[va, "tgt"])], verbose=False)
            ps.append(m.predict(P.loc[te, cols]))
        P.loc[te, "s_" + arm] = np.mean(ps, axis=0)
U = P[P.liq_ok & P["s_" + list(ARMS)[0]].notna()].copy()


def run(col):
    out, held = {}, None
    for ym, z in U.groupby("ym"):
        h = z[z[col].rank(pct=True, ascending=False) <= 0.2]
        s = set(h.stock_code)
        turn = 1.0 if held is None else len(s - held) / max(len(s), 1)
        out[ym] = h.ret_next.mean() - turn * COST
        held = s
    return pd.Series(out)


ew = U.groupby("ym").ret_next.mean()
e = pd.read_csv("quarterly_vol/etf0050_adj.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()
b50 = pd.Series({m.ym: e.asof(m.sd_next) / e.asof(m.sd) - 1 for _, m in FD.iterrows()})
res = {arm: run("s_" + arm) for arm in ARMS}
idx = res[list(ARMS)[0]].index
say("\n===== 財報特徵加進月頻選股（上市、扣成本），2018–2025 =====")
for nm, r in list(res.items()) + [("一般股票（等權）", ew.reindex(idx)), ("0050 含息", b50.reindex(idx))]:
    a, v, shp, dd = perf(r)
    ic = ""
    if nm in ARMS:
        ics = U.groupby("ym").apply(lambda z: spearmanr(z["s_" + nm], z.ret_next)[0])
        ic = f"｜月 Rank IC {ics.mean():+.4f}（t={t_(ics):+.1f}）"
    say(f"  {nm:<14} 年化 {a:+.1%}｜波動 {v:.1%}｜Sharpe {shp:.2f}｜最大回落 {dd:.1%}{ic}")
dF = (res["F 加財報"] - res["A 技術面＋散戶行為"]).dropna()
d50 = (res["F 加財報"] - b50.reindex(idx)).dropna()
dew = (res["F 加財報"] - ew.reindex(idx)).dropna()
yr = dF.groupby(pd.PeriodIndex(dF.index).year).sum()
say(f"  加財報 − 不加：分年 " + " ".join(f"{y}:{v:+.1%}" for y, v in yr.items()))
say(f"  加財報版 相對一般股票 {dew.mean():+.2%}／月（t={t_(dew):+.2f}）")
say(f"\n→ F1（加財報 − 不加 > 0）：{dF.mean():+.3%}／月，t={t_(dF):+.2f} → {'通過' if dF.mean() > 0 and t_(dF) > 1.96 else '未通過'}")
say(f"→ F2（加財報 − 0050 含息 > 0）：{d50.mean():+.2%}／月，t={t_(d50):+.2f} → {'通過' if d50.mean() > 0 and t_(d50) > 1.96 else '未通過'}")
d25 = dF[pd.PeriodIndex(dF.index).year == 2025]
say(f"  2025 年：加財報 − 不加 {d25.sum():+.1%}（{len(d25)} 個月）；相對 0050 {d50[pd.PeriodIndex(d50.index).year == 2025].sum():+.1%}")
say("\n單一財報特徵對下月報酬的 Rank IC（可買範圍，2018–2025）：")
for c in FUND:
    ics = U.groupby("ym").apply(lambda z: spearmanr(z[c], z.ret_next, nan_policy="omit")[0])
    say(f"  {c:<10} {ics.mean():+.4f}（t={t_(ics):+.1f}）")
pd.DataFrame({k: v for k, v in res.items()} | {"ew": ew.reindex(idx), "etf0050": b50.reindex(idx)}).to_csv(OUT + "ml_fund_returns.csv", encoding="utf-8-sig")
P[["stock_code", "fd", "ym", "ret_next", "liq_ok", "mktcap"] + FEAT + FUND + ["s_" + a for a in ARMS]].to_pickle(OUT + "panel_monthly_fund.pkl")
open(OUT + "ml_fund_backtest.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
