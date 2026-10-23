# -*- coding: utf-8 -*-
"""
E 節：法人籌碼（投信、外資、自營買賣超）加進月頻選股＋透明規則「投信買超前 30 檔」。
F 節：以贏 0050 為目標——市值前 50 大、照市值再依 A 模型分數調權重。
事先寫定見 PREREG_ai_strategies.md E、F 節。面板與規格沿用 ml_monthly_backtest.py（執行它「模型」之前的部分）。
cd D:/專題claude && python ai_strategies/inst_beat0050_backtest.py
"""
import glob
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

src = open("ai_strategies/ml_monthly_backtest.py", encoding="utf-8").read().split("# ---------- 模型 ----------")[0]
src = src.replace('"short_bal", "inst_share"]', '"short_bal", "inst_share", "foreign_net", "trust_net", "dealer_net"]')
INJECT = '''
# ---------- E 節：法人籌碼特徵（只用到當天為止；買賣超股數 ÷ 流通股數，‰）----------
shs = (df.mktcap * 1e6 / df.close_unadj).where((df.mktcap > 0) & (df.close_unadj > 0))
for c, k in [("foreign", "f"), ("trust", "t"), ("dealer", "d")]:
    x = df[c + "_net"].fillna(0.0)
    for w, nm in [(21, "1m"), (63, "3m")]:
        s = x.groupby(df.stock_code).transform(lambda v: v.rolling(w, min_periods=int(w * 0.7)).sum())
        df[f"{k}_net{nm}"] = (s / shs * 1000).clip(-300, 300)
    df[f"{k}_days1m"] = (x > 0).astype(float).groupby(df.stock_code).transform(lambda v: v.rolling(21, min_periods=15).mean())
INST = ["f_net1m", "f_net3m", "t_net1m", "t_net3m", "d_net1m", "t_days1m", "f_days1m"]
'''
mk = "# ---------- 每月第一、第二個交易日 ----------"
assert mk in src
src = src.replace(mk, INJECT + mk)
exec(src)                                   # 產生 P、FEAT、INST、FD、say、t_、perf、COST、OUT、df
L.clear()

# ---------- 公開資料核對：TEJ 投信／外資買賣超 vs 證交所 T86 ----------
for day in ["20160304", "20210805", "20250605"]:
    f = sorted(glob.glob(f"chip_raw/t86/{day[:6]}*.json"))
    if not f:
        continue
    j = json.load(open(f[0], encoding="utf-8"))
    d0 = pd.Timestamp(f[0][-13:-5])
    fl = j["fields"]
    it = [i for i, n in enumerate(fl) if "投信" in n and "買賣超" in n][0]
    ifo = [i for i, n in enumerate(fl) if "外" in n and "買賣超" in n][0]
    t86 = pd.DataFrame({"stock_code": [r[0].strip() for r in j["data"]],
                        "t86_t": [float(r[it].replace(",", "")) for r in j["data"]],
                        "t86_f": [float(r[ifo].replace(",", "")) for r in j["data"]]})
    m = df[df.Date == d0][["stock_code", "trust_net", "foreign_net"]].merge(t86, on="stock_code")
    say(f"[核對 T86 {d0.date()}] {len(m)} 檔｜投信 相關 {m.trust_net.corr(m.t86_t):.4f}、完全相同 {(np.isclose(m.trust_net, m.t86_t, atol=1000)).mean():.1%}"
        f"｜外資 相關 {m.foreign_net.corr(m.t86_f):.4f}（T86 欄位「{fl[ifo]}」）")
del df

# ---------- 兩組模型：A（同上線規格）與 A＋法人籌碼 ----------
P["year"] = P.fd.dt.year
ARMS = {"A": FEAT, "I": FEAT + INST}
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
PT = P[P.s_A.notna()].copy()                # 測試期全部上市股（F 節用）
U = PT[PT.liq_ok].copy()                     # 可買範圍（E 節用）


def eq_port(sel):
    """等權組合；成本＝換掉的檔數比例 × 0.471%（同 A）"""
    out, turns, held = {}, {}, None
    for ym, z in U.groupby("ym"):
        h = sel(z)
        s = set(h.stock_code)
        turn = 1.0 if held is None else len(s - held) / max(len(s), 1)
        out[ym], turns[ym] = h.ret_next.mean() - turn * COST, turn
        held = s
    return pd.Series(out), pd.Series(turns)


def cap_port(N, mult=None):
    """市值前 N 大；照市值，mult＝(前三分之一, 中, 後) 依 A 分數調權重；成本＝實際換手（含漂移）× 0.471%"""
    out, turns, prev = {}, {}, None
    for ym, z in PT.groupby("ym"):
        z = z.dropna(subset=["mktcap"]).nlargest(N, "mktcap")
        w = z.mktcap.values.astype(float)
        if mult is not None:
            g3 = pd.qcut(z.s_A.rank(method="first"), 3, labels=False).values      # 0＝最低三分之一
            w = w * np.select([g3 == 2, g3 == 1], [mult[0], mult[1]], mult[2])
        w = pd.Series(w / w.sum(), index=z.stock_code.values)
        r = pd.Series(z.ret_next.values, index=z.stock_code.values)
        turn = 1.0 if prev is None else 0.5 * w.sub(prev, fill_value=0).abs().sum()
        gross = (w * r).sum()
        out[ym], turns[ym] = gross - turn * COST, turn
        prev = w * (1 + r) / (1 + gross)
    return pd.Series(out), pd.Series(turns)


def verdict(d):
    return "通過" if d.mean() > 0 and t_(d) > 1.96 else "未通過"


def line(nm, r):
    a, v, shp, dd = perf(r)
    say(f"  {nm:<18} 年化 {a:+.1%}｜波動 {v:.1%}｜Sharpe {shp:.2f}｜最大回落 {dd:.1%}")


def yearly(d):
    yr = d.groupby(pd.PeriodIndex(d.index).year).sum()
    return " ".join(f"{y}:{v:+.1%}" for y, v in yr.items())


top20 = lambda col: (lambda z: z[z[col].rank(pct=True, ascending=False) <= 0.2])  # noqa: E731
netA, turnA = eq_port(top20("s_A"))
netI, turnI = eq_port(top20("s_I"))
netR, turnR = eq_port(lambda z: z[z.t_net1m > 0].nlargest(30, "t_net1m"))
idx = netA.index
ew = U.groupby("ym").ret_next.mean().reindex(idx)
e = pd.read_csv("quarterly_vol/etf0050_adj.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()
b50 = pd.Series({m.ym: e.asof(m.sd_next) / e.asof(m.sd) - 1 for _, m in FD.iterrows()}).reindex(idx)

# ================= E 節 =================
say("===== E. 法人籌碼（上市、扣成本），2018–2025 =====")
say(f"  每月可買約 {U.groupby('ym').size().mean():.0f} 檔；前 20% 約 {U.groupby('ym').size().mean() * 0.2:.0f} 檔；I2 每月實際檔數平均 "
    f"{U.groupby('ym').apply(lambda z: min(30, int((z.t_net1m > 0).sum()))).mean():.1f}")
for nm, r in [("A 技術面＋散戶行為", netA), ("A＋法人籌碼", netI), ("I2 投信買超前 30", netR), ("一般股票（等權）", ew), ("0050 含息", b50)]:
    line(nm, r)
for nm, col in [("A", "s_A"), ("A＋法人籌碼", "s_I")]:
    ics = U.groupby("ym").apply(lambda z: spearmanr(z[col], z.ret_next)[0])
    say(f"  {nm} 月 Rank IC {ics.mean():+.4f}（t={t_(ics):+.1f}）")
say(f"  換股率：A {turnA.iloc[1:].mean():.0%}、A＋法人 {turnI.iloc[1:].mean():.0%}、I2 {turnR.iloc[1:].mean():.0%}")
dI = (netI - netA).dropna()
dR = (netR - ew).dropna()
say(f"  A＋法人 − A 分年：{yearly(dI)}")
say(f"  I2 − 一般股票 分年：{yearly(dR)}")
say(f"\n→ I1（A＋法人籌碼 − A > 0）：{dI.mean():+.3%}／月，t={t_(dI):+.2f} → {verdict(dI)}")
say(f"→ I2（投信買超前 30 − 一般股票 > 0）：{dR.mean():+.3%}／月，t={t_(dR):+.2f}（贏 {(dR > 0).mean():.0%} 的月） → {verdict(dR)}")
dR50 = (netR - b50).dropna()
say(f"  I2 相對 0050 含息 {dR50.mean():+.2%}／月（t={t_(dR50):+.2f}）")
qe = pd.PeriodIndex(dR.index).month.isin([3, 6, 9, 12])
say(f"  I2 − 一般股票：季底月（3/6/9/12 月訊號）{dR[qe].mean():+.2%}／月（{qe.sum()} 個月，t={t_(dR[qe]):+.2f}）｜其他月 {dR[~qe].mean():+.2%}（t={t_(dR[~qe]):+.2f}）")
for nm, d in [("A＋法人 − A", dI), ("I2 − 一般股票", dR)]:
    d25 = d[pd.PeriodIndex(d.index).year == 2025]
    say(f"  2025 年 {nm}：{d25.sum():+.1%}（{len(d25)} 個月）")
say("\n單一法人特徵對下月報酬的 Rank IC（可買範圍，2018–2025）：")
for c in INST:
    ics = U.groupby("ym").apply(lambda z: spearmanr(z[c], z.ret_next, nan_policy="omit")[0])
    say(f"  {c:<9} {ics.mean():+.4f}（t={t_(ics):+.1f}）")

# ================= F 節 =================
say("\n===== F. 以贏 0050 為目標：市值前 50 大＋A 分數調權重（扣成本），2018–2025 =====")
tilt, turnT = cap_port(50, (1.5, 1.0, 0.5))
cw50, turnC = cap_port(50)
tilt100, _ = cap_port(100, (1.5, 1.0, 0.5))
cw100, _ = cap_port(100)
tilt50s, turnS = cap_port(50, (2.0, 1.0, 0.0))
core = 0.7 * b50 + 0.3 * netA
for nm, r in [("大型股 AI 加權（50）", tilt), ("自建前 50 大市值加權", cw50), ("加強版 2/1/0（50）", tilt50s), ("大型股 AI 加權（100）", tilt100),
              ("自建前 100 大市值加權", cw100), ("核心衛星 0050×0.7＋A×0.3", core), ("0050 含息", b50), ("A 前 20% 等權", netA)]:
    line(nm, r.reindex(idx))
say(f"  換手率（月）：AI 加權 {turnT.iloc[1:].mean():.1%}、自建市值加權 {turnC.iloc[1:].mean():.1%}、加強版 {turnS.iloc[1:].mean():.1%}")
te = (cw50 - b50).dropna()
say(f"  自建前 50 大 − 0050：{te.mean():+.2%}／月（t={t_(te):+.2f}）、追蹤誤差 年化 {te.std() * np.sqrt(12):.1%}、相關 {cw50.reindex(idx).corr(b50):.3f}｜分年 {yearly(te)}")
d1 = (tilt - b50).dropna()
d2 = (tilt - cw50).dropna()
say(f"  AI 加權 − 0050 分年：{yearly(d1)}")
say(f"  AI 加權 − 自建市值加權 分年：{yearly(d2)}")
say(f"\n→ T1（大型股 AI 加權 − 0050 含息 > 0）：{d1.mean():+.3%}／月，t={t_(d1):+.2f}（贏 {(d1 > 0).mean():.0%} 的月） → {verdict(d1)}")
say(f"→ T2（大型股 AI 加權 − 自建前 50 大市值加權 > 0）：{d2.mean():+.3%}／月，t={t_(d2):+.2f}（贏 {(d2 > 0).mean():.0%} 的月） → {verdict(d2)}")
for nm, d in [("前 100 大 AI 加權 − 0050", tilt100 - b50), ("前 100 大 AI 加權 − 自建前 100 大", tilt100 - cw100),
              ("加強版 − 0050", tilt50s - b50), ("加強版 − 自建前 50 大", tilt50s - cw50), ("核心衛星 − 0050", core - b50)]:
    d = d.dropna()
    say(f"  （次要）{nm}：{d.mean():+.3%}／月（t={t_(d):+.2f}）")
for nm, d in [("AI 加權 − 0050", d1), ("AI 加權 − 自建市值加權", d2)]:
    d25 = d[pd.PeriodIndex(d.index).year == 2025]
    say(f"  2025 年 {nm}：{d25.sum():+.1%}（{len(d25)} 個月）")
ics = PT.groupby("ym").apply(lambda z: (lambda h: spearmanr(h.s_A, h.ret_next)[0])(z.nlargest(50, "mktcap")))
say(f"  A 分數在前 50 大內的月 Rank IC {ics.mean():+.4f}（t={t_(ics):+.1f}）")

pd.DataFrame({"A": netA, "A_inst": netI, "I2_trust30": netR, "ew": ew, "etf0050": b50, "tilt50": tilt, "cw50": cw50, "tilt50_strong": tilt50s,
              "tilt100": tilt100, "cw100": cw100, "core70_30": core}).to_csv(OUT + "inst_beat0050_returns.csv", encoding="utf-8-sig")
open(OUT + "inst_beat0050_backtest.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
