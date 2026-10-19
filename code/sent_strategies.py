# -*- coding: utf-8 -*-
"""
用 CMoney 淨情緒選股的 7 種策略（事先寫定見 PREREG_cmoney_full.md 第 F 節；清單、參數不增不改）。
CM_TAG=cm3 → 只有 2020–2024（樣本內）；CM_TAG=cm3x → 2020–2025，另外單獨報 2025（樣本外，主要檢定 S2、S7）。
python cmoney_full/sent_strategies.py
"""
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

TAG = os.environ.get("CM_TAG", "cm3")
END = "2025-12-31" if TAG == "cm3x" else "2024-12-31"
src = open("cmoney_full/lowfreq_cm.py", encoding="utf-8").read().split("ARMS = {")[0]
for a, b in [('CM_TAG = os.environ.get("CM_TAG", "cm3")', f'CM_TAG = "{TAG}"'),
             ('(df.Date <= "2024-12-31")', f'(df.Date <= "{END}")')]:
    assert a in src
    src = src.replace(a, b)
exec(src)
me = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=["stock_code", "Date", "margin_eligible", "in265"], parse_dates=["Date"])
me = me[me.in265 == 1]
C_LONG, C_SHORT = 0.00471, 0.00471 + 0.0008
L = []


def say2(s=""):
    print(s, flush=True)
    L.append(s)


def t_(s):
    s = pd.Series(s).dropna()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 else np.nan


def model_scores(P, test_years):
    valid = P.ret_next.notna() & P.tgt.notna()
    sc = pd.Series(np.nan, index=P.index)
    for Y in test_years:
        tr, va, te = valid & (P.year <= Y - 2), valid & (P.year == Y - 1), valid & (P.year == Y)
        if te.sum() == 0 or tr.sum() == 0:
            continue
        ps = []
        for s in range(3):
            m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=5, early_stopping_rounds=30, tree_method="hist", n_jobs=8, random_state=s, verbosity=0)
            m.fit(P.loc[tr, BASE], P.loc[tr, "tgt"], eval_set=[(P.loc[va, BASE], P.loc[va, "tgt"])], verbose=False)
            ps.append(m.predict(P.loc[te, BASE]))
        sc[te] = np.mean(ps, axis=0)
    return sc


def run(freq, nm, unit, py):
    P = panel(freq)
    m2 = me.copy()
    m2["per"] = m2.Date.dt.to_period(freq)
    m2 = m2.drop_duplicates(["stock_code", "per"], keep="last")[["stock_code", "per", "margin_eligible"]]
    P = P.merge(m2, on=["stock_code", "per"], how="left")
    yrs = sorted(P.year.unique())
    P["score"] = model_scores(P, [y for y in yrs if y >= 2022])
    U = P[P.ret_next.notna() & P.liq_ok].copy()
    ok3 = U.cm_net.notna() & ((U.cm_pos + U.cm_neg) >= 3)
    U["q"] = np.nan
    U.loc[ok3, "q"] = U[ok3].groupby("per").cm_net.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1 if len(x) >= 10 else np.nan)
    U["chg_top"] = False
    okc = ok3 & U.cm_net_chg.notna()
    U.loc[okc, "chg_top"] = U[okc].groupby("per").cm_net_chg.rank(pct=True) > 0.8
    U["mtop"] = U.groupby("per").score.rank(pct=True, ascending=False) <= 0.2
    U["w_cap"] = U.mktcap.fillna(0)
    pers = sorted(U.per.unique())
    G = {p: z for p, z in U.groupby("per")}
    ew = pd.Series({p: G[p].ret_next.mean() for p in pers})
    cw = pd.Series({p: np.average(G[p].ret_next, weights=G[p].w_cap + 1e-9) for p in pers})
    sel = {
        "S1 只買最樂觀 Q5": lambda z: z[z.q == 5],
        "S2 避開清單（剔除 Q1，等權）": lambda z: z[z.q != 1],
        "S4 情緒轉好前 20%": lambda z: z[z.chg_top],
        "S5 反向：只買最悲觀 Q1": lambda z: z[z.q == 1],
        "S6 模型前 20% 剔除 Q1": lambda z: z[z.mtop & (z.q != 1) & z.score.notna()],
        "基準模型前 20%（S6 的對照）": lambda z: z[z.mtop & z.score.notna()],
    }
    res, prev = {}, {}
    for name, f in sel.items():
        r, turn, held = {}, {}, None
        for p in pers:
            h = f(G[p])
            if len(h) == 0:
                continue
            r[p] = h.ret_next.mean()
            s = set(h.stock_code)
            turn[p] = 1.0 if held is None else len(s - held) / len(s)
            held = s
        r, turn = pd.Series(r), pd.Series(turn)
        res[name] = r - turn * C_LONG
    # S7 市值加權避開清單（換股以權重計：新進名單的權重）
    r, turn, held = {}, {}, None
    for p in pers:
        h = G[p][G[p].q != 1]
        w = h.w_cap / h.w_cap.sum()
        r[p] = float((w * h.ret_next).sum())
        turn[p] = 1.0 if held is None else float(w[~h.stock_code.isin(held)].sum())
        held = set(h.stock_code)
    res["S7 避開清單（剔除 Q1，市值加權）"] = pd.Series(r) - pd.Series(turn) * C_LONG
    # S3 多空
    r, tl, ts, hl, hs = {}, {}, {}, None, None
    for p in pers:
        z = G[p]
        lo, sh = z[z.q == 5], z[(z.q == 1) & (z.margin_eligible == 1)]
        if len(lo) == 0 or len(sh) == 0:
            continue
        r[p] = (lo.ret_next.mean() - sh.ret_next.mean()) / 2
        sl, ss = set(lo.stock_code), set(sh.stock_code)
        tl[p] = 1.0 if hl is None else len(sl - hl) / len(sl)
        ts[p] = 1.0 if hs is None else len(ss - hs) / len(ss)
        hl, hs = sl, ss
    res["S3 多空（買 Q5、空 Q1）"] = pd.Series(r) - (pd.Series(tl) * C_LONG + pd.Series(ts) * C_SHORT) / 2
    bench_of = {k: ew for k in res}
    bench_of["S7 避開清單（剔除 Q1，市值加權）"] = cw
    bench_of["S3 多空（買 Q5、空 Q1）"] = pd.Series(0.0, index=ew.index)
    bench_of["S6 模型前 20% 剔除 Q1"] = res["基準模型前 20%（S6 的對照）"]
    bname = {k: "等權" for k in res}
    bname.update({"S7 避開清單（剔除 Q1，市值加權）": "市值加權", "S3 多空（買 Q5、空 Q1）": "0", "S6 模型前 20% 剔除 Q1": "模型前20%"})

    e50 = pd.read_csv("quarterly_vol/etf0050_adj.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()
    e50 = e50.groupby(e50.index.to_period(freq)).last()
    r50 = (e50.shift(-1) / e50 - 1).reindex(ew.index)                       # 以形成期為索引：下一期 0050 含息報酬
    pd.DataFrame({k: v for k, v in res.items()} | {"等權": ew, "市值加權": cw, "0050含息": r50}).to_csv(f"cmoney_full/sent_strategies_{TAG}_{freq}.csv", encoding="utf-8-sig")

    def seg(lbl, mask_fn):
        m50 = r50[mask_fn(r50.index)].dropna()
        say2(f"  【{lbl}】0050 含息 年化 {(1 + m50).prod() ** (py / len(m50)) - 1:+.1%}｜等權 年化 {(1 + ew[mask_fn(ew.index)]).prod() ** (py / mask_fn(ew.index).sum()) - 1:+.1%}｜市值加權 {(1 + cw[mask_fn(cw.index)]).prod() ** (py / mask_fn(cw.index).sum()) - 1:+.1%}")
        out = {}
        for k in ["S1 只買最樂觀 Q5", "S2 避開清單（剔除 Q1，等權）", "S3 多空（買 Q5、空 Q1）", "S4 情緒轉好前 20%", "S5 反向：只買最悲觀 Q1",
                  "S6 模型前 20% 剔除 Q1", "S7 避開清單（剔除 Q1，市值加權）"]:
            r = res[k][mask_fn(res[k].index)]
            if len(r) < 3:
                say2(f"    {k}：期數不足")
                continue
            ex = (r - bench_of[k].reindex(r.index)).dropna()
            ann = (1 + r).prod() ** (py / len(r)) - 1
            e5 = (r - r50.reindex(r.index)).dropna()
            say2(f"    {k:22s} 扣成本年化 {ann:+6.1%}｜相對{bname[k]} {ex.mean():+.2%}/{unit}（t={t_(ex):+.2f}，贏 {(ex > 0).mean():.0%}，{len(ex)} 期）"
                 + (f"｜相對 0050 含息 {e5.mean():+.2%}（t={t_(e5):+.2f}）" if k.startswith(("S2", "S7")) else ""))
            out[k] = ex
        return out

    say2(f"\n===== {nm}（每期可買約 {U.groupby('per').size().mean():.0f} 檔；有 ≥3 篇表態而能分組的約 {U[ok3].groupby('per').size().mean():.0f} 檔）=====")
    hold_year = lambda idx: np.array([(p + 1).end_time.year for p in idx])  # noqa: E731  持有期（形成期的下一期）結束在哪一年
    seg("2020–2024 樣本內（S6 只有 2022–2024）", lambda idx: hold_year(idx) <= 2024)
    oos = None
    if TAG == "cm3x":
        oos = seg("2025 樣本外（持有期在 2025）", lambda idx: hold_year(idx) == 2025)
    return oos


o_m = run("M", "月頻（主要）", "月", 12)
run("W-FRI", "週頻（只報數字）", "週", 52)
run("Q", "季頻（只報數字）", "季", 4)
if o_m is not None:
    say2("\n主要檢定（2025 月頻；Bonferroni 單尾 t > 2.20）")
    for k in ["S2 避開清單（剔除 Q1，等權）", "S7 避開清單（剔除 Q1，市值加權）"]:
        ex = o_m.get(k, pd.Series(dtype=float))
        t = t_(ex)
        say2(f"→ {k}：{ex.mean():+.2%}/月，t={t:+.2f}，單尾 p={stats.t.sf(t, len(ex) - 1):.3f} → {'通過' if ex.mean() > 0 and t > 2.20 else ('方向一致但不顯著' if ex.mean() > 0 else '未通過（方向相反）')}")
open(f"cmoney_full/sent_strategies_{TAG}.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
