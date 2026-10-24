# -*- coding: utf-8 -*-
"""
I 節：G 節兩個策略的穩健性與對 0050 的 α／β（看過 G、H 結果後才加，只報數字）。
參考組員 jameschen108/tw-stock-quant 的 robust.py：成本加倍、持有天數、門檻、分段、對 0050 的 α／β（Newey-West t，lag 3）。
cd D:/專題claude && PYTHONIOENCODING=utf-8 python ai_strategies/surprise_robust.py
"""
import numpy as np
import pandas as pd

src = open("ai_strategies/surprise_event_backtest.py", encoding="utf-8").read().split("\nres = {}")[0]
exec(src)                                   # ev_rev、ev_ern（含 x、dec）、port()、ewm、b50、t_、perf、COST
L.clear()
BASE_COST = COST


def nw_ab(y, x, lag=3):
    """y = α + β x，回傳（α 年化、α 的 Newey-West t、β）"""
    d = pd.concat([y, x], axis=1).dropna()
    Y, X = d.iloc[:, 0].values, np.column_stack([np.ones(len(d)), d.iloc[:, 1].values])
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        G = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None])
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    return b[0] * 12, b[0] / np.sqrt(V[0, 0]), b[1]


def topq(ev, q):
    """最高 q 比例（門檻＝前 12 個月事件 x 的 1−q 分位數）"""
    th = {}
    for m in ev.m.unique():
        past = ev.x[(ev.m < m) & (ev.m >= m - 12)]
        th[m] = np.quantile(past, 1 - q) if len(past) >= 200 else np.inf      # 同 G 節：前 12 個月至少 200 筆才分組
    return ev[ev.x > ev.m.map(th)]


def row(lab, net):
    d = (net - ewm).dropna()
    a, at, b = nw_ab(net, b50)
    ann = perf(net)[0]
    say(f"  {lab:<22} 年化 {ann:+.1%}｜− 一般股票 {d.mean():+.2%}／月（t={t_(d):+.2f}）｜對 0050：α {a:+.1%}／年（NW t={at:+.2f}）、β {b:.2f}")


for nm, ev, H, alts in [("月營收驚喜", ev_rev, 20, (10, 40)), ("盈餘驚喜", ev_ern, 60, (40, 90))]:
    top = ev[ev.dec == 10]
    say(f"\n===== {nm}（2018–2025；只報數字）=====")
    base, _, _ = port([(top, H, 0)])
    row(f"原設定（前 10%、{H} 天）", base)
    globals()["COST"] = BASE_COST * 2
    row("成本 ×2", port([(top, H, 0)])[0])
    globals()["COST"] = BASE_COST
    for h in alts:
        row(f"持有 {h} 天", port([(top, h, 0)])[0])
    for q in (0.05, 0.20):
        row(f"前 {q:.0%}", port([(topq(ev, q), H, 0)])[0])
    say("  分段（原設定）：")
    for lab, a0, a1 in [("2018–2021", 2018, 2021), ("2022–2023", 2022, 2023), ("2024–2025", 2024, 2025)]:
        k = (base.index.year >= a0) & (base.index.year <= a1)
        d = (base - ewm)[k].dropna()
        a, at, b = nw_ab(base[k], b50[k])
        d50 = (base - b50)[k].dropna()
        say(f"    {lab}  年化 {perf(base[k])[0]:+.1%}（0050 {perf(b50[k])[0]:+.1%}）｜− 一般股票 {d.mean():+.2%}／月（t={t_(d):+.2f}）｜"
            f"− 0050 {d50.mean():+.2%}／月（t={t_(d50):+.2f}）｜α {a:+.1%}（t={at:+.2f}）、β {b:.2f}")

say("\n===== 2026-01～09（H 節公開資料的月報酬）=====")
o = pd.read_csv("ai_strategies/oos2026_surprise_returns.csv", index_col=0)
o.index = pd.PeriodIndex(o.index, freq="M")
o = o[o.index <= pd.Period("2026-09", "M")]
for nm, c in [("月營收驚喜", "rev_top10"), ("盈餘驚喜", "ern_top10")]:
    a, at, b = nw_ab(o[c], o.etf0050, lag=2)
    say(f"  {nm}：累積 {(1 + o[c]).prod() - 1:+.1%}（0050 {(1 + o.etf0050).prod() - 1:+.1%}）｜α {a:+.1%}／年（NW t={at:+.2f}）、β {b:.2f}（9 個月，僅供參考）")
open(OUT + "surprise_robust.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
