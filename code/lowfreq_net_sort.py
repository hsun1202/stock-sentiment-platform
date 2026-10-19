# -*- coding: utf-8 -*-
"""事後探索：依 CMoney 當期淨情緒分 5 組，看下期報酬（不用模型、不需訓練，所以 2020–2024 全期都能看；但規則是看到 IC 後才挑的，要標明）。"""
import os, sys
import numpy as np, pandas as pd
src = open("cmoney_full/lowfreq_cm.py", encoding="utf-8").read().split("BASE = [")[0]
exec(src)
L = []
def say2(s): print(s, flush=True); L.append(s)
for freq, nm, py, unit in [("W-FRI", "週頻", 52, "週"), ("M", "月頻", 12, "月"), ("Q", "季頻", 4, "季")]:
    P = panel(freq)
    U = P[P.ret_next.notna() & P.liq_ok].copy()
    bench = U.groupby("per").ret_next.mean()
    S = U[U.cm_net.notna() & ((U.cm_pos + U.cm_neg) >= 3)].copy()          # 至少 3 篇表態，淨情緒才有意義
    S["q"] = S.groupby("per").cm_net.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1 if len(x) >= 10 else np.nan)
    R = S.groupby(["per", "q"]).ret_next.mean().unstack()
    held = S[S.q == 5].groupby("per").stock_code.apply(set)
    turn = pd.Series([1.0] + [len(held.iloc[i] - held.iloc[i - 1]) / max(len(held.iloc[i]), 1) for i in range(1, len(held))], index=held.index)
    net5 = R[5] - turn.reindex(R.index).fillna(1) * 0.00471
    ex = net5 - bench.reindex(R.index)
    ls = R[5] - R[1]
    t = lambda s: s.mean() / (s.std(ddof=1) / np.sqrt(s.count()))
    say2(f"{nm}（每期平均 {S.groupby('per').size().mean():.0f} 檔有 ≥3 篇表態，共 {len(R)} 期）：下期報酬 Q1(最悲觀)…Q5(最樂觀) = " + " ".join(f"{R[k].mean():+.2%}" for k in range(1, 6)))
    say2(f"  Q5−Q1 {ls.mean():+.2%}/{unit}（t={t(ls):+.2f}）｜Q5 每{unit}換股 {turn.iloc[1:].mean():.0%}，扣成本 {net5.mean():+.2%}/{unit} → 年化 {(1 + net5).prod() ** (py / len(net5)) - 1:+.1%}｜相對等權 {ex.mean():+.2%}/{unit}（t={t(ex):+.2f}）")
    yr = ex.groupby(pd.PeriodIndex(ex.index).year).mean() * py
    say2("  分年（Q5 扣成本相對等權，年化）：" + " ".join(f"{y}:{v:+.1%}" for y, v in yr.items()))
open("cmoney_full/lowfreq_net_sort.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
