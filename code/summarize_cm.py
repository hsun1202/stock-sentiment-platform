# -*- coding: utf-8 -*-
"""彙總 cmoney_full/lstm/ 的 per-stock LSTM 結果（每檔先對 seeds 平均，再跨股票做配對 t），寫 cmoney_full/lstm_summary.log"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

D = os.environ.get("LSTM_DIR", "cmoney_full/lstm")
rows = []
for f in glob.glob(f"{D}/*_seed*/per_stock.csv"):
    arm, seed = os.path.basename(os.path.dirname(f)).rsplit("_seed", 1)
    rows.append(pd.read_csv(f, dtype={"stock_code": str}).assign(arm=arm, seed=int(seed)))
R = pd.concat(rows)
S = R.groupby(["arm", "stock_code"])[["mape", "naive_mape", "dir_acc"]].mean().reset_index()
out = [f"每組 seeds：{R.groupby('arm').seed.nunique().to_dict()}｜股票 {S.stock_code.nunique()} 檔｜測試期起點中位數 {R.test_start.median() if R.test_start.dtype != object else sorted(R.test_start)[len(R)//2]}"]
for arm in [a for a in ["技術面", "+PTT", "+CMoney", "+PTT+CMoney", "+CMoney收盤前"] if a in set(S.arm)]:
    s = S[S.arm == arm]
    out.append(f"{arm:12s} MAPE {s.mape.mean():.2%}｜天真 {s.naive_mape.mean():.2%}｜方向準確率 {s.dir_acc.mean():.1%}｜贏天真的股票 {(s.mape < s.naive_mape).mean():.0%}")
W = S.pivot(index="stock_code", columns="arm", values=["mape", "dir_acc"])
for arm in [a for a in ["+PTT", "+CMoney", "+PTT+CMoney", "+CMoney收盤前"] if a in set(S.arm)]:
    d = (W["mape"][arm] - W["mape"]["技術面"]).dropna()
    da = (W["dir_acc"][arm] - W["dir_acc"]["技術面"]).dropna()
    t, p = stats.ttest_1samp(d, 0)
    ta, pa = stats.ttest_1samp(da, 0)
    out.append(f"{arm} − 技術面：MAPE {d.mean()*100:+.2f}pp（t={t:+.2f}, p={p:.3f}，變好的股票 {(d < 0).mean():.0%}）｜方向準確率 {da.mean()*100:+.2f}pp（p={pa:.3f}）")
d = (W["mape"]["+CMoney"] - W["mape"]["+PTT"]).dropna()
out.append(f"+CMoney − +PTT：MAPE {d.mean()*100:+.2f}pp（p={stats.ttest_1samp(d, 0)[1]:.3f}）")
d = (W["mape"]["+CMoney"] - W["mape"]["技術面"]).dropna()
t, p = stats.ttest_1samp(d, 0)
out.append(f"→ L1（+CMoney MAPE 低於技術面，p<0.05）：{d.mean()*100:+.2f}pp, p={p:.3f} → {'通過' if d.mean() < 0 and p < 0.05 else '未通過'}")
print("\n".join(out))
open(f"{D}_summary.log", "w", encoding="utf-8").write("\n".join(out) + "\n")
