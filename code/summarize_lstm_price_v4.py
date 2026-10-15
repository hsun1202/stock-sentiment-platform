# -*- coding: utf-8 -*-
"""彙總 run_lstm_price_v4.sh：各組 MAPE / 方向準確率（10 seeds），與「技術面」配對比較（同種子相減），並列天真預測。"""
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

rows = [json.load(open(f, encoding="utf-8")) for f in glob.glob("lstm_price_v4/*/metrics.json")]
m = pd.DataFrame(rows)
lines = [f"完成 {len(m)} 次（組×種子）；天真預測（拿今天收盤當明天）MAPE {m.naive_mape.mean():.2%}"]
for arm, g in m.groupby("arm"):
    lines.append(f"{arm:10s} MAPE {g.mape.mean():.2%}±{g.mape.std(ddof=1):.2%}  方向準確率 {g.dir_acc.mean():.1%}  （{len(g)} seeds）")
base = m[m.arm == "技術面"].set_index("seed")
out = {}
for arm in ("+PTT", "+融資融券", "+PTT+融資融券"):
    g = m[m.arm == arm].set_index("seed")
    s = sorted(set(g.index) & set(base.index))
    for met, name in (("mape", "MAPE"), ("dir_acc", "方向準確率")):
        d = g.loc[s, met] - base.loc[s, met]
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
        p = stats.ttest_1samp(d, 0).pvalue
        out[f"{arm}_{met}"] = {"diff": float(d.mean()), "ci95": [float(ci[0]), float(ci[1])], "p": float(p), "n": len(d)}
        lines.append(f"  {arm} vs 技術面 {name}差 {d.mean()*100:+.3f}pp  95%CI[{ci[0]*100:+.3f},{ci[1]*100:+.3f}]  p={p:.3f}  進步{int((d < 0).sum() if met == 'mape' else (d > 0).sum())}/{len(d)}")
print("\n".join(lines))
json.dump({"arms": m.groupby("arm")[["mape", "dir_acc", "naive_mape"]].mean().to_dict(), "paired": out},
          open("summary_lstm_price_v4.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
