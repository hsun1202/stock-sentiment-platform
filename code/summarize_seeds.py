# -*- coding: utf-8 -*-
"""
彙整多組種子的「含情緒 vs 不含情緒」LSTM整體MAPE，算平均差異、95%信賴區間、配對t檢定、Wilcoxon。
python summarize_seeds.py --dir lstm_stitched_seeds --out summary_seeds_stitched.json
"""
import os
import re
import json
import argparse
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="lstm_stitched_seeds")
ap.add_argument("--out", default="summary_seeds_stitched.json")
args = ap.parse_args()

rows = []
for name in sorted(os.listdir(args.dir)):
    m = re.fullmatch(r"seed(\d+)_with", name)
    if not m:
        continue
    seed = int(m.group(1))
    pw = os.path.join(args.dir, name, "metrics.json")
    pn = os.path.join(args.dir, f"seed{seed}_without", "metrics.json")
    if not (os.path.exists(pw) and os.path.exists(pn)):
        continue
    w = json.load(open(pw, encoding="utf-8"))["overall_mape"] * 100
    n = json.load(open(pn, encoding="utf-8"))["overall_mape"] * 100
    rows.append({"seed": seed, "with": w, "without": n, "diff": w - n})

d = np.array([r["diff"] for r in rows])
n = len(d)
mean, sd = d.mean(), d.std(ddof=1)
se = sd / np.sqrt(n)
tcrit = stats.t.ppf(0.975, n - 1)
t_stat, p_t = stats.ttest_1samp(d, 0.0)
try:
    p_w = stats.wilcoxon(d).pvalue
except ValueError:
    p_w = float("nan")
res = {
    "n": n, "rows": rows, "mean_diff": mean, "std_diff": sd, "se_diff": se,
    "ci95": [mean - tcrit * se, mean + tcrit * se], "t_stat": float(t_stat), "p_ttest": float(p_t),
    "p_wilcoxon": float(p_w), "n_improve": int((d < 0).sum()), "n_worse": int((d > 0).sum()),
    "mean_with": float(np.mean([r["with"] for r in rows])), "mean_without": float(np.mean([r["without"] for r in rows])),
}
json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"種子數 n={n}")
print(f"含情緒 平均MAPE {res['mean_with']:.3f}%   不含情緒 平均MAPE {res['mean_without']:.3f}%")
print(f"平均差異(含-不含) {mean:+.3f} pp  (負=情緒有幫助)   95%CI [{res['ci95'][0]:+.3f}, {res['ci95'][1]:+.3f}]")
print(f"配對t檢定 t={t_stat:.3f} p={p_t:.3f}   Wilcoxon p={p_w:.3f}   進步 {res['n_improve']} 組 / 退步 {res['n_worse']} 組")
print("結論:", "95%CI涵蓋0，無統計顯著差異" if res["ci95"][0] <= 0 <= res["ci95"][1] else ("情緒因子顯著改善" if mean < 0 else "情緒因子顯著變差"))
