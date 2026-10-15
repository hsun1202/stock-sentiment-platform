# -*- coding: utf-8 -*-
"""
樣本外檢查：沒收漲停的大漲日之後會不會回吐？（2001~2013，完全沒看過；這段期間沒有新聞資料，所以只能驗「不分有沒有消息」的版本）
背景：newsmove_backtest.py（2015~2025）發現——剔除收漲停的之後，沒消息的大漲日 5 天後超額 −0.6%（兩組股票、11 年每年為負），有消息的約 0。
      2015~2025 全體（有消息約占兩成）加權後約 −0.45%。
事先寫定的假說（2026-10-10，看 2001~2013 的結果之前寫下；1 個主要檢定，|t|>2）：
  R1：2001~2013，大漲日（|超額|>自己 60 天標準差 3 倍且>3%、超額>0）且當天沒有收漲停，收盤買進持有 5 個交易日的累積超額報酬 < 0。
  次要：收漲停的大漲日之後 5 天 > 0（漲停鎖住的延續）；大跌且沒收跌停的之後 5 天（方向不預設）。
python bigup_reversal_pre2014.py
"""
import numpy as np, pandas as pd
p = pd.read_pickle("tej_raw/tej_price_daily_pre2014.pkl")
p = p[p.date <= "2013-12-31"].sort_values(["code", "date"]).reset_index(drop=True)
d = pd.DataFrame({"code": p.code, "date": p.date, "r": p["報酬率％"] / 100, "lim": p["漲跌停"].astype(str).str.strip()})
d["x"] = d.r - d.groupby("date").r.transform("mean")
g = d.groupby("code")
d["sd"] = g.x.transform(lambda s: s.rolling(60, min_periods=30).std().shift(1))
d["big"] = ((d.x / d.sd).abs() > 3) & (d.x.abs() > 0.03)
d["post5"] = sum(g.x.shift(-k) for k in range(1, 6))
d["ym"], d["year"] = d.date.dt.to_period("M"), d.date.dt.year
def tm(s):
    z = s[["post5", "ym"]].dropna(); mm = z.groupby("ym").post5.mean()
    return z.post5.mean(), mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm))), len(z)
out = []
B = d[d.big & d.sd.notna()]
out.append(f"2001~2013：大漲日 {int((B.x>0).sum()):,}（收漲停 {(B[B.x>0].lim=='+').mean():.1%}）、大跌日 {int((B.x<0).sum()):,}（收跌停 {(B[B.x<0].lim=='-').mean():.1%}）")
for lab, s in [("大漲・沒收漲停★R1", B[(B.x > 0) & (B.lim != "+")]), ("大漲・收漲停", B[(B.x > 0) & (B.lim == "+")]),
               ("大跌・沒收跌停", B[(B.x < 0) & (B.lim != "-")]), ("大跌・收跌停", B[(B.x < 0) & (B.lim == "-")])]:
    a, t, n = tm(s); yr = s.groupby("year").post5.mean()
    out.append(f"  {lab:<12s} n={n:,}｜之後 5 天超額 {a:+.2%}（t={t:+.1f}）｜為負的年數 {(yr<0).sum()}/{len(yr)}｜分年 " + " ".join(f"{y}:{v:+.1%}" for y, v in yr.items()))
a, t, n = tm(B[(B.x > 0) & (B.lim != "+")])
out.append(f"→ R1：{a:+.2%}，t={t:+.2f} → {'通過' if (a < 0 and t < -2) else '未通過'}")
print("\n".join(out)); open("bigup_reversal_pre2014.log", "w", encoding="utf-8").write("\n".join(out) + "\n")
