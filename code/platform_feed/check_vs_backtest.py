# -*- coding: utf-8 -*-
"""檢查：平台兩份名單的分組，和 H 節回測（oos2026_surprise.py）同一批事件的分組是否一致。"""
import sys
import pandas as pd
src = open("ai_strategies/oos2026_surprise.py", encoding="utf-8").read().split("def portP(")[0]
exec(src)
sys.path.insert(0, "platform_feed")
import earnings_signals  # noqa: E402
import revenue_signals  # noqa: E402
for nm, bt, fe, months in [("月營收", pub_rev, revenue_signals.build(log=lambda s: None), ["2026-08", "2026-09", "2026-10"]),
                           ("盈餘", pub_ern, earnings_signals.build(log=lambda s: None), ["2026-03", "2026-05", "2026-08"])]:
    for m in months:
        a = bt[bt.m == pd.Period(m, "M")][["code", "per", "dec"]]
        b = fe[fe.m == pd.Period(m, "M")][["code", "per", "dec"]]
        j = a.merge(b, on=["code", "per"], suffixes=("_bt", "_feed"))
        print(f"[{nm}] {m} 公布：回測 {len(a)} 筆（最高組 {(a.dec == 10).mean():.0%}）、平台 {len(b)} 筆（最高組 {(b.dec == 10).mean():.0%}）、對到 {len(j)}｜"
              f"分組相同 {(j.dec_bt == j.dec_feed).mean():.1%}｜最高組一致 {((j.dec_bt == 10) == (j.dec_feed == 10)).mean():.1%}｜"
              f"最低組一致 {((j.dec_bt == 1) == (j.dec_feed == 1)).mean():.1%}", flush=True)
