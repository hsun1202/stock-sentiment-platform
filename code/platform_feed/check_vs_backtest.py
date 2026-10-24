# -*- coding: utf-8 -*-
"""檢查：平台名單的分組是否和 H 節回測（oos2026_surprise.py）同一批事件的分組一致。"""
import sys
import pandas as pd
src = open("ai_strategies/oos2026_surprise.py", encoding="utf-8").read().split("def portP(")[0]
exec(src)
sys.path.insert(0, "platform_feed")
import revenue_signals  # noqa: E402
fe = revenue_signals.build(log=lambda s: None)
for m in ["2026-08", "2026-09", "2026-10"]:
    a = pub_rev[pub_rev.m == pd.Period(m, "M")][["code", "per", "dec"]]
    b = fe[fe.m == pd.Period(m, "M")][["code", "per", "dec"]]
    j = a.merge(b, on=["code", "per"], suffixes=("_bt", "_feed"))
    print(f"{m} 公布：回測 {len(a)} 筆、平台 {len(b)} 筆、對到 {len(j)}｜分組完全相同 {(j.dec_bt == j.dec_feed).mean():.1%}｜"
          f"最高組一致 {((j.dec_bt == 10) == (j.dec_feed == 10)).mean():.1%}｜最低組一致 {((j.dec_bt == 1) == (j.dec_feed == 1)).mean():.1%}", flush=True)
