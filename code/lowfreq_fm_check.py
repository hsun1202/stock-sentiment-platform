# -*- coding: utf-8 -*-
"""事後探索（看到 lowfreq_cm 單一變數 IC 後才追加）：CMoney 討論量／淨情緒對下期報酬的效果，控制規模、成交值、波動、動能後還在不在？
每期橫斷面迴歸（變數先轉成橫斷面排名再標準化），Fama-MacBeth 平均係數與 t。"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.getcwd(), "cmoney_full"))
import builtins
_open = builtins.open
src = _open("cmoney_full/lowfreq_cm.py", encoding="utf-8").read().split("BASE = [")[0]
exec(src)
L = []
def say2(s): print(s, flush=True); L.append(s)
for freq, nm in [("W-FRI", "週頻"), ("M", "月頻")]:
    P = panel(freq)
    P["logcap"] = np.log(P.mktcap)
    U = P[P.ret_next.notna() & P.liq_ok & (P.year >= 2020)].copy()
    CTRL = ["logcap", "logdv", "vol20", "max20", "m1", "m2", "m3", "smr"]
    for X in [["cm_att"], ["cm_net"], ["cm_att", "cm_net"]]:
        for ctrl in [[], CTRL]:
            cols = X + ctrl
            coefs = []
            for _, z in U.groupby("per"):
                z = z[cols + ["ret_next"]].dropna()
                if len(z) < 50: continue
                Z = z[cols].rank(pct=True); Z = (Z - Z.mean()) / Z.std()
                yv = z.ret_next.rank(pct=True) - 0.5
                A = np.column_stack([np.ones(len(Z)), Z.values])
                b = np.linalg.lstsq(A, yv.values, rcond=None)[0][1:1 + len(X)]
                coefs.append(b)
            c = np.array(coefs)
            t = c.mean(0) / (c.std(0, ddof=1) / np.sqrt(len(c)))
            say2(f"{nm} {'+'.join(X):14s} {'控制規模/成交值/波動/動能/券資比' if ctrl else '不控制':18s}：" + "｜".join(f"{x} {m:+.4f}（t={tt:+.2f}）" for x, m, tt in zip(X, c.mean(0), t)) + f"｜{len(c)} 期")
_open("cmoney_full/lowfreq_fm_check.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
