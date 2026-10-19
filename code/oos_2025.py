# -*- coding: utf-8 -*-
"""
2025 年真正樣本外：CMoney 淨情緒對下期報酬（事先寫定見 PREREG_cmoney_full.md 第 E 節）。
變數、分組、Fama-MacBeth 完全照 lowfreq_net_sort.py／lowfreq_fm_check.py；資料 final_dataset_v3_cm3x（2025＝build_2025.py）。
樣本外＝下期報酬落在 2025 年的期間：月頻形成月 2024-12 ~ 2025-11（12 期）；週頻次要。
python cmoney_full/oos_2025.py
"""
import numpy as np
import pandas as pd
from scipy import stats

src = open("cmoney_full/lowfreq_cm.py", encoding="utf-8").read().split("BASE = [")[0]
for a, b in [('CM_TAG = os.environ.get("CM_TAG", "cm3")', 'CM_TAG = "cm3x"'),
             ('(df.Date <= "2024-12-31")', '(df.Date <= "2025-12-31")')]:
    assert a in src
    src = src.replace(a, b)
exec(src)
L = []
T1 = 1.80                                        # 單尾 p<0.05、df=11


def say2(s):
    print(s, flush=True)
    L.append(s)


def t_(s):
    s = pd.Series(s).dropna()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


CTRL = ["logcap", "logdv", "vol20", "max20", "m1", "m2", "m3", "smr"]


def analyse(freq, nm, unit, oos_start, oos_end, py):
    P = panel(freq)
    P["logcap"] = np.log(P.mktcap)
    U = P[P.ret_next.notna() & P.liq_ok].copy()
    O = U[(U.per >= pd.Period(oos_start, freq)) & (U.per <= pd.Period(oos_end, freq))].copy()
    bench = O.groupby("per").ret_next.mean()
    S = O[O.cm_net.notna() & ((O.cm_pos + O.cm_neg) >= 3)].copy()
    S["q"] = S.groupby("per").cm_net.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1 if len(x) >= 10 else np.nan)
    R = S.groupby(["per", "q"]).ret_next.mean().unstack()
    ls = R[5] - R[1]
    held = S[S.q == 5].groupby("per").stock_code.apply(set)
    turn = pd.Series([1.0] + [len(held.iloc[i] - held.iloc[i - 1]) / max(len(held.iloc[i]), 1) for i in range(1, len(held))], index=held.index)
    net5 = R[5] - turn.reindex(R.index).fillna(1) * 0.00471
    ex = net5 - bench.reindex(R.index)
    coefs, coefs_att = [], []
    for _, z in O.groupby("per"):
        for X, store in [(["cm_net"], coefs), (["cm_att"], coefs_att)]:
            cols = X + CTRL
            zz = z[cols + ["ret_next"]].dropna()
            if len(zz) < 50:
                continue
            Z = zz[cols].rank(pct=True)
            Z = (Z - Z.mean()) / Z.std()
            yv = zz.ret_next.rank(pct=True) - 0.5
            A = np.column_stack([np.ones(len(Z)), Z.values])
            store.append(np.linalg.lstsq(A, yv.values, rcond=None)[0][1])
    c, ca = pd.Series(coefs), pd.Series(coefs_att)
    say2(f"\n===== {nm} 樣本外（形成 {oos_start} ~ {oos_end}，{len(R)} 期；每期約 {S.groupby('per').size().mean():.0f} 檔有 ≥3 篇表態）=====")
    say2(f"  下期報酬 Q1(最悲觀)…Q5(最樂觀) = " + " ".join(f"{R[k].mean():+.2%}" for k in range(1, 6)))
    say2(f"  Q5−Q1 {ls.mean():+.2%}/{unit}（t={t_(ls):+.2f}，單尾 p={stats.t.sf(t_(ls), len(ls.dropna()) - 1):.3f}；為正 {(ls > 0).sum()}/{ls.count()} 期）")
    say2(f"  Fama-MacBeth（控制 8 個變數）淨情緒 {c.mean():+.4f}（t={t_(c):+.2f}，單尾 p={stats.t.sf(t_(c), len(c) - 1):.3f}；為正 {(c > 0).sum()}/{len(c)}）｜討論量 {ca.mean():+.4f}（t={t_(ca):+.2f}）")
    say2(f"  只買 Q5：換股 {turn.iloc[1:].mean():.0%}/{unit}、扣成本 {net5.mean():+.2%}/{unit}（年化 {(1 + net5).prod() ** (py / len(net5)) - 1:+.1%}）｜等權 {bench.mean():+.2%}/{unit}｜相對等權 {ex.mean():+.2%}（t={t_(ex):+.2f}）")
    if freq == "M":
        say2("  逐月 Q5−Q1：" + " ".join(f"{str(p)[2:]}:{v:+.1%}" for p, v in ls.items()))
    return ls, c


ls_m, c_m = analyse("M", "月頻（主要）", "月", "2024-12", "2025-11", 12)
ls_w, c_w = analyse("W-FRI", "週頻（次要）", "週", "2024-12-27", "2025-12-19", 52)
o1 = ls_m.mean() > 0 and t_(ls_m) > T1
o2 = c_m.mean() > 0 and t_(c_m) > T1
say2(f"\n→ O1（月頻 Q5−Q1 > 0，單尾 t>{T1}）：{ls_m.mean():+.2%}/月，t={t_(ls_m):+.2f} → {'通過' if o1 else ('方向一致但不顯著' if ls_m.mean() > 0 else '未通過（方向相反）')}")
say2(f"→ O2（月頻 FM 淨情緒 > 0，單尾 t>{T1}）：{c_m.mean():+.4f}，t={t_(c_m):+.2f} → {'通過' if o2 else ('方向一致但不顯著' if c_m.mean() > 0 else '未通過（方向相反）')}")
say2(f"→ 樣本外結論：{'成立（O1、O2 都通過）' if o1 and o2 else ('方向一致但證據不足' if ls_m.mean() > 0 and c_m.mean() > 0 else '不成立')}")
say2(f"  （次要）週頻：Q5−Q1 {ls_w.mean():+.2%}/週（t={t_(ls_w):+.2f}）｜FM {c_w.mean():+.4f}（t={t_(c_w):+.2f}）")
open("cmoney_full/oos_2025.log", "w", encoding="utf-8").write("\n".join(L) + "\n")
