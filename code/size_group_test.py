# -*- coding: utf-8 -*-
"""
依規模分組檢驗「PTT 在小型股有沒有增量」（日頻版；用 daytrade_test.py 已存好的 XGBoost 預測，不重訓）
  規模代理：60日成交值中位數（close_unadj × 成交量；Yahoo 成交量壞值當缺值），每天橫斷面分三組：0=最小、1=中、2=最大。
  目標：B＝隔天開盤→收盤、B2＝隔兩天開盤→收盤（同 daytrade_test.py 的設定，測試年 2018~2024）。
  各組 ΔAUC＝(加入某訊號) − (只有技術面)；週 block bootstrap；彙總 7 年平均與 t；另算「小−大」差。
python size_group_test.py
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BOOT, YEARS = 300, range(2018, 2025)
raw = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str},
                  usecols=["stock_code", "Date", "Volume", "close_unadj"], parse_dates=["Date"])
raw = raw.sort_values(["stock_code", "Date"]).reset_index(drop=True)
dv = raw.close_unadj * raw.Volume.where(raw.Volume > 0)
liq60 = dv.groupby(raw.stock_code).transform(lambda x: x.rolling(60, min_periods=20).median())
pct = liq60.groupby(raw.Date).rank(pct=True)
tercile = (np.ceil(pct * 3) - 1).values
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


GN = {0: "最小", 1: "中", 2: "最大"}
ARMS = ["+PTT", "+融資融券", "+當沖比率"]
res = {}
for tgt in ["B", "B2"]:
    res[tgt] = {}
    for Y in YEARS:
        o = pickle.load(open(f"daytrade_preds/{tgt}_{Y}.pkl", "rb"))
        idx, y, P = o["idx"], o["y"], o["P"]
        tq = tercile[idx]
        wk = raw.Date.dt.strftime("%G-%V").values[idx]
        u = np.unique(wk); wi = {w: np.where(wk == w)[0] for w in u}; rs = np.random.default_rng(Y)
        boots = [np.concatenate([wi[w] for w in rs.choice(u, size=len(u), replace=True)]) for _ in range(BOOT)]
        masks = {g: (tq == g) for g in (0, 1, 2)}
        masks["全部"] = ~np.isnan(tq)
        base_pt = {g: auc(y[m], P["技術面"][m]) for g, m in masks.items()}
        base_bs = {g: np.array([auc(y[b][m[b]], P["技術面"][b][m[b]]) for b in boots]) for g, m in masks.items()}
        res[tgt][Y] = {"base": base_pt}
        for arm in ARMS:
            d_pt, d_bs = {}, {}
            for g, m in masks.items():
                d_pt[g] = auc(y[m], P[arm][m]) - base_pt[g]
                d_bs[g] = np.array([auc(y[b][m[b]], P[arm][b][m[b]]) for b in boots]) - base_bs[g]
            r = {g: {"d": float(d_pt[g]), "lo": float(np.nanpercentile(d_bs[g], 2.5)), "hi": float(np.nanpercentile(d_bs[g], 97.5))} for g in masks}
            diff_bs = d_bs[0] - d_bs[2]
            r["小−大"] = {"d": float(d_pt[0] - d_pt[2]), "lo": float(np.nanpercentile(diff_bs, 2.5)), "hi": float(np.nanpercentile(diff_bs, 97.5))}
            res[tgt][Y][arm] = r
        say(f"[完成] {tgt} {Y}：各組 技術面 AUC " + " ".join(f"{GN.get(g, g)} {v:.4f}" for g, v in base_pt.items()))


def agg(tgt, arm, key):
    d = np.array([res[tgt][Y][arm][key]["d"] for Y in YEARS])
    sig = sum(res[tgt][Y][arm][key]["lo"] > 0 or res[tgt][Y][arm][key]["hi"] < 0 for Y in YEARS)
    return d.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(len(d))), int((d > 0).sum()), sig


say("\n================ 依規模分組的 ΔAUC（加入訊號 − 只有技術面；2018~2024 平均）================")
for tgt, nm in [("B", "隔天 開盤→收盤"), ("B2", "隔兩天 開盤→收盤")]:
    say(f"\n--- 目標 {nm} ---")
    bases = {g: np.mean([res[tgt][Y]["base"][g] for Y in YEARS]) for g in (0, 1, 2, "全部")}
    say("  技術面基準 AUC：" + "｜".join(f"{GN.get(g, g)} {v:.4f}" for g, v in bases.items()))
    for arm in ARMS:
        say(f" [{arm}]")
        for key, label in [(0, "最小"), (1, "中"), (2, "最大"), ("全部", "全部"), ("小−大", "小−大 差")]:
            m, t, pos, sig = agg(tgt, arm, key)
            say(f"    {label:<8s} 平均ΔAUC {m:+.4f}（t={t:+.2f}）｜為正 {pos}/7｜顯著 {sig}/7")
open("size_group_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
