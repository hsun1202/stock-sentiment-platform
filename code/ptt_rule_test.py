# -*- coding: utf-8 -*-
"""
只用 PTT 情緒預測漲跌方向 / 投資報酬（不搭配技術面）
====================================================
(1) 交易規則（不需訓練，測 2015-2024 十年）：PTT 淨情緒 (正−負)/(正+負) 大於 0 就買、小於 0 就賣（放空）
    A 開盤當沖：用到「隔天開盤前」的留言 → 隔天開盤進場、收盤出場（成本：當沖 32 bp 六折 / 44 bp 無折扣）
    B 收盤持有一天：只用到「今天收盤前」的留言 → 今天收盤進場、隔天收盤出場（成本：一般交易 47 bp 六折 / 59 bp 無折扣）
    變體：全部有表態留言的股票日／強訊號（表態留言 ≥5 則且 |淨情緒| ≥0.5）／只做多（情緒>0 才買）
    每天把所有部位等權平均；沒有部位的日子報酬記 0。
(2) 只用 PTT 特徵的模型（XGBoost，逐年滾動 2018-2024，目標＝隔天開盤→收盤）：方向 AUC、報酬 Rank IC
python ptt_rule_test.py
"""
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from features_v4 import build

warnings.filterwarnings("ignore")
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


df = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                 usecols=["stock_code", "Date", "Open", "Close", "abnormal_price", "ptt_pre_n", "ptt_pre_pos", "ptt_pre_neg",
                          "ptt_intra_n", "ptt_intra_pos", "ptt_intra_neg"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
for c in ["ptt_pre_pos", "ptt_pre_neg", "ptt_intra_pos", "ptt_intra_neg"]:
    df[c] = df[c].fillna(0)
pos_c = df.ptt_pre_pos + df.ptt_intra_pos            # 今天收盤前
neg_c = df.ptt_pre_neg + df.ptt_intra_neg
pos_o = pos_c + g.ptt_pre_pos.shift(-1).fillna(0)     # 再加上隔天開盤前
neg_o = neg_c + g.ptt_pre_neg.shift(-1).fillna(0)
abn = df.abnormal_price.fillna(0) != 0
abn_n = g.abnormal_price.shift(-1).fillna(0) != 0
nO, nC = g.Open.shift(-1), g.Close.shift(-1)
ok = ~abn & ~abn_n & nO.notna() & nC.notna()
df["r_oc"] = (nC / nO - 1).where(ok)
df["r_cc"] = (nC / df.Close - 1).where(ok)
RULES = {
    "A 開盤當沖（隔天開盤前的留言）": (pos_o, neg_o, "r_oc", {"六折32bp": 0.0032, "無折扣44bp": 0.0044}),
    "B 收盤進場持有一天（今天收盤前的留言）": (pos_c, neg_c, "r_cc", {"六折47bp": 0.0047, "無折扣59bp": 0.0059}),
}
days = pd.Series(sorted(df.Date.unique()))
say("只用 PTT 情緒的交易規則｜2015~2024｜等權｜沒有部位的日子記 0")
for name, (p, n, rcol, costs) in RULES.items():
    ex = p + n
    net = (p - n) / ex.replace(0, np.nan)
    say(f"\n=== {name} ===")
    for vname, mask in [("全部有表態", ex >= 1), ("強訊號(≥5則且|淨|≥0.5)", (ex >= 5) & (net.abs() >= 0.5))]:
        for side in ["多空（>0買、<0賣）", "只做多（>0才買）"]:
            m = mask & df[rcol].notna() & (net != 0) & net.notna()
            if side.startswith("只做多"):
                m = m & (net > 0)
            r = df.loc[m, rcol]
            sgn = np.sign(net[m])
            pr = r * sgn
            hit = (pr > 0).mean()
            nday = df.loc[m, "Date"].nunique()
            line = f"  [{vname}｜{side}] {m.sum():,} 筆（約 {m.sum() / nday:.1f} 檔/有訊號日，{nday} 天）｜猜對 {hit:.1%}"
            for cname, c in [("不扣成本", 0.0), *costs.items()]:
                daily = (pr - c).groupby(df.loc[m, "Date"]).mean().reindex(days).fillna(0)
                cum = (1 + daily).prod() - 1
                tt = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))
                line += f"｜{cname} 每筆 {(pr - c).mean() * 1e4:+.1f}bp、十年累積 {cum:+.0%}（t={tt:+.2f}）"
            say(line)
    # 逐年（全部有表態、多空、不扣成本）
    m = (ex >= 1) & df[rcol].notna() & (net != 0) & net.notna()
    pr = df.loc[m, rcol] * np.sign(net[m])
    yr = pr.groupby(df.loc[m, "Date"].dt.year).mean() * 1e4
    say("  逐年每筆平均（全部有表態、多空、不扣成本，bp）：" + " ".join(f"{y}:{v:+.1f}" for y, v in yr.items()))

# (2) 只用 PTT 特徵的模型
feat, meta, A = build()
PTT = [c for c in A["+PTT(開盤前可得)"] if c not in A["技術面"]]
keep, y, date, end = meta["valid"], meta["y"], meta["date"], meta["end_date"]
r_oc = (g.Close.shift(-1) / g.Open.shift(-1) - 1).values
has = (feat["ptt_open_att"] > 0).values
say(f"\n=== 只用 PTT 特徵的模型（{PTT}；XGBoost；逐年滾動 2018-2024；目標＝隔天開盤→收盤）===")
rows = []
for Y in range(2018, 2025):
    tr = keep & (end <= f"{Y-1}-06-30"); va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
    te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end <= f"{Y+1}-01-31")
    m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=50, subsample=0.8,
                          colsample_bytree=0.8, random_state=0, tree_method="hist", n_jobs=8, early_stopping_rounds=30, eval_metric="logloss")
    m.fit(feat.loc[tr, PTT], y[tr], eval_set=[(feat.loc[va, PTT], y[va])], verbose=False)
    p = m.predict_proba(feat.loc[te, PTT])[:, 1]
    idx = np.where(te.values)[0]
    rows.append(pd.DataFrame({"date": date.values[idx], "y": y.values[idx], "p": p, "r": r_oc[idx], "has": has[idx]}))
R = pd.concat(rows)
for label, sub in [("全部股票日", R), ("有 PTT 留言的股票日", R[R.has])]:
    auc = roc_auc_score(sub.y, sub.p)
    ic = sub.dropna().groupby("date").apply(lambda x: spearmanr(x.p, x.r)[0] if len(x) >= 10 else np.nan).dropna()
    say(f"  {label}（{len(sub):,} 筆）：方向 AUC {auc:.4f}（0.5＝亂猜）｜報酬 Rank IC {ic.mean():+.4f}（t={ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))):+.2f}）")
say("  （對照：只用技術面 AUC 約 0.556；技術面 IC 約 0.13）")
open("ptt_rule_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
