# -*- coding: utf-8 -*-
"""
檢驗推測：「台股當沖多 → 資訊消化快 → 散戶情緒/籌碼訊號衰減快」
============================================================
資料：證交所每日當沖統計（chip_raw/daytrade_stock.csv，由 daytrade_build.py 產生）＋ final_dataset_v3 十年個股資料
設定：沿用 retail_test_v4.py（XGBoost、逐年滾動、測試年 2018~2024、週 block bootstrap），預測對象有兩個：
  B  = 隔天（t+1）開盤→收盤漲跌（原本的目標）
  B2 = 隔兩天（t+2）開盤→收盤漲跌（同樣只用 t 的資訊；檢驗訊號隔多一天還剩多少）
分組：每天依「60日當沖比率」把個股分高/低兩組；為了不要只是在比大小股，先依「60日成交值中位數」分三個流動性層，
      高/低組是在每個流動性層內部、每天各自切（高＝當天該層內當沖比率高於中位數）。
      另做「大型股（最高流動性層）」的高/低作為穩健性檢查。
當沖比率 ＝ 當沖成交股數 ÷ 成交股數；Yahoo 成交量為 0 或比率 > 1（壞值，集中在 2019-12、2021-2022）當缺值。
陣列：技術面 / +融資融券（散戶做）/ +PTT（散戶說）/ +當沖比率（當沖本身當特徵）
事先寫下的預測（若假說成立）：
  (1) 高當沖組的訊號從 B 到 B2 衰減得比低當沖組更快（B2/B 比值較小）；
  (2) 或高當沖組在 B 的 ΔAUC 與低當沖組顯著不同。
  結果不支持就誠實寫不支持，不再往下挖。
輸出：daytrade_test.log、daytrade_test_results.json；每個（目標,年）的預測機率存 daytrade_preds/（可重跑評估不必重訓）。
python daytrade_test.py              # 完整版（較久，建議用工作排程器）
SMOKE=1 python daytrade_test.py      # 小測試（只跑 2024、1 seed、50 棵樹）
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from features_v4 import build

SMOKE = os.environ.get("SMOKE") == "1"
SEEDS = 1 if SMOKE else 3
BOOT = 20 if SMOKE else 200
NEST = 50 if SMOKE else 500
TEST_YEARS = [2024] if SMOKE else list(range(2018, 2025))
PRED_DIR = "daytrade_preds_smoke" if SMOKE else "daytrade_preds"
TAG = "_smoke" if SMOKE else ""
os.makedirs(PRED_DIR, exist_ok=True)

# ---------------- 特徵與資料 ----------------
feat, meta, A = build()
TECH = A["技術面"]
PTT = [c for c in A["+PTT(開盤前可得)"] if c not in TECH]
MARGIN = ["margin_chg", "margin_chg5", "short_chg", "short_margin_ratio", "margin_eligible"]
date = meta["date"]

raw = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str},
                  usecols=["stock_code", "Date", "Open", "Close", "Volume", "close_unadj", "abnormal_price"])
raw["Date"] = pd.to_datetime(raw.Date)
raw = raw.sort_values(["stock_code", "Date"]).reset_index(drop=True)
assert (raw.stock_code.values == meta.stock_code.values).all() and (raw.Date.values == meta.date.values).all()

S = pd.read_csv("chip_raw/daytrade_stock.csv", dtype={"stock_code": str}, parse_dates=["Date"])
S = S.drop_duplicates(["stock_code", "Date"])[["stock_code", "Date", "dt_shares"]]
n0 = len(raw)
raw = raw.merge(S, on=["stock_code", "Date"], how="left")
assert len(raw) == n0
raw["dt_shares"] = raw.dt_shares.fillna(0)          # 沒列在當沖統計＝當天沒有當沖（早年多數股票也不能當沖）
vol = raw.Volume.where(raw.Volume > 0)
ratio = (raw.dt_shares / vol)
raw["dt_r1"] = ratio.where(ratio <= 1)               # >1 是 Yahoo 成交量壞值


def roll(s, w, mp, fn="mean"):
    return s.groupby(raw.stock_code).transform(lambda x: getattr(x.rolling(w, min_periods=mp), fn)())


feat["dt_r1"] = raw.dt_r1
feat["dt_r5"] = roll(raw.dt_r1, 5, 3)
feat["dt_r60"] = roll(raw.dt_r1, 60, 20)
DT = ["dt_r1", "dt_r5", "dt_r60"]
liq60 = roll(raw.close_unadj * vol, 60, 20, "median")

# ---------------- 分組：流動性層內的高/低當沖 ----------------
ok = feat.dt_r60.notna() & liq60.notna()
pct = liq60.where(ok).groupby(date).rank(pct=True)
liq_bin = np.ceil(pct * 3) - 1                                       # 0,1,2（2＝最大/最活躍）
med = feat.dt_r60.where(ok).groupby([date, liq_bin]).transform("median")
hi = (feat.dt_r60 > med) & ok
lo = (feat.dt_r60 <= med) & ok
GROUPS = {"高當沖": hi.values, "低當沖": lo.values,
          "大型·高當沖": (hi & (liq_bin == 2)).values, "大型·低當沖": (lo & (liq_bin == 2)).values}
DIFFS = {"高−低": ("高當沖", "低當沖"), "大型 高−低": ("大型·高當沖", "大型·低當沖")}

# ---------------- 目標：B（t+1）與 B2（t+2）的開盤→收盤 ----------------
g = raw.groupby("stock_code")
abn = raw.abnormal_price
nO2, nC2 = g.Open.shift(-2), g.Close.shift(-2)
valid2 = nC2.notna() & nO2.notna() & (nC2 != nO2) & (g.abnormal_price.shift(-2).fillna(0) == 0) & (abn == 0) \
    & feat[TECH].notna().all(axis=1)
TARGETS = {
    "B": (meta["valid"], meta["y"], meta["end_date"]),
    "B2": (valid2, (nC2 > nO2).astype(int), pd.to_datetime(g.Date.shift(-2))),
}
ARMS = {"技術面": TECH, "+融資融券": TECH + MARGIN, "+PTT": TECH + PTT, "+當沖比率": TECH + DT}
BASE = "技術面"
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


say(f"[設定] SMOKE={SMOKE} SEEDS={SEEDS} BOOT={BOOT} 測試年 {TEST_YEARS[0]}~{TEST_YEARS[-1]}")
for k, v in GROUPS.items():
    say(f"  分組 {k}：{int(v.sum()):,} 列（占全部 {v.mean():.1%}）")
for t, (kp, _, _) in TARGETS.items():
    say(f"  目標 {t} 有效列：{int(kp.sum()):,}")


# ---------------- 訓練（逐年、可續跑）----------------
def run_year(tgt, Y):
    path = f"{PRED_DIR}/{tgt}_{Y}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    keep, y, end = TARGETS[tgt]
    tr = keep & (end <= f"{Y-1}-06-30")
    va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
    te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end <= f"{Y+1}-01-31")
    P = {}
    for arm, cols in ARMS.items():
        ps = []
        for s in range(SEEDS):
            m = xgb.XGBClassifier(n_estimators=NEST, learning_rate=0.04, max_depth=4, min_child_weight=50, subsample=0.8,
                                  colsample_bytree=0.8, random_state=s, tree_method="hist", n_jobs=8,
                                  early_stopping_rounds=30, eval_metric="logloss")
            m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
            ps.append(m.predict_proba(feat.loc[te, cols])[:, 1])
        P[arm] = np.mean(ps, axis=0)
    out = {"idx": np.where(te.values)[0], "y": y[te].values, "P": P}
    pickle.dump(out, open(path, "wb"))
    say(f"  [訓練完成] {tgt} {Y}：訓練 {int(tr.sum()):,}／測試 {int(te.sum()):,}")
    return out


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


# ---------------- 評估 ----------------
res = {}
for tgt in TARGETS:
    res[tgt] = {}
    for Y in TEST_YEARS:
        o = run_year(tgt, Y)
        idx, yy, P = o["idx"], o["y"], o["P"]
        wk = date.dt.strftime("%G-%V").values[idx]
        u = np.unique(wk)
        wi = {w: np.where(wk == w)[0] for w in u}
        rs = np.random.default_rng(Y)
        boots = [np.concatenate([wi[w] for w in rs.choice(u, size=len(u), replace=True)]) for _ in range(BOOT)]
        masks = {"全部": np.ones(len(idx), bool), **{k: v[idx] for k, v in GROUPS.items()}}
        # 每個組別 × 每個臂的 ΔAUC（點估計＋bootstrap）
        d_pt, d_bs = {}, {}
        for gname, gm in masks.items():
            base_bs = np.array([auc(yy[b][gm[b]], P[BASE][b][gm[b]]) for b in boots])
            for arm in ARMS:
                if arm == BASE:
                    continue
                d_pt[(arm, gname)] = auc(yy[gm], P[arm][gm]) - auc(yy[gm], P[BASE][gm])
                d_bs[(arm, gname)] = np.array([auc(yy[b][gm[b]], P[arm][b][gm[b]]) for b in boots]) - base_bs
        res[tgt][Y] = {}
        for arm in ARMS:
            if arm == BASE:
                continue
            r = {}
            for gname in masks:
                bs = d_bs[(arm, gname)]
                r[gname] = {"d": float(d_pt[(arm, gname)]), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5))}
            for dn, (a, b_) in DIFFS.items():
                diff_pt = d_pt[(arm, a)] - d_pt[(arm, b_)]
                bs = d_bs[(arm, a)] - d_bs[(arm, b_)]
                r[dn] = {"d": float(diff_pt), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5))}
            res[tgt][Y][arm] = r
        say(f"[評估完成] {tgt} {Y}：基準AUC 全部 {auc(yy, P[BASE]):.4f}")


def agg(tgt, arm, key):
    d = np.array([res[tgt][Y][arm][key]["d"] for Y in TEST_YEARS])
    sig = sum(res[tgt][Y][arm][key]["lo"] > 0 or res[tgt][Y][arm][key]["hi"] < 0 for Y in TEST_YEARS)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
    return d.mean(), t, int((d > 0).sum()), sig


say("\n================ 彙總（各年 ΔAUC 平均；t 為 7 年平均的 t；為正/顯著為年數）================")
KEYS = ["全部", "高當沖", "低當沖", "高−低", "大型·高當沖", "大型·低當沖", "大型 高−低"]
for tgt, nm in [("B", "B：隔天 開盤→收盤"), ("B2", "B2：隔兩天 開盤→收盤")]:
    say(f"\n--- 目標 {nm} ---")
    for arm in ARMS:
        if arm == BASE:
            continue
        say(f" [{arm}]")
        for k in KEYS:
            m, t, pos, sig = agg(tgt, arm, k)
            say(f"    {k:<10s} 平均ΔAUC {m:+.4f}（t={t:+.2f}）｜為正 {pos}/{len(TEST_YEARS)}｜顯著 {sig}/{len(TEST_YEARS)}")

say("\n================ 衰減：B2 的平均 ΔAUC ÷ B 的平均 ΔAUC（分母太小時只當參考）================")
for arm in ARMS:
    if arm == BASE:
        continue
    row = []
    for k in ["全部", "高當沖", "低當沖", "大型·高當沖", "大型·低當沖"]:
        b, b2 = agg("B", arm, k)[0], agg("B2", arm, k)[0]
        row.append(f"{k} {b:+.4f}→{b2:+.4f}（{b2 / b if abs(b) > 1e-4 else float('nan'):.2f}）")
    say(f" [{arm}] " + "｜".join(row))

# ---------------- 年度對照：大盤當沖占比 vs 融資融券訊號強度 ----------------
M = pd.read_csv("chip_raw/daytrade_market.csv", parse_dates=["Date"])
mk = M.assign(year=M.Date.dt.year).groupby("year").dt_buy_pct.mean()
say("\n================ 年度對照：大盤當沖占比（買進金額%）與 融資融券 ΔAUC（全部個股）================")
for Y in TEST_YEARS:
    say(f"  {Y}  大盤當沖占比 {mk[Y]:5.1f}%  B {res['B'][Y]['+融資融券']['全部']['d']:+.4f}  B2 {res['B2'][Y]['+融資融券']['全部']['d']:+.4f}")

open(f"daytrade_test{TAG}.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
json.dump({t: {str(Y): v for Y, v in d.items()} for t, d in res.items()},
          open(f"daytrade_test{TAG}_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
say("\nDONE")
