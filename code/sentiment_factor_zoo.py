# -*- coding: utf-8 -*-
"""
情緒因子比較：同一套預測設定下，比較「不同方式量出來的 PTT 散戶情緒因子」誰有預測力（十年 2015-2024，逐年滾動 2018~2024）
============================================================
動機：目前只用了一種因子（分類器三分類 → (正−負)/(正+負)），分類器準確率 59%；可能不是 PTT 沒資訊，而是因子沒量對。
預測設定（同 direction_test_v4 / retail_test_v4 的目標 B）：t+1 開盤→收盤漲跌；技術面基準；XGBoost 5 seeds；週 block bootstrap
時間窗：只用 t+1 開盤前可得的留言（依留言自己的時間；前一交易日收盤後 ~ t+1 開盤前）
因子（每組都另外帶「討論量」log(1+n)，讓各組比較基準一致）：
  F1 淨情緒（現有）   ：(正−負)/(正+負)，正負來自 Breeze 分類器
  F2 推噓比           ：(推−噓)/(推+噓)，網友自己按的推/噓，不經過分類器
  F3 情緒變化         ：F1 − 過去20個有討論日的 F1 平均（意外的情緒轉變）
  F4 意見分歧度       ：1 − |正−負|/(正+負)（多空都很多 → 分歧）
  F5 標的文多空       ：窗內「[標的] … 多」原發文數 − 「… 空」原發文數（散戶最明確的預測表態），另加5日累計
  F6 可信度加權情緒   ：每則留言依「該帳號在這則留言之前（前一個交易日以前）」的看多/看空準確率加權（時間點一致）：
                        準確率 = (過去猜對數+10)/(過去表態數+20)（貝氏收縮），權重 = 準確率 − 0.5；加權淨情緒 = Σ w·(±1) / Σ|w|
                        （第一版用「整段訓練期」的準確率當權重，訓練資料等於偷看同期結果 → 2018 年 ΔAUC −0.0049，已改掉）
輸出：sentiment_factor_zoo.log、sentiment_factor_zoo_results.json
python sentiment_factor_zoo.py
"""
import glob
import json
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from features_v4 import build
from match_ptt_2015_2024_267 import find_matches_v2

SEEDS, BOOT, TEST_YEARS = 5, 300, range(2018, 2025)
feat, meta, A = build()
TECH = A["技術面"]
base = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, usecols=["stock_code", "Date"])
base = base.sort_values(["stock_code", "Date"]).reset_index(drop=True)  # 與 features_v4 相同排序
base["Date"] = pd.to_datetime(base["Date"])
CAL = np.array(sorted(base.Date.unique()), dtype="datetime64[ns]")
OPEN, CLOSE = np.timedelta64(9 * 3600, "s"), np.timedelta64(13 * 3600 + 30 * 60, "s")


def assign(times):
    """時間 → (交易日 t, 是否在 t 開盤前)。t = 第一個收盤時間 >= 該時間的交易日。"""
    tv = pd.to_datetime(times).values
    i = np.searchsorted(CAL + CLOSE, tv, side="left")
    ok = i < len(CAL)
    t = np.where(ok, CAL[np.minimum(i, len(CAL) - 1)], np.datetime64("NaT"))
    pre = ok & (tv < CAL[np.minimum(i, len(CAL) - 1)] + OPEN)
    return t, pre, ok


# ---------------- 留言 ----------------
cm = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False,
                            usecols=["stock_code", "comment_time", "time_anomaly", "push_type", "user_id", "text_sentiment"])
                for f in glob.glob("sentiment_output_267_2015_2024/*_comment_labels.csv")], ignore_index=True)
cm = cm[(cm.comment_time != "") & (cm.time_anomaly != "1")].copy()
t, pre, ok = assign(cm.comment_time)
cm["t"], cm["pre"] = t, pre
cm = cm[ok].copy()
cm["s"] = np.select([cm.text_sentiment == "positive", cm.text_sentiment == "negative"], [1, -1], 0)
cm["push"] = (cm.push_type == "推").astype(int)
cm["boo"] = (cm.push_type == "噓").astype(int)
cm["pos"] = (cm.s == 1).astype(int)
cm["neg"] = (cm.s == -1).astype(int)
print(f"留言 {len(cm):,} 則", flush=True)

# 下一交易日開盤→收盤報酬（算帳號可信度用）
px = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str},
                 usecols=["stock_code", "Date", "Open", "Close", "abnormal_price"])
px["Date"] = pd.to_datetime(px.Date)
px = px.sort_values(["stock_code", "Date"])
g = px.groupby("stock_code")
px["next_oc"] = g.Close.shift(-1) / g.Open.shift(-1) - 1
px.loc[(px.abnormal_price == 1) | (g.abnormal_price.shift(-1) == 1), "next_oc"] = np.nan
# 留言在窗 (t 前一收盤, t 收盤] 或 t+1 開盤前 → 它「預測」的是它之後第一個完整交易日盤中；簡化：
# 盤前留言(pre) 對應當天 t 的盤中；盤中留言(intra) 對應 t+1 的盤中
px["this_oc"] = px.Close / px.Open - 1
cm = cm.merge(px[["stock_code", "Date", "this_oc", "next_oc"]].rename(columns={"Date": "t"}), on=["stock_code", "t"], how="left")
cm["target_ret"] = np.where(cm.pre, cm.this_oc, cm.next_oc)


def to_rows(df_parts, cols):
    """(股票, t, pre/intra) 聚合 → 每列特徵 = pre(t)+intra(t)+pre(t+1)（t+1 開盤前可得）"""
    a = df_parts.groupby(["stock_code", "t", "pre"])[cols].sum().reset_index()
    p = a[a.pre].drop(columns="pre").rename(columns={"t": "Date"})
    q = a[~a.pre].drop(columns="pre").rename(columns={"t": "Date"})
    m = base.merge(p, on=["stock_code", "Date"], how="left").fillna({c: 0 for c in cols})
    n = base.merge(q, on=["stock_code", "Date"], how="left").fillna({c: 0 for c in cols})
    out = m[cols] + n[cols]
    nxt = m.groupby("stock_code")[cols].shift(-1).fillna(0)
    return out + nxt


agg = to_rows(cm, ["pos", "neg", "push", "boo"])
agg["n"] = to_rows(cm.assign(one=1), ["one"])["one"]
lg = lambda s: s.groupby(base.stock_code)  # noqa: E731
feat["att"] = np.log1p(agg.n)
pn = agg.pos + agg.neg
feat["F1_net"] = np.where(pn > 0, (agg.pos - agg.neg) / pn.replace(0, 1), np.nan)
pb = agg.push + agg.boo
feat["F2_push"] = np.where(pb > 0, (agg.push - agg.boo) / pb.replace(0, 1), np.nan)
past = lg(feat["F1_net"]).transform(lambda s: s.shift(1).rolling(60, min_periods=3).mean())
feat["F3_shock"] = feat["F1_net"] - past
feat["F4_disagree"] = np.where(pn > 0, 1 - (agg.pos - agg.neg).abs() / pn.replace(0, 1), np.nan)

# ---------------- 標的文多空 ----------------
arts = pd.read_csv("ptt_2015_2024/articles.csv", usecols=["title", "category", "post_time"], dtype=str, keep_default_na=False)
tg = arts[(arts.category == "標的") & ~arts.title.str.match(r"^\s*(Re|Fw)", case=False)].copy()
tg["dir"] = tg.title.str.extract(r"(多|空)\s*[\)\]）!！~～.。]*\s*$")[0]
tg = tg[tg.dir.notna()]
rows = []
for title, pt, d in zip(tg.title, tg.post_time, tg.dir):
    for code, _, _ in find_matches_v2(title):
        rows.append((code, pt, 1 if d == "多" else 0, 1 if d == "空" else 0))
tgc = pd.DataFrame(rows, columns=["stock_code", "post_time", "long", "short"])
t2, pre2, ok2 = assign(tgc.post_time)
tgc["t"], tgc["pre"] = t2, pre2
tgc = tgc[ok2]
tagg = to_rows(tgc, ["long", "short"])
feat["F5_target"] = tagg.long - tagg.short
feat["F5_target5"] = lg(feat["F5_target"]).transform(lambda s: s.rolling(5, min_periods=1).sum())
print(f"標的文（原發文、標題以多/空結尾、比對到267檔）{len(tgc):,} 筆：多 {tgc.long.sum():,}、空 {tgc.short.sum():,}", flush=True)

# ---------------- F6：時間點一致的帳號可信度 ----------------
d = cm[(cm.s != 0)].sort_values(["user_id", "t"]).copy()
d["hit"] = np.where(d.target_ret.notna() & (d.target_ret != 0), (np.sign(d.target_ret) == d.s).astype(float), np.nan)
d["has"] = d.hit.notna().astype(float)
d["hit0"] = d.hit.fillna(0)
# 每個帳號、每個交易日先加總，再做「截至前一個有留言日」的累計，確保同一天的結果不會被用到
day = d.groupby(["user_id", "t"])[["hit0", "has"]].sum().groupby(level=0).cumsum()
day = day.groupby(level=0).shift(1).fillna(0).rename(columns={"hit0": "cum_hit", "has": "cum_n"})
d = d.merge(day, left_on=["user_id", "t"], right_index=True, how="left")
d["w"] = (d.cum_hit + 10) / (d.cum_n + 20) - 0.5
d["ws"], d["aw"] = d.w * d.s, d.w.abs()
wa = to_rows(d, ["ws", "aw"])
feat["F6_cred"] = np.where(wa.aw > 1e-9, wa.ws / wa.aw.replace(0, 1), np.nan)
print(f"F6：有過去紀錄的表態留言比例 {(d.cum_n > 0).mean():.1%}；權重 |w|>0.05 的比例 {(d.w.abs() > 0.05).mean():.1%}", flush=True)

FACTORS = {"F1 淨情緒(現有)": ["F1_net", "att"], "F2 推噓比": ["F2_push", "att"], "F3 情緒變化": ["F3_shock", "att"],
           "F4 意見分歧度": ["F4_disagree", "att"], "F5 標的文多空": ["F5_target", "F5_target5"],
           "F6 可信度加權情緒": ["F6_cred", "att"]}
keep, y, date, end = meta["valid"], meta["y"], meta["date"], meta["end_date"]
lines, res = [], {}


def log(s=""):
    print(s, flush=True)
    lines.append(s)


for Y in TEST_YEARS:
    tr = keep & (end <= f"{Y-1}-06-30")
    va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
    te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end <= f"{Y+1}-01-31")
    yte = y[te].values
    wk = date[te].dt.strftime("%G-%V").values
    uu = np.unique(wk)
    wi = {k: np.where(wk == k)[0] for k in uu}
    rs = np.random.default_rng(Y)
    boots = [np.concatenate([wi[k] for k in rs.choice(uu, size=len(uu), replace=True)]) for _ in range(BOOT)]
    P = {}
    for arm, cols in {"技術面": TECH, **{k: TECH + v for k, v in FACTORS.items()}}.items():
        ps = []
        for s in range(SEEDS):
            m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.8,
                                  colsample_bytree=0.8, random_state=s, tree_method="hist", n_jobs=16,
                                  early_stopping_rounds=30, eval_metric="logloss")
            m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
            ps.append(m.predict_proba(feat.loc[te, cols])[:, 1])
        P[arm] = np.mean(ps, axis=0)
    b0 = roc_auc_score(yte, P["技術面"])
    res[Y] = {"base": b0}
    line = f"{Y}: 技術面 {b0:.4f}"
    for arm in FACTORS:
        dd = roc_auc_score(yte, P[arm]) - b0
        bd = np.array([roc_auc_score(yte[b], P[arm][b]) - roc_auc_score(yte[b], P["技術面"][b]) for b in boots])
        lo, hi = float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))
        res[Y][arm] = {"d": dd, "lo": lo, "hi": hi}
        line += f"｜{arm.split()[0]} {dd:+.4f}{'✔' if lo > 0 else ''}"
    log(line)
log("--- 7年彙總（相對技術面的 ΔAUC）---")
for arm in FACTORS:
    ds = np.array([res[Y][arm]["d"] for Y in TEST_YEARS])
    sig = sum(res[Y][arm]["lo"] > 0 for Y in TEST_YEARS)
    log(f"{arm:14s} 平均 {ds.mean():+.4f}（t={ds.mean()/(ds.std(ddof=1)/np.sqrt(7)):+.2f}）｜顯著 {sig}/7｜為正 {(ds > 0).sum()}/7")
json.dump({str(k): v for k, v in res.items()}, open("sentiment_factor_zoo_results.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
open("sentiment_factor_zoo.log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
