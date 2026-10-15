# -*- coding: utf-8 -*-
"""
新聞情緒分析（news_sentiment_classify.py 的結果）：三種檢驗
============================================================
(1) 個股新聞事件研究：比對到 267 檔的 [新聞]，依 LLM 判斷「好/壞/無」，看之後的「超額報酬」
    進場 = 發文後第一個能成交的開盤（09:00 前發文 → 當天開盤；之後 → 下一交易日開盤），不偷看
    報酬：進場日 開盤→收盤（1日）、開盤→第5/第20個交易日收盤；超額 = 個股 − 267檔等權平均（同一段期間）
    t 值：依進場日群聚（同日多則新聞不當獨立樣本）
(2) 放進預測模型（同 direction_test_v4 目標B、逐年滾動、XGBoost 5 seeds）：
    特徵 = t+1 開盤前可得的新聞：新聞數、平均 news_score、好−壞 則數（依發文時間歸屬）
(3) 大盤新聞溫度：每個交易日窗 (t-1 收盤, t 收盤] 內「全部」新聞的平均 news_score（5日平均），
    預測加權指數未來 1/5/20 日報酬，控制過去5/20日報酬，Newey-West
輸出：news_sentiment_analysis.log、news_sentiment_results.json
python news_sentiment_analysis.py
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from features_v4 import build

ns = pd.read_csv("news_sentiment/news_scores.csv", dtype={"article_id": str, "stock_codes": str}, keep_default_na=False)
ns["post_time"] = pd.to_datetime(ns.post_time)
px = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str},
                 usecols=["stock_code", "Date", "Open", "Close", "abnormal_price"])
px["Date"] = pd.to_datetime(px.Date)
px = px.sort_values(["stock_code", "Date"]).reset_index(drop=True)
CAL = np.array(sorted(px.Date.unique()), dtype="datetime64[ns]")
OPEN, CLOSE = np.timedelta64(9 * 3600, "s"), np.timedelta64(13 * 3600 + 30 * 60, "s")
lines, res = [], {}


def log(s=""):
    print(s, flush=True)
    lines.append(s)


log(f"新聞 {len(ns):,} 篇；LLM 判斷 " + "、".join(f"{k} {v:.1%}" for k, v in ns.news_label.value_counts(normalize=True).items()))

# ---------------- (1) 事件研究 ----------------
g = px.groupby("stock_code")
for h in (1, 5, 20):
    px[f"r{h}"] = g.Close.shift(-(h - 1)) / px.Open - 1          # 從當天開盤到第 h 個交易日收盤
    bad = sum(g.abnormal_price.shift(-k).fillna(1) for k in range(0, h))
    px.loc[bad > 0, f"r{h}"] = np.nan
    px[f"ew{h}"] = px.groupby("Date")[f"r{h}"].transform("mean")
    px[f"ar{h}"] = px[f"r{h}"] - px[f"ew{h}"]
ev = ns[ns.stock_codes != ""].assign(stock_code=lambda d: d.stock_codes.str.split(",")).explode("stock_code")
tv = ev.post_time.values
i = np.searchsorted(CAL + OPEN, tv, side="left")      # 第一個「開盤時間 ≥ 發文時間」的交易日
ev = ev[i < len(CAL)].copy()
ev["Date"] = CAL[i[i < len(CAL)]]
ev = ev.merge(px[["stock_code", "Date"] + [f"ar{h}" for h in (1, 5, 20)] + [f"r{h}" for h in (1, 5, 20)]],
              on=["stock_code", "Date"], how="left")
log(f"\n==== (1) 個股新聞事件研究：{len(ev):,} 筆（股票×新聞），超額報酬 = 個股 − 267檔等權平均，t值依進場日群聚 ====")
strong = ev.news_score.abs() >= ev.news_score.abs().quantile(0.5)
for lab in ("好", "壞", "無"):
    for tag, sel in (("全部", ev.news_label == lab), ("強度前50%", (ev.news_label == lab) & strong)):
        out = []
        for h in (1, 5, 20):
            d = ev.loc[sel, [f"ar{h}", "Date"]].dropna()
            if len(d) < 30:
                continue
            m = sm.OLS(d[f"ar{h}"].values, np.ones(len(d))).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.Date)[0]})
            out.append((h, float(m.params[0]), float(m.tvalues[0]), len(d)))
            res.setdefault("event", {}).setdefault(f"{lab}_{tag}", {})[f"{h}d"] = {"ar": out[-1][1], "t": out[-1][2], "n": len(d)}
        log(f"  [{lab}｜{tag}] " + "｜".join(f"{h}日 {a:+.2%}(t={t:+.2f}, n={n:,})" for h, a, t, n in out))
d = ev[["ar1", "ar5", "ar20", "news_score", "Date"]].dropna()
for h in (1, 5, 20):
    m = sm.OLS(d[f"ar{h}"], sm.add_constant(d[["news_score"]])).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.Date)[0]})
    log(f"  迴歸 超額報酬{h}日 ~ news_score：係數 {m.params['news_score']:+.4%}/分（t={m.tvalues['news_score']:+.2f}）")

# ---------------- (2) 預測模型 ----------------
feat, meta, A = build()
TECH = A["技術面"]
base = px[["stock_code", "Date"]]
sv = ev.copy()
ti = np.searchsorted(CAL + CLOSE, sv.post_time.values, side="left")
sv = sv[ti < len(CAL)].copy()
sv["t"] = CAL[ti[ti < len(CAL)]]
sv["pre"] = sv.post_time.values < sv.t.values + OPEN
sv["good"] = (sv.news_label == "好").astype(int)
sv["bad"] = (sv.news_label == "壞").astype(int)
sv["one"] = 1
cols = ["one", "good", "bad", "news_score"]
a2 = sv.groupby(["stock_code", "t", "pre"])[cols].sum().reset_index()
P = base.merge(a2[a2.pre].drop(columns="pre").rename(columns={"t": "Date"}), on=["stock_code", "Date"], how="left").fillna(0)
Q = base.merge(a2[~a2.pre].drop(columns="pre").rename(columns={"t": "Date"}), on=["stock_code", "Date"], how="left").fillna(0)
W = P[cols] + Q[cols] + P.groupby("stock_code")[cols].shift(-1).fillna(0)   # t 收盤前 + t 收盤後~t+1 開盤前
feat["news_n"] = np.log1p(W.one.values)
feat["news_net"] = (W.good - W.bad).values
feat["news_avg"] = np.where(W.one > 0, W.news_score / W.one.replace(0, 1), np.nan)
NEWS = ["news_n", "news_net", "news_avg"]
log(f"\n==== (2) 預測模型：t+1 開盤→收盤漲跌，技術面 vs +新聞情緒（有新聞的股票日 {(W.one > 0).mean():.2%}） ====")
keep, y, date, end = meta["valid"], meta["y"], meta["date"], meta["end_date"]
ds, sig = [], 0
for Y in range(2018, 2025):
    tr = keep & (end <= f"{Y-1}-06-30"); va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
    te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31") & (end <= f"{Y+1}-01-31")
    yte = y[te].values
    has = (W.one.values[te.values] > 0)
    pr = {}
    for arm, c in (("技術面", TECH), ("+新聞", TECH + NEWS)):
        ps = []
        for s in range(5):
            m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.8,
                                  colsample_bytree=0.8, random_state=s, tree_method="hist", n_jobs=16,
                                  early_stopping_rounds=30, eval_metric="logloss")
            m.fit(feat.loc[tr, c], y[tr], eval_set=[(feat.loc[va, c], y[va])], verbose=False)
            ps.append(m.predict_proba(feat.loc[te, c])[:, 1])
        pr[arm] = np.mean(ps, axis=0)
    dd = roc_auc_score(yte, pr["+新聞"]) - roc_auc_score(yte, pr["技術面"])
    dh = roc_auc_score(yte[has], pr["+新聞"][has]) - roc_auc_score(yte[has], pr["技術面"][has]) if has.sum() > 200 else np.nan
    wk = date[te].dt.strftime("%G-%V").values; uu = np.unique(wk); wi = {k: np.where(wk == k)[0] for k in uu}
    rs = np.random.default_rng(Y)
    bd = [roc_auc_score(yte[b], pr["+新聞"][b]) - roc_auc_score(yte[b], pr["技術面"][b])
          for b in (np.concatenate([wi[k] for k in rs.choice(uu, len(uu))]) for _ in range(300))]
    lo = np.percentile(bd, 2.5); sig += lo > 0; ds.append(dd)
    res.setdefault("model", {})[Y] = {"d_auc": dd, "d_auc_news_days": dh, "ci_lo": float(lo)}
    log(f"  {Y}: ΔAUC {dd:+.4f}{'✔' if lo > 0 else ''}｜只看有新聞的股票日 {dh:+.4f}（{has.sum():,} 筆）")
ds = np.array(ds)
log(f"  7年平均 ΔAUC {ds.mean():+.4f}（t={ds.mean()/(ds.std(ddof=1)/np.sqrt(7)):+.2f}）｜顯著 {sig}/7｜為正 {(ds > 0).sum()}/7")

# ---------------- (3) 大盤新聞溫度 ----------------
tx = pd.read_csv("market_sentiment/taiex_official.csv", parse_dates=["Date"]).set_index("Date")
tj = np.searchsorted(CAL + CLOSE, ns.post_time.values, side="left")
nn = ns[tj < len(CAL)].copy(); nn["Date"] = CAL[tj[tj < len(CAL)]]
day = nn.groupby("Date").agg(tone=("news_score", "mean"), n=("news_score", "size"))
dfm = tx.join(day, how="left")
dfm["tone5"] = dfm.tone.rolling(5, min_periods=3).mean()
dfm["tone_z"] = (dfm.tone5 - dfm.tone5.expanding(250).mean()) / dfm.tone5.expanding(250).std()
lr = np.log(dfm.taiex)
dfm["past5"], dfm["past20"] = lr - lr.shift(5), lr - lr.shift(20)
dfm = dfm.loc["2015-01-01":"2024-12-31"]
log(f"\n==== (3) 大盤新聞溫度（每日新聞中位數 {dfm.n.median():.0f} 則）：與過去5日大盤報酬相關 {dfm.tone_z.corr(dfm.past5):+.2f} ====")
out = []
for h in (1, 5, 20):
    dfm[f"f{h}"] = lr.shift(-h) - lr
    d = dfm[[f"f{h}", "tone_z", "past5", "past20"]].dropna()
    m = sm.OLS(d[f"f{h}"], sm.add_constant(d[["tone_z", "past5", "past20"]])).fit(cov_type="HAC", cov_kwds={"maxlags": max(h, 5)})
    out.append((h, m.params["tone_z"], m.tvalues["tone_z"]))
    res.setdefault("market", {})[f"h{h}"] = {"coef": float(m.params["tone_z"]), "t": float(m.tvalues["tone_z"])}
log("  未來大盤報酬 ~ 新聞溫度(1σ)+過去報酬：" + "｜".join(f"{h}日 {b:+.2%}(t={t:+.2f})" for h, b, t in out))
json.dump(res, open("news_sentiment_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
open("news_sentiment_analysis.log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
