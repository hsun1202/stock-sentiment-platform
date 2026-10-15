# -*- coding: utf-8 -*-
"""
控制美股隔夜報酬後：PTT「盤後/隔夜閒聊」情緒 → 隔天大盤漲跌（承 afterhours_replication.py）
============================================================================
論文（Taiwan FinBERT + CNN-BiLSTM-SA）：盤後閒聊情緒＋開高低收 → 預測加權指數隔天漲跌，準確率 87.5%→90.62%（疑似測試集僅約 32 天）。
本腳本用我們已有的閒聊情緒標籤（market_sentiment/chat_labels.csv）＋官方加權指數，做嚴格版本：
  - 情緒只用「開盤前已發」的留言（留言時間 < 當日 09:00，含前一晚盤後與隔夜）
  - 技術面基準：過去報酬/波動/量能/RSI/前一日 OHLC 型態（全部只用前一日收盤前資料）
  - 模型：邏輯迴歸、XGBoost（輕量，CPU 幾分鐘）
  - 逐年滾動：測試年 2018~2024，訓練 = 之前所有年份
  - 三種目標：(1) 收盤→收盤（論文設定）(2) 前收→開盤（隔夜跳空）(3) 開盤→收盤（盤中）
  - 檢定：以月為區塊的配對 bootstrap（ΔAUC、Δ準確率）；另外模擬「只看 32 天測試集」時準確率差異有多雜訊
python afterhours_replication.py > afterhours_replication.log
"""
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
TEST_YEARS = list(range(2018, 2025))
SEEDS = 5
B = 2000

# ---------------- 價格 ----------------
cal = pd.read_csv("market_sentiment/taiex_official.csv", parse_dates=["Date"]).sort_values("Date")
im = pd.read_csv("market_sentiment/taiex_intraday_1min.csv", parse_dates=["Date"])
g = im.groupby("Date").taiex
# 09:00 那筆＝前一日收盤（指數尚未有成交），開盤價改用 09:01 那筆
op = im[im.time.str[:5] == "09:01"].set_index("Date").taiex
ohlc = pd.DataFrame({"open": op, "high": g.max(), "low": g.min()})
px = cal.merge(ohlc, left_on="Date", right_index=True, how="left").set_index("Date")
px = px[px.index >= "2014-06-03"]
close, openp, high, low, turn = px.taiex, px.open, px.high, px.low, px.turnover
ret = close.pct_change()

# ---------------- 技術面特徵（第 k 日收盤後可得，之後 shift(1) 對齊到預測日） ----------------
f = pd.DataFrame(index=px.index)
f["ret1"] = ret
f["ret2"] = ret.shift(1)
f["ret3"] = ret.shift(2)
f["ret5"] = close / close.shift(5) - 1
f["ret10"] = close / close.shift(10) - 1
f["ret20"] = close / close.shift(20) - 1
f["vol20"] = ret.rolling(20).std()
f["turn_ratio"] = np.log(turn / turn.rolling(20).mean())
f["ma5_ratio"] = close / close.rolling(5).mean() - 1
f["ma20_ratio"] = close / close.rolling(20).mean() - 1
d = close.diff()
rs = d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean()
f["rsi14"] = 100 - 100 / (1 + rs)
f["range"] = (high - low) / close
f["gap"] = openp / close.shift(1) - 1
f["oc"] = close / openp - 1
TECH = list(f.columns)
F = f.shift(1)  # 預測日 d 用 d-1 以前

# ---------------- 目標 ----------------
Y = pd.DataFrame(index=px.index)
Y["cc"] = (close > close.shift(1)).astype(float)                       # 收盤→收盤（論文）
Y["gap"] = np.where(openp.notna(), (openp > close.shift(1)).astype(float), np.nan)  # 前收→開盤
Y["oc"] = np.where(openp.notna(), (close > openp).astype(float), np.nan)            # 開盤→收盤
Y = Y.where(ret.notna(), np.nan)

# ---------------- 情緒特徵 ----------------
c = pd.read_csv("market_sentiment/chat_labels.csv", dtype=str, keep_default_na=False)
c["t"] = pd.to_datetime(c.comment_time)
c["Date"] = pd.to_datetime(c.Date)
c["ah"] = c.t < (c.Date + pd.Timedelta(hours=9))  # 開盤前已發（前一晚盤後＋隔夜）
c["pos"] = (c.text_sentiment == "positive").astype(int)
c["neg"] = (c.text_sentiment == "negative").astype(int)


def net(gr):
    p, n = gr.pos.sum(), gr.neg.sum()
    return pd.Series({"net": (p - n) / (p + n) if p + n > 0 else np.nan, "n_expr": p + n})


assert (c[c.ah].t < c[c.ah].Date + pd.Timedelta(hours=9)).all()
hm = c.t.dt.hour * 60 + c.t.dt.minute
c["eve"] = c.ah & (hm >= 13 * 60 + 30) & (hm < 21 * 60 + 30)   # 前晚（美股開盤前）
c["night"] = c.ah & ~c.eve                                      # 深夜～開盤前（美股交易時段及收盤後）
S_ah = c[c.ah].groupby("Date").apply(net)     # 開盤前已發
S_eve = c[c.eve].groupby("Date").apply(net)
S_night = c[c.night].groupby("Date").apply(net)
S_all = c.groupby("Date").apply(net)          # 整個窗口（前收→當日收盤）
S = pd.DataFrame(index=px.index)
S["S_ah"] = S_ah["net"].reindex(px.index)
S["S_ah_n"] = S_ah["n_expr"].reindex(px.index)
S["dS_ah"] = S.S_ah - S.S_ah.shift(1).rolling(20, min_periods=10).mean()
# 前一日整日情緒（d-1 窗口，開盤前已知）
S["S_day_prev"] = S_all["net"].reindex(px.index).shift(1)
S["S_eve"] = S_eve["net"].reindex(px.index)
S["S_night"] = S_night["net"].reindex(px.index)
SENT_A = ["S_ah", "dS_ah"]
SENT_B = ["S_ah", "dS_ah", "S_day_prev"]
SENT_EVE = ["S_eve"]
SENT_NIGHT = ["S_night"]

D = pd.concat([F, S, Y.add_prefix("y_")], axis=1)
D = D[(D.index >= "2015-03-01")].dropna(subset=TECH + ["S_ah", "dS_ah", "S_day_prev"])
D["S_eve"] = D.S_eve.fillna(0)
D["S_night"] = D.S_night.fillna(0)

# ---------------- 美股隔夜報酬：台股交易日 d 用「日期嚴格早於 d 的最近一個美股交易日」的報酬 ----------------
us = pd.read_csv("market_sentiment/us_daily.csv", parse_dates=["Date"]).set_index("Date").sort_index()
usf = pd.DataFrame({"us_spx": us.spx.pct_change(), "us_ndx": us.ndx.pct_change(),
                    "us_sox": us.sox.pct_change(), "us_vix": us.vix.pct_change()}).dropna().reset_index()
usf = usf.rename(columns={"Date": "usdate"})
tmp = pd.merge_asof(pd.DataFrame({"Date": D.index}).sort_values("Date"), usf, left_on="Date", right_on="usdate",
                    allow_exact_matches=False)
tmp = tmp.set_index("Date")
assert ((tmp.index - tmp.usdate).dt.days.between(1, 6)).all()
assert (tmp.usdate < tmp.index).all()
US = ["us_spx", "us_ndx", "us_sox", "us_vix"]
D = D.join(tmp[US])
D = D.dropna(subset=US)
print(f"美股對齊後樣本 {len(D)} 日；美股報酬與台股跳空相關：" +
      "、".join(f"{u} {D[u].corr(((openp / close.shift(1) - 1).reindex(D.index))):+.2f}" for u in US))
D["year"] = D.index.year
D["ym"] = D.index.year * 100 + D.index.month
print(f"分析樣本 {D.index.min().date()}~{D.index.max().date()}，{len(D)} 個交易日；測試年 {TEST_YEARS[0]}~{TEST_YEARS[-1]}")
print(f"開盤前已發且有表態的留言/日：中位數 {S.S_ah_n.median():.0f} 則")

# ---------------- 描述性診斷 ----------------
r_prev = ret.shift(1).reindex(D.index)
r_now = ret.reindex(D.index)
gap_now = (openp / close.shift(1) - 1).reindex(D.index)
print("\n[診斷] 開盤前情緒 S_ah 與……的相關（全期）：")
print(f"  前一日報酬 r_(d-1)  {D.S_ah.corr(r_prev):+.3f}   ← 情緒跟著前一日行情走（價格帶動情緒）")
print(f"  當日跳空 gap_d      {D.S_ah.corr(gap_now):+.3f}   ← 隔夜資訊（美股/夜盤）已反映在開盤前留言")
print(f"  當日報酬 r_d        {D.S_ah.corr(r_now):+.3f}")
oc_now = (close / openp - 1).reindex(D.index)
print(f"  開盤→收盤報酬       {D.S_ah.corr(oc_now):+.3f}   ← 若只有跳空相關、盤中不相關，代表資訊在開盤前就被價格吸收")
for nm in ["S_eve", "S_night"]:
    print(f"  {nm:<8s} vs 前一日報酬 {D[nm].corr(r_prev):+.3f}｜跳空 {D[nm].corr(gap_now):+.3f}｜開盤→收盤 {D[nm].corr(oc_now):+.3f}")
print(f"  基準：上漲日比例（全期）cc={D.y_cc.mean():.3f}")


# ---------------- 模型 ----------------
def fit_predict(kind, Xtr, ytr, Xte):
    if kind == "LR":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xte))[:, 1]
    p = np.zeros(len(Xte))
    for s in range(SEEDS):
        m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=10, reg_lambda=5, tree_method="hist", n_jobs=4, random_state=s,
                              verbosity=0)
        m.fit(Xtr, ytr)
        p += m.predict_proba(Xte)[:, 1]
    return p / SEEDS


SETS = {"技術面": TECH, "技術面+美股": TECH + US,
        "+美股+盤後情緒": TECH + US + SENT_A, "+美股+盤後+前日情緒": TECH + US + SENT_B,
        "+美股+只有前晚": TECH + US + SENT_EVE, "+美股+只有深夜": TECH + US + SENT_NIGHT}
BASE = "技術面+美股"


def walk(target, kind):
    """回傳 dict: 特徵組 -> DataFrame(index=date, p, y)"""
    y = D["y_" + target]
    out = {k: [] for k in SETS}
    base_rate = []
    for Y_ in TEST_YEARS:
        tr = (D.year < Y_) & y.notna()
        te = (D.year == Y_) & y.notna()
        base_rate.append(pd.DataFrame({"p": y[tr].mean(), "y": y[te]}, index=D.index[te]))
        for k, cols in SETS.items():
            p = fit_predict(kind, D.loc[tr, cols].values, y[tr].values, D.loc[te, cols].values)
            out[k].append(pd.DataFrame({"p": p, "y": y[te].values}, index=D.index[te]))
    res = {k: pd.concat(v) for k, v in out.items()}
    res["多數類"] = pd.concat(base_rate)
    return res


def acc(df):
    return ((df.p > 0.5) == (df.y == 1)).mean()


def auc(df):
    return roc_auc_score(df.y, df.p)


def boot_diff(a, b, fn):
    """月區塊配對 bootstrap：fn(b)-fn(a)"""
    ym = pd.Series(a.index.year * 100 + a.index.month, index=a.index)
    groups = [np.where(ym.values == m)[0] for m in ym.unique()]
    ds = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        try:
            ds.append(fn(b.iloc[idx]) - fn(a.iloc[idx]))
        except ValueError:
            pass
    ds = np.array(ds)
    return ds.mean(), np.percentile(ds, 2.5), np.percentile(ds, 97.5), (ds > 0).mean()


results = {}
NAMES = {"cc": "收盤→收盤（論文設定）", "gap": "前收→開盤（隔夜跳空）", "oc": "開盤→收盤（盤中）"}
for target in ["cc", "gap", "oc"]:
    for kind in ["LR", "XGB"]:
        R = walk(target, kind)
        base = R[BASE]
        n = len(base)
        print(f"\n=== 目標：{NAMES[target]}　模型：{kind}　測試日數 {n}（上漲比例 {base.y.mean():.3f}）===")
        print(f"  多數類猜法 準確率 {acc(R['多數類']):.4f}")
        row = {"n": n, "majority_acc": acc(R["多數類"]), "sets": {}}
        for k in SETS:
            df = R[k]
            line = f"  {k:<10s} 準確率 {acc(df):.4f}  AUC {auc(df):.4f}"
            entry = {"acc": acc(df), "auc": auc(df)}
            if k not in ("技術面", BASE):
                m, lo, hi, pos = boot_diff(base, df, auc)
                m2, lo2, hi2, pos2 = boot_diff(base, df, acc)
                line += f"   ΔAUC {m:+.4f} [{lo:+.4f},{hi:+.4f}] P(>0)={pos:.2f} | Δ準確率 {m2*100:+.2f}pp [{lo2*100:+.2f},{hi2*100:+.2f}]"
                entry.update({"dAUC": m, "dAUC_lo": lo, "dAUC_hi": hi, "dAcc": m2, "dAcc_lo": lo2, "dAcc_hi": hi2})
            print(line)
            row["sets"][k] = entry
        # 每年
        yr = []
        for Y_ in TEST_YEARS:
            b_, s_ = base[base.index.year == Y_], R["+美股+盤後情緒"][R["+美股+盤後情緒"].index.year == Y_]
            yr.append((Y_, auc(s_) - auc(b_), acc(s_) - acc(b_)))
        print("  逐年 ΔAUC(美股基準→+盤後情緒)：" + "  ".join(f"{y}:{a:+.4f}" for y, a, _ in yr)
              + f"   為正 {sum(a > 0 for _, a, _ in yr)}/{len(yr)} 年")
        results[f"{target}_{kind}"] = row
        if target == "cc":
            results[f"cc_{kind}_preds"] = {k: R[k] for k in [BASE, "+美股+盤後情緒"]}

# ---------------- 32 天測試集有多雜訊 ----------------
print("\n=== 「只看 32 天測試集」時，加情緒 vs 不加情緒 的準確率差有多雜訊？（目標 cc）===")
for kind in ["LR", "XGB"]:
    P = results[f"cc_{kind}_preds"]
    b, s = P[BASE], P["+美股+盤後情緒"]
    n = len(b)
    correct_b = ((b.p > 0.5) == (b.y == 1)).values.astype(int)
    correct_s = ((s.p > 0.5) == (s.y == 1)).values.astype(int)
    nb = n // 32
    blk_b = correct_b[: nb * 32].reshape(nb, 32).sum(1)
    blk_s = correct_s[: nb * 32].reshape(nb, 32).sum(1)
    dlt = blk_s - blk_b
    print(f"  [{kind}] 連續 {nb} 個 32 天區塊：")
    print(f"    不加情緒 每區塊答對天數：平均 {blk_b.mean():.1f}/32（{blk_b.mean()/32*100:.1f}%），"
          f"標準差 {blk_b.std():.1f} 天（{blk_b.std()/32*100:.1f}pp），最好 {blk_b.max()}/32（{blk_b.max()/32*100:.1f}%），最差 {blk_b.min()}/32")
    print(f"    加情緒 − 不加：平均 {dlt.mean():+.2f} 天，標準差 {dlt.std():.2f} 天；"
          f"多答對≥1天 {np.mean(dlt >= 1)*100:.0f}% 的區塊、少答對≥1天 {np.mean(dlt <= -1)*100:.0f}%、剛好一樣 {np.mean(dlt == 0)*100:.0f}%")
    print(f"    答對 ≥28/32（87.5%）的區塊：不加情緒 {np.sum(blk_b >= 28)} 個、加情緒 {np.sum(blk_s >= 28)} 個（共 {nb} 個）")
    print(f"    答對 ≥29/32（90.6%）的區塊：不加情緒 {np.sum(blk_b >= 29)} 個、加情緒 {np.sum(blk_s >= 29)} 個")

json.dump({k: v for k, v in results.items() if not k.endswith("_preds")},
          open("afterhours_us_control_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nDONE")
