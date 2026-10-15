# -*- coding: utf-8 -*-
"""
「情緒當開關」的穩健性檢查（sentiment_regime_test.py 三個主要檢定都通過之後做；2026-10-09，跑之前寫定要檢查什麼）
問題：(1) 情緒狀態會連續好幾天相同，逐日 t 可能高估顯著性 → 改用 Newey-West（HAC，落後 20 天）標準誤。
      (2) 情緒跟著行情走（恐慌高＝剛大跌、波動大）。技術面訊號好壞的差異，會不會只是「市場狀態」造成、跟情緒本身無關？
          → 迴歸：隔天技術面 Rank IC ~ 情緒狀態（用過去 250 天標準化的連續值）＋ 市場控制變數
            控制：過去 5 天大盤報酬、過去 20 天大盤報酬、過去 20 天大盤波動、今天大盤報酬的絕對值、過去 5 天平均橫斷面離散度（個股報酬的截面標準差）。
          判讀（事先寫定）：加入控制後情緒係數仍顯著（HAC |t|>2）→ 情緒有價格以外的資訊；不顯著 → 開關其實是市場狀態，情緒只是它的影子。
      (3) 三個狀態一起放，看誰留下來。
python sentiment_regime_robust.py
"""
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


raw = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                  usecols=["stock_code", "Date", "Open", "Close", "abnormal_price", "margin_bal"])
raw = raw.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = raw.groupby("stock_code")
raw["r_next"] = (g.Close.shift(-1) / g.Open.shift(-1) - 1) * 1e4
abn = raw.abnormal_price.fillna(0) != 0
raw["r"] = g.Close.pct_change().where(~(abn | abn.groupby(raw.stock_code).shift(1).fillna(False)))
rows = []
for Y in range(2018, 2025):
    o = pickle.load(open(f"daytrade_preds/B_{Y}.pkl", "rb"))
    rows.append(pd.DataFrame({"i": o["idx"], "p": o["P"]["技術面"]}))
pr = pd.concat(rows)
pr["Date"], pr["r"] = raw.Date.values[pr.i], raw.r_next.values[pr.i]
pr = pr.dropna(subset=["r"])
pr["rp"], pr["rr"] = pr.groupby("Date").p.rank(), pr.groupby("Date").r.rank()
ic = pr.groupby("Date").apply(lambda z: z.rp.corr(z.rr))

mk = raw.groupby("Date").r.mean()
disp = raw.groupby("Date").r.std()
em = pd.read_csv("emotion/daily_emotion_index.csv", encoding="utf-8-sig", parse_dates=["Date"]).set_index("Date").sort_index()
mb = raw.groupby("Date").margin_bal.sum()
idx = mk.index
X = pd.DataFrame(index=idx)
X["S1多空"] = (em["看多"] - em["看空"]).reindex(idx).rolling(5, min_periods=3).mean()
X["S2恐慌"] = em["恐慌"].reindex(idx).rolling(5, min_periods=3).mean()
X["S3融資"] = mb.reindex(idx).pct_change(20)
for c in ["S1多空", "S2恐慌", "S3融資"]:        # 用過去 250 天（不含今天）標準化
    m_, s_ = X[c].rolling(250, min_periods=120).mean().shift(1), X[c].rolling(250, min_periods=120).std().shift(1)
    X[c] = ((X[c] - m_) / s_).clip(-3, 3)
X["大盤5日報酬"] = mk.rolling(5).sum()
X["大盤20日報酬"] = mk.rolling(20).sum()
X["大盤20日波動"] = mk.rolling(20).std()
X["今天大盤|報酬|"] = mk.abs()
X["5日橫斷面離散度"] = disp.rolling(5).mean()
CTRL = ["大盤5日報酬", "大盤20日報酬", "大盤20日波動", "今天大盤|報酬|", "5日橫斷面離散度"]
for c in CTRL:
    X[c] = (X[c] - X[c].rolling(250, min_periods=120).mean().shift(1)) / X[c].rolling(250, min_periods=120).std().shift(1)
D = X.join(ic.rename("ic"), how="inner").dropna()
say(f"[樣本] {len(D):,} 個交易日（{D.index.min().date()}~{D.index.max().date()}）；每日技術面 Rank IC 平均 {D.ic.mean():.4f}、標準差 {D.ic.std():.4f}、一階自我相關 {D.ic.autocorr():.2f}")
say("  情緒狀態與市場狀態的相關：" + "；".join(f"{s}–{c} {D[s].corr(D[c]):+.2f}" for s in ["S1多空", "S2恐慌", "S3融資"] for c in ["大盤5日報酬", "大盤20日波動"]))


def ols(cols, label):
    r = sm.OLS(D.ic, sm.add_constant(D[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
    say(f"  {label:<26s}" + "｜".join(f"{c} {r.params[c]:+.4f}(t={r.tvalues[c]:+.2f})" for c in cols if c.startswith("S")) + f"｜R² {r.rsquared:.3f}")
    return r


say("\n===== 隔天技術面 Rank IC 對情緒狀態（每 1 個標準差）的迴歸；Newey-West 標準誤（落後 20 天）=====")
for s in ["S1多空", "S2恐慌", "S3融資"]:
    ols([s], f"{s}：不加控制")
    ols([s] + CTRL, f"{s}：加市場控制")
    for a, b in [("2018", "2020"), ("2021", "2024")]:
        d = D[(D.index >= a) & (D.index <= b + "-12-31")]
        r = sm.OLS(d.ic, sm.add_constant(d[[s] + CTRL])).fit(cov_type="HAC", cov_kwds={"maxlags": 20})
        say(f"      {a}~{b} 加控制：{r.params[s]:+.4f}（t={r.tvalues[s]:+.2f}）")
say("")
ols(["S1多空", "S2恐慌", "S3融資"], "三個一起：不加控制")
r = ols(["S1多空", "S2恐慌", "S3融資"] + CTRL, "三個一起：加市場控制")
say("  控制變數係數：" + "｜".join(f"{c} {r.params[c]:+.4f}(t={r.tvalues[c]:+.2f})" for c in CTRL))
r0 = sm.OLS(D.ic, sm.add_constant(D[CTRL])).fit()
say(f"  只有市場控制的 R² {r0.rsquared:.3f} → 加三個情緒狀態 {r.rsquared:.3f}")
open("sentiment_regime_robust.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
