# -*- coding: utf-8 -*-
"""
大盤散戶情緒溫度計（3/3）：建立每日指數，檢驗能否預測加權指數（2015-2024）
============================================================
成分（交易日 t；只用 t 收盤前可得的資訊）：
  散戶說  ptt_mood  ：閒聊串抽樣留言（每日≤300則）淨情緒 (正−負)/(正+負)，取5日平均
  散戶關注 ptt_att  ：閒聊串全部留言數 log，減去過去60日平均（異常熱度）
  散戶做  margin_buy：全市場融資買進金額 / 大盤成交金額（散戶借錢買股的熱度），取5日平均
          margin_chg：全市場融資餘額 5 日變化率
  綜合指數 retail_idx：上面四項各自「只用過去資料」標準化（擴張視窗 z 分數，至少250日）後平均
    （融資餘額與買進佔比在 t 當晚 21:00 公布，嚴格來說 t 收盤後才可得；預測目標從 t+1 收盤起算的版本見 lag1 檢驗）
預測目標：加權指數（證交所官方收盤）未來報酬 t 收盤 → t+h 收盤，h = 1、5、20、60 個交易日
檢驗：
  (1) 迴歸 未來報酬 ~ 指標(z) + 過去5日與20日大盤報酬（控制「情緒反應價格」），Newey-West 標準誤（lag=h）
  (2) 五分位：依指標分五組（極度恐慌 → 極度貪婪），各組之後的平均報酬
  (3) 情緒反應價格：今天的指標 vs 過去5日大盤報酬的相關係數
輸出：market_sentiment/retail_index_daily.csv、market_sentiment_analysis.log、market_sentiment_results.json
python market_sentiment_analysis.py
"""
import glob
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

D = "market_sentiment"
tx = pd.read_csv(f"{D}/taiex_official.csv", parse_dates=["Date"]).set_index("Date")
lab = pd.read_csv(f"{D}/chat_labels.csv", parse_dates=["Date"])
vol = pd.read_csv(f"{D}/chat_daily_volume.csv", parse_dates=["Date"]).set_index("Date")

# ---- 散戶說：閒聊情緒 ----
lab["pos"] = (lab.text_sentiment == "positive").astype(int)
lab["neg"] = (lab.text_sentiment == "negative").astype(int)
day = lab.groupby("Date")[["pos", "neg"]].sum()
day["n_lab"] = lab.groupby("Date").size()
df = tx.join(day, how="left").join(vol, how="left")
df["mood_raw"] = (df.pos - df.neg) / (df.pos + df.neg)
df["ptt_mood"] = df.mood_raw.rolling(5, min_periods=3).mean()
la = np.log1p(df.chat_n)
df["ptt_att"] = la - la.shift(1).rolling(60, min_periods=20).mean()

# ---- 散戶做：全市場融資 ----
rows = []
for f in sorted(glob.glob("chip_raw/margin/*.json")):
    j = json.load(open(f, encoding="utf-8"))
    t = [t for t in (j.get("tables") or []) if t.get("data") and len(t["data"]) < 10 and "信用交易統計" in (t.get("title") or "")]
    if not t:
        continue
    r = {x[0]: x for x in t[0]["data"]}.get("融資金額(仟元)")
    if r:
        num = lambda s: float(str(s).replace(",", ""))  # noqa: E731
        rows.append({"Date": pd.Timestamp(f[-13:-5]), "mg_buy": num(r[1]) * 1000, "mg_bal": num(r[5]) * 1000})
mg = pd.DataFrame(rows).set_index("Date")
df = df.join(mg, how="left")
df["margin_buy"] = (df.mg_buy / df.turnover).rolling(5, min_periods=3).mean()
df["margin_chg"] = df.mg_bal / df.mg_bal.shift(5) - 1
df = df.loc["2015-01-01":"2024-12-31"]
COMP = ["ptt_mood", "ptt_att", "margin_buy", "margin_chg"]


def ez(s, min_n=250):  # 擴張視窗 z 分數：只用 t 以前（含 t）的資料
    mu = s.expanding(min_n).mean()
    sd = s.expanding(min_n).std()
    return (s - mu) / sd


for c in COMP:
    df[c + "_z"] = ez(df[c])
df["retail_idx"] = df[[c + "_z" for c in COMP]].mean(axis=1, skipna=True)
df["retail_say"] = df[["ptt_mood_z", "ptt_att_z"]].mean(axis=1)
df["retail_do"] = df[["margin_buy_z", "margin_chg_z"]].mean(axis=1)
lr = np.log(df.taiex)
df["past5"] = lr - lr.shift(5)
df["past20"] = lr - lr.shift(20)
H = (1, 5, 20, 60)
for h in H:
    df[f"fut{h}"] = lr.shift(-h) - lr
    df[f"fut{h}_lag1"] = lr.shift(-h - 1) - lr.shift(-1)   # 從 t+1 收盤起算（融資資料 t 晚上才公布，這版完全不偷看）
df.to_csv(f"{D}/retail_index_daily.csv", encoding="utf-8-sig")

SIG = {"綜合散戶情緒指數": "retail_idx", "散戶說(閒聊情緒+熱度)": "retail_say", "散戶做(融資熱度+融資增減)": "retail_do",
       "  閒聊淨情緒": "ptt_mood_z", "  閒聊異常熱度": "ptt_att_z", "  融資買進佔成交": "margin_buy_z", "  融資餘額5日變化": "margin_chg_z"}
lines, res = [], {}


def log(s=""):
    print(s, flush=True)
    lines.append(s)


ok = df.retail_idx.notna()
log(f"有效期間 {df.index[ok].min().date()} ~ {df.index[ok].max().date()}（{ok.sum()} 個交易日；z 分數需 250 日暖機）")
log(f"閒聊每日抽樣分類 中位數 {df.n_lab.median():.0f} 則；閒聊淨情緒平均 {df.mood_raw.mean():+.3f}")
log("\n【情緒反應價格】今天的指標 與 過去5日大盤報酬 的相關：" +
    "；".join(f"{k.strip()} {df[v].corr(df.past5):+.2f}" for k, v in SIG.items()))
for target_suffix, tname in (("", "t收盤→t+h收盤"), ("_lag1", "t+1收盤→t+1+h收盤（完全不偷看）")):
    log(f"\n==== 迴歸：未來大盤報酬 ~ 指標(1σ) + 過去5日/20日報酬；Newey-West；{tname} ====")
    for name, col in SIG.items():
        out = []
        for h in H:
            y = f"fut{h}{target_suffix}"
            d = df[[y, col, "past5", "past20"]].dropna()
            if len(d) < 100:
                continue
            m = sm.OLS(d[y], sm.add_constant(d[[col, "past5", "past20"]])).fit(cov_type="HAC", cov_kwds={"maxlags": max(h, 5)})
            out.append((h, float(m.params[col]), float(m.tvalues[col])))
            res.setdefault(tname, {}).setdefault(name.strip(), {})[f"h{h}"] = {"coef_per_1sd": out[-1][1], "t": out[-1][2], "n": len(d)}
        log(f"  {name:22s} " + "｜".join(f"{h}日 {b:+.2%}(t={t:+.2f})" for h, b, t in out))
log("\n==== 五分位（依綜合散戶情緒指數；1=極度恐慌 … 5=極度貪婪），之後大盤平均報酬 ====")
for col, nm in (("retail_idx", "綜合"), ("retail_say", "散戶說"), ("retail_do", "散戶做")):
    d = df[[col] + [f"fut{h}" for h in H]].dropna()
    if len(d) < 100:
        continue
    q = pd.qcut(d[col], 5, labels=[1, 2, 3, 4, 5])
    t = d.groupby(q, observed=True)[[f"fut{h}" for h in H]].mean()
    for h in H:
        res.setdefault("quintile", {}).setdefault(nm, {})[f"h{h}"] = t[f"fut{h}"].tolist()
    log(f"  [{nm}] " + "｜".join(f"{h}日: " + " ".join(f"{v:+.2%}" for v in t[f'fut{h}']) for h in H))
json.dump(res, open("market_sentiment_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
open("market_sentiment_analysis.log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
