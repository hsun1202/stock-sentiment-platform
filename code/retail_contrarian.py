# -*- coding: utf-8 -*-
"""
散戶「反指標」檢驗：散戶越樂觀，之後幾個月越差？（十年資料 2015-2024，月頻率）
============================================================
理論：Baker & Wurgler 等——散戶情緒造成的錯誤定價會在數週到數月後修正，所以情緒高 → 之後報酬低（反向）。
      先前只測過隔天/一週（沒效果），這裡測 1~3 個月。
每月底 m、每檔股票：
  散戶說的：當月 PTT 淨情緒 (正−負)/(正+負)（依留言時間歸屬；當月表態留言 ≥20 則才算）
  散戶做的：當月融資餘額變化率（月底 / 上月底 − 1）；以及月底券資比
  控制：當月報酬（短期反轉）、過去12個月報酬（動能，跳過最近1月）、log(當月平均成交金額)（規模代理）
未來報酬：m 月底收盤 → m+1、m+2、m+3 月底收盤（還原股價；期間有異常價格日的排除）
兩種檢驗：
  (1) 五分位投組：每月依情緒排序分五組等權，最高組 − 最低組 的平均月報酬（t 值用 Newey-West）
  (2) Fama-MacBeth：每月橫斷面迴歸 未來報酬 ~ 情緒(標準化) + 控制，係數取時間平均，Newey-West t 值
輸出：retail_contrarian.log、retail_contrarian_results.json
python retail_contrarian.py
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str},
                 usecols=["stock_code", "Date", "Close", "Volume", "abnormal_price", "margin_bal", "short_bal",
                          "margin_eligible", "ptt_pre_pos", "ptt_pre_neg", "ptt_intra_pos", "ptt_intra_neg"])
df["Date"] = pd.to_datetime(df["Date"])
df["month"] = df["Date"].dt.to_period("M")
df["pos"] = df.ptt_pre_pos + df.ptt_intra_pos
df["neg"] = df.ptt_pre_neg + df.ptt_intra_neg
df["dollar"] = df.Close * df.Volume
df = df.sort_values(["stock_code", "Date"])
m = df.groupby(["stock_code", "month"]).agg(close=("Close", "last"), pos=("pos", "sum"), neg=("neg", "sum"),
                                            margin=("margin_bal", "last"), short=("short_bal", "last"),
                                            elig=("margin_eligible", "last"), abn=("abnormal_price", "max"),
                                            dollar=("dollar", "mean")).reset_index()
m = m.sort_values(["stock_code", "month"]).reset_index(drop=True)
g = m.groupby("stock_code")
m["ret"] = m.close / g.close.shift(1) - 1
m["mom12"] = g.close.shift(1) / g.close.shift(12) - 1
m["size"] = np.log(m.dollar + 1)
m["ptt_sent"] = np.where(m.pos + m.neg >= 20, (m.pos - m.neg) / (m.pos + m.neg).replace(0, 1), np.nan)
m["margin_chg"] = np.where((m.elig == 1) & (g.margin.shift(1) > 0), m.margin / g.margin.shift(1) - 1, np.nan)
m["margin_chg"] = m["margin_chg"].clip(-0.8, 3)
m["short_margin"] = np.where((m.elig == 1) & (m.margin > 0), (m.short / m.margin).clip(0, 2), np.nan)
abn_fwd = {h: sum(g.abn.shift(-k).fillna(1) for k in range(1, h + 1)) for h in (1, 2, 3)}
for h in (1, 2, 3):
    m[f"fut{h}"] = np.where(abn_fwd[h] == 0, g.close.shift(-h) / m.close - 1, np.nan)
m = m[(m.month >= pd.Period("2015-02", "M")) & (m.month <= pd.Period("2024-11", "M"))]
SIGNALS = {"散戶說(PTT淨情緒)": "ptt_sent", "散戶做(融資月增減)": "margin_chg", "券資比": "short_margin"}
lines, res = [], {}


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def nw_t(x, lags):
    x = pd.Series(x).dropna()
    r = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.params[0]), float(r.tvalues[0]), len(x)


log(f"樣本：{m.stock_code.nunique()} 檔、{m.month.nunique()} 個月（2015-02 ~ 2024-11）")
for name, col in SIGNALS.items():
    cov = m[col].notna().groupby(m.month).sum()
    log(f"\n==== {name}：每月平均有 {cov.mean():.0f} 檔有值 ====")
    res[name] = {}
    for h in (1, 2, 3):
        y = f"fut{h}"
        # (1) 五分位：每月至少 15 檔才排序
        spreads, q_means = [], []
        for mo, x in m[[col, y]].dropna().groupby(m.month):
            if len(x) < 15:
                continue
            q = pd.qcut(x[col].rank(method="first"), 5, labels=False)
            qm = x.groupby(q)[y].mean()
            spreads.append(qm.iloc[-1] - qm.iloc[0])
            q_means.append(qm.values)
        mean_sp, t_sp, n_sp = nw_t(spreads, h)
        qavg = np.nanmean(np.array(q_means), axis=0)
        # (2) Fama-MacBeth
        coefs = []
        for mo, x in m[[col, y, "ret", "mom12", "size"]].dropna().groupby(m.month):
            if len(x) < 15:
                continue
            X = x[[col, "ret", "mom12", "size"]].copy()
            X[col] = (X[col] - X[col].mean()) / (X[col].std() + 1e-12)
            coefs.append(sm.OLS(x[y].values, sm.add_constant(X.values)).fit().params[1])
        b, tb, nb = nw_t(coefs, h)
        res[name][f"{h}m"] = {"spread_high_minus_low": mean_sp, "spread_t": t_sp, "months": n_sp,
                              "quintile_mean": qavg.tolist(), "fm_coef_per_1sd": b, "fm_t": tb}
        log(f"  未來{h}個月｜五分位(最樂觀−最悲觀) {mean_sp:+.2%}（t={t_sp:+.2f}，{n_sp}個月）"
            f"｜各組 " + " ".join(f"{v:+.1%}" for v in qavg) +
            f"｜Fama-MacBeth 每1σ {b:+.2%}（t={tb:+.2f}）")
log("\n判讀：反指標假說 → 最樂觀−最悲觀 < 0、Fama-MacBeth 係數 < 0；|t|>2 約為 5% 顯著")
json.dump(res, open("retail_contrarian_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
open("retail_contrarian.log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
