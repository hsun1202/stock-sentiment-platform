# -*- coding: utf-8 -*-
"""
格蘭傑因果檢定：PTT 情緒/討論熱度 與 個股報酬，誰領先誰？（十年資料 2015-2024）
============================================================
時間對齊（依留言自己的時間，build_dataset_v3.py 已分好時段）：
    交易日 t 的 PTT = 前一交易日收盤(13:30)後 ~ t 收盤前發的留言（ptt_pre + ptt_intra）
    交易日 t 的報酬 r_t = log(Close_t / Close_{t-1})   → 兩者涵蓋同一段時間
變數：sent_t = (正−負)/(正+負+1)（沒有表態留言時 = 0）；att_t = log(1+留言數)；報酬
檢定（落後 p = 1, 3, 5 天）：
    A. 股價 → 情緒： y=sent_t（或 att_t），控制自己的落後期後，r_{t-1..t-p} 聯合是否顯著
    B. 情緒 → 股價： y=r_t，控制自己的落後期後，sent_{t-1..t-p}（或 att）聯合是否顯著
兩種做法：
    (1) 合併迴歸（panel）：所有股票、個股固定效果（去平均）、標準誤依交易日群聚（處理同日共同衝擊），Wald 檢定
        另外只用「討論量前30檔」再做一次（多數日子有留言，情緒變數比較有意義）
    (2) 逐檔：討論量前30檔各自做 statsmodels grangercausalitytests（F檢定），數幾檔在 5% 顯著
輸出：granger_ptt.log、granger_ptt_results.json
python granger_ptt.py
"""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")
LAGS = (1, 3, 5)

df = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str},
                 usecols=["stock_code", "stock_name", "Date", "Close", "abnormal_price",
                          "ptt_pre_n", "ptt_pre_pos", "ptt_pre_neg", "ptt_intra_n", "ptt_intra_pos", "ptt_intra_neg"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
df["r"] = np.log(df.Close / g.Close.shift(1))
n = df.ptt_pre_n + df.ptt_intra_n
p = df.ptt_pre_pos + df.ptt_intra_pos
q = df.ptt_pre_neg + df.ptt_intra_neg
df["sent"] = (p - q) / (p + q + 1)
df["att"] = np.log1p(n)
df.loc[df.abnormal_price == 1, "r"] = np.nan  # 減資/停牌復牌等異常報酬不用
for v in ("r", "sent", "att"):
    for k in range(1, max(LAGS) + 1):
        df[f"{v}_l{k}"] = df.groupby("stock_code")[v].shift(k)

tot = df.groupby("stock_code").apply(lambda x: (x.ptt_pre_n + x.ptt_intra_n).sum())
top30 = tot.sort_values(ascending=False).index[:30].tolist()
names = df.drop_duplicates("stock_code").set_index("stock_code").stock_name
lines, res = [], {}


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def panel_test(data, y, own, cause, p):
    cols = [f"{own}_l{k}" for k in range(1, p + 1)] + [f"{cause}_l{k}" for k in range(1, p + 1)]
    d = data[[y, "stock_code", "Date"] + cols].dropna()
    Z = d[[y] + cols]
    Z = Z - Z.groupby(d["stock_code"]).transform("mean")  # 個股固定效果
    X = sm.add_constant(Z[cols])
    m = sm.OLS(Z[y], X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["Date"])[0]})
    names_c = [f"{cause}_l{k}" for k in range(1, p + 1)]
    R = np.zeros((p, len(X.columns)))
    for i, c in enumerate(names_c):
        R[i, list(X.columns).index(c)] = 1
    w = m.wald_test(R, scalar=True)
    # 效果量：原因變數增加1個標準差，結果變數變化幾個標準差（第1期落後的係數）
    eff = m.params[f"{cause}_l1"] * d[f"{cause}_l1"].std() / d[y].std()
    return float(w.statistic), float(w.pvalue), float(eff), int(len(d))


for scope, data in (("全部股票", df), ("討論量前30檔", df[df.stock_code.isin(top30)])):
    log(f"\n==== 合併迴歸（{scope}；個股固定效果、標準誤依交易日群聚） ====")
    for sv in ("sent", "att"):
        label = "情緒" if sv == "sent" else "討論熱度"
        for p_ in LAGS:
            fa, pa, ea, na = panel_test(data, sv, sv, "r", p_)
            fb, pb, eb, nb = panel_test(data, "r", "r", sv, p_)
            res[f"{scope}_{sv}_p{p_}"] = {"price_to_ptt": {"F": fa, "p": pa, "eff_sd": ea, "n": na},
                                          "ptt_to_price": {"F": fb, "p": pb, "eff_sd": eb, "n": nb}}
            log(f"  [{label}] 落後{p_}天｜股價→{label}: F={fa:8.1f} p={pa:.2g} 效果(1σ報酬→{ea:+.3f}σ)"
                f"｜{label}→股價: F={fb:6.2f} p={pb:.2g} 效果(1σ{label}→{eb:+.4f}σ)")

log("\n==== 逐檔格蘭傑檢定（討論量前30檔，F檢定，5%顯著的檔數） ====")
per = []
for c in top30:
    x = df[df.stock_code == c][["r", "sent", "att"]].dropna()
    row = {"code": c, "name": names[c], "days": len(x)}
    for sv in ("sent", "att"):
        for p_ in (1, 5):
            # grangercausalitytests(data[[y, x]]) 檢定「第2欄 → 第1欄」
            pa = grangercausalitytests(x[[sv, "r"]], maxlag=[p_], verbose=False)[p_][0]["ssr_ftest"][1]
            pb = grangercausalitytests(x[["r", sv]], maxlag=[p_], verbose=False)[p_][0]["ssr_ftest"][1]
            row[f"price_to_{sv}_p{p_}"], row[f"{sv}_to_price_p{p_}"] = pa, pb
    per.append(row)
per = pd.DataFrame(per)
for sv, label in (("sent", "情緒"), ("att", "討論熱度")):
    for p_ in (1, 5):
        a = (per[f"price_to_{sv}_p{p_}"] < 0.05).sum()
        b = (per[f"{sv}_to_price_p{p_}"] < 0.05).sum()
        log(f"  [{label}] 落後{p_}天：股價→{label} 顯著 {a}/30 檔｜{label}→股價 顯著 {b}/30 檔（純巧合預期約1.5檔）")
res["per_stock"] = per.to_dict(orient="records")
log("\n討論量前30檔：" + "、".join(f"{names[c]}" for c in top30))
json.dump(res, open("granger_ptt_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
open("granger_ptt.log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
