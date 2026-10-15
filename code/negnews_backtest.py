# -*- coding: utf-8 -*-
"""
「負面新聞之後會續跌」——能不能執行？（cnyes_bigmove 的發現：開盤前已發布的負面新聞，影響日之後 4 天再 −0.43%〔889 檔〕／−0.11%〔265 檔〕）
這批資料兩組股票都已看過，所以這裡不是驗證，只是把「理論上的續跌」換成「實際能做的」算一次。
事先寫定的規則（2026-10-10，跑之前寫下，不搜尋參數）：
  事件：影響日開盤前已發布的負面語氣新聞（同 cnyes_bigmove.py 的 NEG 詞表、且不含正面詞）；同一股票日只算一次。
  放空版：影響日收盤放空、4 個交易日後收盤回補。只取可信用交易的股票（margin_eligible=1）；影響日收跌停 → 視為空不到，剔除。
          成本：來回 0.47%／0.585%（另有借券費與融券限制，未計）。
  持有者版（不需要放空）：已經持有的人，影響日收盤賣掉 vs 續抱 4 天，差多少（＝之後 4 天的原始報酬與超額報酬）。
  報：265 檔、889 檔分開；分年；另報「正面新聞」當對照。
python negnews_backtest.py
"""
import numpy as np, pandas as pd
POS = r"創新高|新高|年增|月增|季增|成長|大增|倍增|優於預期|超乎預期|轉盈|由虧轉盈|獲利創|調升|上修|喊買|喊進|看好|看旺|樂觀|受惠|利多|報喜|接單|大單|訂單滿|滿載|漲價|調漲|供不應求|缺貨|買超|加碼|回升|回溫|復甦|強勁|亮眼|暢旺|旺到|突破|擴產|打入|通過認證|得標"
NEG = r"創新低|新低|年減|月減|季減|衰退|大減|腰斬|不如預期|低於預期|轉虧|由盈轉虧|虧損|調降|下修|降評|看淡|看壞|保守|悲觀|利空|砍單|減產|跌價|降價|供過於求|賣超|減碼|疲弱|低迷|重挫|暴跌|崩跌|下滑|下跌|裁員|停工|違約|掏空|弊案|搜索|起訴|裁罰|罰鍰|火災|爆炸|下市|跳票|遭|示警|警訊"
LOG = []
def say(s=""):
    print(s, flush=True); LOG.append(s)
key = ["stock_code", "Date"]
px = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str, "limit_flag": str}, parse_dates=["Date"],
                 usecols=["stock_code", "Date", "Close", "limit_flag", "abnormal_price", "margin_eligible", "in265"]).sort_values(key).reset_index(drop=True)
g = px.groupby("stock_code")
px["r"] = px.Close / g.Close.shift(1) - 1
px["mkt"] = px.groupby("Date").r.transform("mean")
cs = np.log1p(px.mkt.fillna(0)).groupby(px.stock_code).cumsum()
px["m4"] = np.expm1(cs.groupby(px.stock_code).shift(-4) - cs)
px["bad"] = sum(g.abnormal_price.shift(-k).fillna(0).ne(0) for k in range(1, 5)) > 0
px["raw4"] = (g.Close.shift(-4) / px.Close - 1).where(~px.bad)
px["ex4"] = px.raw4 - px.m4
M = pd.read_pickle("cnyes_matched.pkl")
M = M[M.slot != "盤中"]
pos, neg = M.title.str.contains(POS, regex=True), M.title.str.contains(NEG, regex=True)
M["tone"] = np.where(neg & ~pos, "負面", np.where(pos & ~neg, "正面", "不分"))
E = M[M.tone != "不分"].drop_duplicates(key + ["tone"]).merge(px[key + ["limit_flag", "margin_eligible", "in265", "raw4", "ex4"]], on=key, how="inner")
E["ym"], E["year"] = E.Date.dt.to_period("M"), E.Date.dt.year
def tm(v, ym):
    z = pd.DataFrame({"v": v, "m": ym}).dropna(); mm = z.groupby("m").v.mean()
    return z.v.mean(), mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm))), len(z)
for grp, lab in [(1, "原 265 檔"), (0, "另外 889 檔")]:
    for tone in ["負面", "正面"]:
        s = E[(E.in265 == grp) & (E.tone == tone)]
        a, t, n = tm(s.ex4, s.ym); r_, tr, _ = tm(s.raw4, s.ym)
        say(f"\n===== {lab}・開盤前已發布的{tone}新聞 {n:,} 個股票日 =====")
        say(f"  持有者版：影響日收盤後續抱 4 天 → 原始 {r_:+.2%}（t={tr:+.1f}）、超額 {a:+.2%}（t={t:+.1f}）｜之後 4 天下跌的比例 {(s.raw4 < 0).mean():.1%}")
        if tone == "負面":
            say(f"    影響日收跌停 {(s.limit_flag == '-').mean():.1%}；可信用交易 {(s.margin_eligible == 1).mean():.1%}")
            k = s[(s.limit_flag != "-") & (s.margin_eligible == 1)]
            a, t, n = tm(-k.ex4, k.ym); r_, tr, _ = tm(-k.raw4, k.ym)
            say(f"  放空版（剔除跌停、只取可信用交易）n={n:,}｜毛：原始 {r_:+.2%}（t={tr:+.1f}）、超額 {a:+.2%}（t={t:+.1f}）"
                f"｜扣0.47%：原始 {r_ - 0.0047:+.2%}（t={tm(-k.raw4 - 0.0047, k.ym)[1]:+.1f}）、超額 {a - 0.0047:+.2%}（t={tm(-k.ex4 - 0.0047, k.ym)[1]:+.1f}）｜扣0.585%：超額 {a - 0.00585:+.2%}")
            yr = k.groupby("year").agg(n=("ex4", "count"), ex=("ex4", "mean"))
            say("    放空版分年 超額(毛，放空方向)：" + " ".join(f"{y}:{-r.ex:+.1%}({int(r.n)})" for y, r in yr.iterrows()) + f"｜為正 {(-yr.ex > 0).sum()}/{len(yr)}；扣0.47%後為正 {((-yr.ex - 0.0047) > 0).sum()}/{len(yr)}")
open("negnews_backtest.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
