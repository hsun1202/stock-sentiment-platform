# -*- coding: utf-8 -*-
"""
新方向初探（2026-10-08 使用者討論後提出）：先找「股價波動最大的日子」，回頭看那幾天的新聞，找關鍵字規律，看是什麼在影響股價。
這是探索性、描述性的分析（沒有事先寫定的假說；找到的規律要另外用沒看過的資料驗證）。
資料：final_dataset_v3_tej（265 檔、TEJ 股價、2015-2024）＋ news_sentiment/news_scores.csv（PTT 股板 [新聞] 文章標題 83,002 篇，其中標題比對到個股的才用得上）。
大波動日：個股當天超額報酬（減當天全體平均）的絕對值 > 自己過去 60 天超額報酬標準差的 3 倍，且 > 3%。
新聞歸屬（相對交易日 t）：盤前＝前一交易日 13:30 後 ~ t 日 09:00 前；盤中＝t 日 09:00~13:30；盤後＝t 日 13:30 後 ~ 下一交易日 09:00 前（算在 t 日的「盤後」，同時也是 t+1 的盤前）。
python bigmove_news.py
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


df = pd.read_csv("final_dataset_v3_tej/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                 usecols=["stock_code", "stock_name", "Date", "Close", "abnormal_price"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
bad = (df.abnormal_price.fillna(0) != 0) | g.abnormal_price.shift(1).fillna(0).ne(0)
df["r"] = (df.Close / g.Close.shift(1) - 1).where(~bad)
df["x"] = df.r - df.groupby("Date").r.transform("mean")
df["sd"] = df.groupby("stock_code").x.transform(lambda s: s.rolling(60, min_periods=30).std().shift(1))
df["z"] = df.x / df.sd
df["x_next"] = df.groupby("stock_code").x.shift(-1)
df["big"] = (df.z.abs() > 3) & (df.x.abs() > 0.03)
df["up"], df["dn"] = df.big & (df.x > 0), df.big & (df.x < 0)
V = df[df.z.notna()]
say(f"[股價] {len(V):,} 個股票日、{V.stock_code.nunique()} 檔、2015-2024；大波動日 {int(V.big.sum()):,} 個（占 {V.big.mean():.2%}；大漲 {int(V.up.sum()):,}、大跌 {int(V.dn.sum()):,}）；"
    f"大波動日的超額報酬絕對值中位數 {V.loc[V.big, 'x'].abs().median():.1%}")

# ---------------- 新聞 → 交易日與時段 ----------------
nw = pd.read_csv("news_sentiment/news_scores.csv", dtype=str)
nw = nw[nw.stock_codes.notna()].copy()
nw["t"] = pd.to_datetime(nw.post_time)
nw["title"] = nw.title.str.replace(r"^\s*(Re:\s*)?\[新聞\]\s*", "", regex=True).str.strip()
nw["stock_code"] = nw.stock_codes.str.split(r"[,;| ]+")
nw = nw.explode("stock_code")
nw = nw[nw.stock_code.isin(df.stock_code.unique())]
cal = np.array(sorted(df.Date.unique()), dtype="datetime64[ns]")
d0 = nw.t.dt.normalize().values.astype("datetime64[ns]")
i = np.searchsorted(cal, d0)                       # 第一個 >= 發文日的交易日
is_td = (i < len(cal)) & (cal[np.minimum(i, len(cal) - 1)] == d0)
hm = nw.t.dt.hour * 60 + nw.t.dt.minute
# 盤前(pre)歸 t；盤中(intra)歸 t；盤後(post)歸「下一交易日的盤前」，另記錄它是哪個交易日的盤後
slot = np.where(is_td & (hm < 540), "pre", np.where(is_td & (hm < 810), "intra", "post"))
day_i = np.where(slot == "post", np.where(is_td, i + 1, i), i)     # 影響的交易日（之後第一個開盤）
nw["slot"] = np.where(slot == "post", "pre", slot)                 # 對「影響的交易日」而言都是盤前
nw["after_close_of"] = np.where((slot == "post") & is_td, i, -1)   # 是哪個交易日收盤後發的
nw = nw[day_i < len(cal)].copy()
nw["Date"] = cal[day_i[day_i < len(cal)]]
nw = nw[(nw.Date >= "2015-01-01") & (nw.Date <= "2024-12-31")]
say(f"[新聞] 標題有點名 265 檔的 PTT [新聞] {nw.article_id.nunique():,} 篇（{len(nw):,} 個 新聞×股票）；盤前（含前一天盤後）{(nw.slot == 'pre').mean():.0%}、盤中 {(nw.slot == 'intra').mean():.0%}")

key = ["stock_code", "Date"]
cnt = nw.groupby(key + ["slot"]).size().unstack(fill_value=0).reindex(columns=["pre", "intra"], fill_value=0).reset_index()
V = V.merge(cnt, on=key, how="left").fillna({"pre": 0, "intra": 0})
ac = nw[nw.after_close_of >= 0].assign(Date=lambda z: cal[z.after_close_of.values]).groupby(key).size().rename("post").reset_index()
V = V.merge(ac, on=key, how="left").fillna({"post": 0})

say("\n===== 1) 大波動日有沒有新聞？（有 PTT 新聞的股票日比例）=====")
for lab, m in [("一般日", ~V.big), ("大漲日", V.up), ("大跌日", V.dn)]:
    s = V[m]
    say(f"  {lab}（{len(s):,}）：盤前有新聞 {(s.pre > 0).mean():.1%}｜盤中 {(s.intra > 0).mean():.1%}｜當天收盤後 {(s.post > 0).mean():.1%}｜三者任一 {((s.pre + s.intra + s.post) > 0).mean():.1%}")
top = V[V.stock_code.isin(V.groupby('stock_code').pre.sum().nlargest(30).index)]
say(f"  （只看新聞最多的 30 檔）大波動日三者任一有新聞 {((top[top.big].pre + top[top.big].intra + top[top.big].post) > 0).mean():.1%}；一般日 {((top[~top.big].pre + top[~top.big].intra + top[~top.big].post) > 0).mean():.1%}")

say("\n===== 2) 反過來：有新聞的日子，大波動的機率 =====")
base = V.big.mean()
for lab, m in [("沒有新聞", (V.pre + V.intra) == 0), ("盤前有新聞", V.pre > 0), ("盤中有新聞", V.intra > 0)]:
    s = V[m]
    say(f"  {lab}（{len(s):,}）：大波動 {s.big.mean():.2%}（全體 {base:.2%}，{s.big.mean() / base:.1f} 倍）｜大漲 {s.up.mean():.2%}、大跌 {s.dn.mean():.2%}｜當天超額報酬絕對值平均 {s.x.abs().mean():.2%}")
s = V[V.post > 0]
say(f"  當天收盤後才有新聞（{len(s):,}）：當天就大波動的 {s.big.mean():.2%}（{s.big.mean() / base:.1f} 倍）→ 新聞在報導已經發生的大波動")

# ---------------- 3) 關鍵字：大漲日／大跌日的盤前+盤中新聞 vs 一般日 ----------------
N = nw.merge(V[key + ["big", "up", "dn", "x", "x_next", "stock_name"]], on=key, how="inner")
names = set(df.stock_name.dropna().unique())
name_re = re.compile("|".join(sorted((re.escape(n) for n in names if len(n) >= 2), key=len, reverse=True)))


def grams(title):
    t = name_re.sub("", title)
    t = re.sub(r"[A-Za-z0-9０-９％%\.\-\+]+", " ", t)
    out = set()
    for seg in re.split(r"[^一-鿿]+", t):
        for n in (2, 3, 4):
            for k in range(len(seg) - n + 1):
                out.add(seg[k:k + n])
    return out


N["g"] = N.title.map(grams)


def logodds(a, b, min_a=12, topn=30):
    ca, cb = Counter(x for s in a for x in s), Counter(x for s in b for x in s)
    na, nb = len(a), len(b)
    rows = []
    for w, c in ca.items():
        if c >= min_a:
            pa, pb = (c + 0.5) / (na + 1), (cb.get(w, 0) + 0.5) / (nb + 1)
            lo = np.log(pa / (1 - pa)) - np.log(pb / (1 - pb))
            se = np.sqrt(1 / (c + 0.5) + 1 / (cb.get(w, 0) + 0.5))
            rows.append((w, c, cb.get(w, 0), pa / pb, lo / se))
    r = pd.DataFrame(rows, columns=["詞", "大波動篇數", "一般日篇數", "倍數", "z"]).sort_values("z", ascending=False)
    keep, seen = [], []
    for _, x in r.iterrows():                      # 去掉互相包含的重複片語（留 z 最高的）
        if not any((x["詞"] in s) or (s in x["詞"]) for s in seen):
            keep.append(x); seen.append(x["詞"])
        if len(keep) >= topn:
            break
    return pd.DataFrame(keep)


norm = N[~N.big].g.tolist()
for lab, m in [("大漲日", N.up), ("大跌日", N.dn)]:
    a = N[m].g.tolist()
    r = logodds(a, norm)
    say(f"\n===== 3) {lab}的新聞標題（{len(a):,} 篇）比一般日（{len(norm):,} 篇）特別常出現的詞 =====")
    say("  " + "、".join(f"{x['詞']}({int(x['大波動篇數'])}篇,{x['倍數']:.1f}倍)" for _, x in r.iterrows()))

# ---------------- 4) 依事件類別 ----------------
CATS = {
    "營收/財報/獲利": r"營收|財報|獲利|EPS|每股|盈餘|毛利|虧損|由盈轉虧|轉盈|季報|年報|財測|業績",
    "法說/展望": r"法說|展望|下修|上修|看淡|看好|保守|樂觀",
    "訂單/客戶/產品": r"訂單|接單|打入|供應鏈|出貨|量產|新品|蘋果|iPhone|特斯拉|輝達|NVIDIA|客戶",
    "漲價/缺貨/報價": r"漲價|缺貨|報價|調漲|降價|跌價|供不應求",
    "併購/入股/公開收購": r"併購|收購|入股|合併|公開收購|私募|下市",
    "外資/法人/目標價": r"外資|投信|法人|目標價|評等|調升|調降|買超|賣超|喊進|喊買",
    "漲停/跌停/飆股": r"漲停|跌停|飆|暴漲|暴跌|重挫|大漲|大跌|噴|崩",
    "處置/注意/違約/弊案": r"處置|注意股|違約|掏空|弊案|檢調|搜索|起訴|裁罰|停工|火災|爆炸|罷工",
    "股利/減資/庫藏股": r"股利|配息|配股|除權|除息|減資|增資|庫藏股|殖利率",
    "政策/關稅/匯率/疫情": r"關稅|政策|補貼|匯率|升息|降息|疫情|制裁|禁令|貿易戰",
}
say("\n===== 4) 依新聞類別（標題關鍵字歸類，一篇可屬多類；只看盤前與盤中的新聞）=====")
say(f"  全部新聞×股票 {len(N):,}：當天大波動 {N.big.mean():.1%}（沒有新聞的日子約 {V[(V.pre + V.intra) == 0].big.mean():.1%}）")
rows = []
pre = N[N.slot == "pre"]
for c, pat in CATS.items():
    m = N.title.str.contains(pat, regex=True)
    s, sp = N[m], pre[pre.title.str.contains(pat, regex=True)]
    rows.append((c, len(s), s.big.mean(), s.up.mean(), s.dn.mean(), s.x.mean(), s.x.abs().mean(), sp.x.mean(), len(sp)))
other = N[~N.title.str.contains("|".join(CATS.values()), regex=True)]
rows.append(("（都不屬於）", len(other), other.big.mean(), other.up.mean(), other.dn.mean(), other.x.mean(), other.x.abs().mean(), np.nan, 0))
for r in sorted(rows, key=lambda z: -z[2]):
    say(f"  {r[0]:<14s} {r[1]:>5,} 篇｜當天大波動 {r[2]:.1%}（大漲 {r[3]:.1%}／大跌 {r[4]:.1%}）｜當天超額報酬平均 {r[5]:+.2%}、絕對值 {r[6]:.2%}"
        + (f"｜盤前新聞→當天超額 {r[7]:+.2%}（{r[8]:,} 篇）" if r[8] else ""))

# ---------------- 5) 例子 ----------------
say("\n===== 5) 例子：波動最大且有盤前/盤中新聞的 25 個股票日 =====")
ex = N[N.big].sort_values("x", key=lambda s: s.abs(), ascending=False).drop_duplicates(key).head(25)
for _, r in ex.iterrows():
    say(f"  {r.Date.date()} {r.stock_code} {r.stock_name}  超額 {r.x:+.1%}｜[{r.slot}] {r.title[:38]}")
N.drop(columns=["g"]).to_csv("bigmove_news_matched.csv", index=False, encoding="utf-8-sig")
open("bigmove_news.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
