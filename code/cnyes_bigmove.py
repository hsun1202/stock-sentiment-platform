# -*- coding: utf-8 -*-
"""
大波動日 → 回頭看媒體新聞（鉅亨網）→ 類別／語氣／關鍵字規律 → 是什麼在影響；並與公司重大訊息合併看能解釋多少大波動日。
資料：cnyes_matched.pkl（cnyes_build.py）、final_dataset_v3_tejall25（TEJ 還原價）、mops_bigmove_matched*.csv（重大訊息對到的股票日）。
用法：python cnyes_bigmove.py        # 原 265 檔：探索（規則、關鍵字只在 2015~2019 找；2020~2025 並列當第一層驗證）
      python cnyes_bigmove.py out    # 另外 889 檔：驗證（規則一字不改）
定義同 mops_bigmove.py：超額＝個股−全市場等權；大波動日＝|超額|>自己 60 天標準差 3 倍且>3%；
  影響日＝開盤前與盤中的新聞算當天、收盤後的算下一個交易日。「先有新聞」只指影響日開盤前就已發布的（前一天收盤後＋當天盤前）。
類別與語氣詞表是看報酬之前先寫好的（2026-10-09）：
  語氣：正面詞／負面詞各一組（見 POS、NEG）；一則標題只含正面詞＝正面、只含負面詞＝負面、兩種都有或都沒有＝不分。
事先寫定、要在 889 檔上驗的主要檢定（3 個，Bonferroni p<0.0167；t 依影響日月份群聚）：
  N1：開盤前已發布的「正面」新聞，影響日超額報酬 > 0。
  N2：開盤前已發布的「負面」新聞，影響日超額報酬 < 0。
  N3：開盤前已發布的新聞，正面 − 負面 的影響日超額報酬差 > 0，且之後 4 天不反轉（之後 4 天的差 ≥ 0 視為不反轉，只報數字）。
python cnyes_bigmove.py [out]
"""
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

LOG = []
OUTS = len(sys.argv) > 1 and sys.argv[1] == "out"
TAG = "_out889" if OUTS else ""


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


df = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"],
                 usecols=["stock_code", "stock_name", "Date", "Open", "Close", "abnormal_price", "in265"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
bad = (df.abnormal_price.fillna(0) != 0) | g.abnormal_price.shift(1).fillna(0).ne(0)
df["r"] = (df.Close / g.Close.shift(1) - 1).where(~bad)
df["r_on"] = (df.Open / g.Close.shift(1) - 1).where(~bad)
df["r_id"] = (df.Close / df.Open - 1).where(~bad)
for c in ("r", "r_on", "r_id"):
    df["x" + c[1:]] = df[c] - df.groupby("Date")[c].transform("mean")
df = df.rename(columns={"x": "x"})
df = df[df.in265 == (0 if OUTS else 1)].reset_index(drop=True)
g = df.groupby("stock_code")
df["sd"] = g.x.transform(lambda s: s.rolling(60, min_periods=30).std().shift(1))
df["big"] = ((df.x / df.sd).abs() > 3) & (df.x.abs() > 0.03)
df["up"], df["dn"] = df.big & (df.x > 0), df.big & (df.x < 0)
df["x_m1"] = g.x.shift(1)
df["post1_4"] = sum(g.x.shift(-k) for k in range(1, 5))
df["pre5"] = g.x.transform(lambda s: s.rolling(5, min_periods=4).sum().shift(1))
V = df[df.sd.notna()].copy()
base = V.big.mean()
say(f"[{'驗證：另外 889 檔' if OUTS else '探索：原 265 檔'}] {V.stock_code.nunique()} 檔、{len(V):,} 股票日；大波動日 {int(V.big.sum()):,}（{base:.2%}；大漲 {int(V.up.sum()):,}、大跌 {int(V.dn.sum()):,}）")

key = ["stock_code", "Date"]
M = pd.read_pickle("cnyes_matched.pkl")
N = M.merge(V[key + ["stock_name", "x", "x_on", "x_id", "x_m1", "big", "up", "dn", "post1_4", "pre5"]], on=key, how="inner")
N["ym"] = N.Date.dt.to_period("M")
N["half"] = np.where(N.Date.dt.year <= 2019, "2015-19", "2020-25")
N["first"] = N.slot != "盤中"                       # 影響日開盤前就已發布（前一天收盤後＋當天盤前）
say(f"[新聞] 對得到的 {N.id.nunique():,} 則 → {len(N):,} 個 新聞×股票日；開盤前已發布 {N['first'].mean():.0%}、盤中 {(~N['first']).mean():.0%}")

# ---------------- 1) 覆蓋率（含與重大訊息合併）----------------
cnt = N.groupby(key).agg(n_news=("id", "size"), n_first=("first", "sum")).reset_index()
V = V.merge(cnt, on=key, how="left").fillna({"n_news": 0, "n_first": 0})
after = M[M.slot == "收盤後"].assign(Date=lambda z: z.dt.dt.normalize()).groupby(key).size().rename("n_after").reset_index()
V = V.merge(after, on=key, how="left").fillna({"n_after": 0})
try:
    mp = pd.read_csv(f"mops_bigmove_matched{TAG}.csv", dtype={"stock_code": str}, parse_dates=["Date"], usecols=["stock_code", "Date"]).drop_duplicates()
    V = V.merge(mp.assign(mops=1), on=key, how="left").fillna({"mops": 0})
except FileNotFoundError:
    V["mops"] = 0
say("\n===== 1) 大波動日找得到什麼？=====")
for lab, m in [("一般日", ~V.big), ("大漲日", V.up), ("大跌日", V.dn)]:
    s = V[m]
    say(f"  {lab}（{len(s):,}）：有媒體新聞 {(s.n_news > 0).mean():.1%}（開盤前已發布 {(s.n_first > 0).mean():.1%}）｜有重大訊息 {(s.mops > 0).mean():.1%}｜兩者任一 {((s.n_news > 0) | (s.mops > 0)).mean():.1%}"
        f"｜當天收盤後才有新聞 {(s.n_after > 0).mean():.1%}")
topc = V.groupby("stock_code").n_news.sum().nlargest(50).index
s = V[V.stock_code.isin(topc)]
say(f"  （新聞最多的 50 檔）大波動日有媒體新聞或重大訊息 {((s[s.big].n_news > 0) | (s[s.big].mops > 0)).mean():.1%}；一般日 {((s[~s.big].n_news > 0) | (s[~s.big].mops > 0)).mean():.1%}")
say("\n===== 2) 有新聞的日子，大波動的機率；以及誰先動 =====")
for lab, m in [("沒有新聞", V.n_news == 0), ("開盤前已有新聞", V.n_first > 0), ("只有盤中新聞", (V.n_news > 0) & (V.n_first == 0))]:
    s = V[m]
    say(f"  {lab}（{len(s):,}）：大波動 {s.big.mean():.2%}（{s.big.mean() / base:.1f} 倍）｜大漲 {s.up.mean():.2%}、大跌 {s.dn.mean():.2%}｜超額絕對值 {s.x.abs().mean():.2%}")
s = V[V.n_after > 0]
say(f"  當天收盤後才出新聞的日子（{len(s):,}）：當天（新聞發布前）已大波動 {s.big.mean():.2%}（{s.big.mean() / base:.1f} 倍）、超額絕對值 {s.x.abs().mean():.2%} → 新聞在追報已經發生的波動")


def tm(v, ym):
    z = pd.DataFrame({"v": v, "m": ym}).dropna()
    if len(z) < 20:
        return np.nan, np.nan
    mm = z.groupby("m").v.mean()
    return z.v.mean(), (mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm))) if len(mm) > 5 else np.nan)


# ---------------- 3) 語氣（事先寫定的詞表）----------------
POS = r"創新高|新高|年增|月增|季增|成長|大增|倍增|優於預期|超乎預期|轉盈|由虧轉盈|獲利創|調升|上修|喊買|喊進|看好|看旺|樂觀|受惠|利多|報喜|接單|大單|訂單滿|滿載|漲價|調漲|供不應求|缺貨|買超|加碼|回升|回溫|復甦|強勁|亮眼|暢旺|旺到|突破|擴產|打入|通過認證|得標"
NEG = r"創新低|新低|年減|月減|季減|衰退|大減|腰斬|不如預期|低於預期|轉虧|由盈轉虧|虧損|調降|下修|降評|看淡|看壞|保守|悲觀|利空|砍單|減產|跌價|降價|供過於求|賣超|減碼|疲弱|低迷|重挫|暴跌|崩跌|下滑|下跌|裁員|停工|違約|掏空|弊案|搜索|起訴|裁罰|罰鍰|火災|爆炸|下市|跳票|遭|示警|警訊"
pos, neg = N.title.str.contains(POS, regex=True), N.title.str.contains(NEG, regex=True)
N["tone"] = np.where(pos & ~neg, "正面", np.where(neg & ~pos, "負面", "不分"))
T = N.drop_duplicates(key + ["tone", "first"])
say("\n===== 3) 標題語氣（事先寫定的詞表）→ 影響日超額報酬；分「開盤前已發布」與「盤中發布」=====")
say(f"  語氣分布：" + "、".join(f"{k} {v:.0%}" for k, v in N.tone.value_counts(normalize=True).items()))
res = {}
for fl, lab in [(True, "開盤前已發布"), (False, "盤中發布")]:
    for tone in ["正面", "負面", "不分"]:
        s = T[(T["first"] == fl) & (T.tone == tone)]
        a, t = tm(s.x, s.ym)
        on, ton = tm(s.x_on, s.ym)
        idr, tid = tm(s.x_id, s.ym)
        p, tp_ = tm(s.post1_4, s.ym)
        res[(fl, tone)] = (a, t, p, tp_, len(s))
        h1, h2 = s[s.half == "2015-19"], s[s.half == "2020-25"]
        say(f"  {lab}・{tone} n={len(s):>6,}｜當天 {a:+.2%}（t={t:+.1f}）＝隔夜 {on:+.2%}（{ton:+.1f}）＋盤中 {idr:+.2%}（{tid:+.1f}）｜2015-19 {h1.x.mean():+.2%} → 2020-25 {h2.x.mean():+.2%}"
            f"｜之後4天 {p:+.2%}（t={tp_:+.1f}）｜前一天 {s.x_m1.mean():+.2%}、前5天 {s.pre5.mean():+.2%}｜大漲 {s.up.mean():.1%}/大跌 {s.dn.mean():.1%}")
s = T[T["first"] & T.tone.isin(["正面", "負面"])]
dd = s.assign(sg=np.where(s.tone == "正面", 1, -1))
mm = dd.groupby(["ym", "tone"]).x.mean().unstack().dropna()
d1 = mm["正面"] - mm["負面"]
mm4 = dd.groupby(["ym", "tone"]).post1_4.mean().unstack().dropna()
d4 = mm4["正面"] - mm4["負面"]
tt = lambda v: v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
say(f"  開盤前已發布：正面−負面 當天 {d1.mean():+.2%}（t={tt(d1):+.1f}，{len(d1)} 個月）｜之後 4 天 {d4.mean():+.2%}（t={tt(d4):+.1f}）")
if OUTS:
    a, t, *_ = res[(True, "正面")]
    b, tb, *_ = res[(True, "負面")]
    say(f"  → N1 正面>0：{a:+.2%}，t={t:+.2f} → {'通過' if (a > 0 and t > 2.43) else '未通過'}｜N2 負面<0：{b:+.2%}，t={tb:+.2f} → {'通過' if (b < 0 and tb < -2.43) else '未通過'}"
        f"｜N3 差>0：{d1.mean():+.2%}，t={tt(d1):+.2f} → {'通過' if tt(d1) > 2.43 else '未通過'}（之後 4 天的差 {d4.mean():+.2%}）｜門檻 |t|>2.43（p<0.0167）")

# ---------------- 4) 類別（事先寫定）----------------
CATS = [
    ("月營收", r"營收"),
    ("獲利/EPS/財報", r"EPS|每股|獲利|純益|淨利|財報|毛利|虧損|盈餘|賺"),
    ("法說/展望", r"法說|展望|財測|下半年|明年|今年|Q[1-4]"),
    ("外資/法人/目標價", r"外資|投信|法人|目標價|評等|買超|賣超|大摩|小摩|高盛|里昂|瑞銀|花旗"),
    ("訂單/客戶/出貨", r"訂單|接單|出貨|客戶|供應鏈|打入|蘋果|iPhone|特斯拉|輝達|NVIDIA|AI伺服器"),
    ("漲價/缺貨/報價", r"漲價|調漲|缺貨|報價|降價|跌價|供不應求"),
    ("併購/入股/合作", r"併購|收購|入股|合併|公開收購|策略聯盟|合作|結盟|合資"),
    ("股利/庫藏股/減資/增資", r"股利|配息|配股|殖利率|庫藏股|減資|增資|私募|可轉債"),
    ("新產品/技術/擴產", r"新品|新產品|量產|擴產|新廠|技術|研發|認證|發表|推出"),
    ("熱門股/漲跌停/飆股", r"熱門股|漲停|跌停|飆|暴漲|暴跌|重挫|大漲|大跌|噴|崩"),
    ("處置/注意/違約/弊案/事故", r"處置|注意股|違約|掏空|弊案|檢調|搜索|起訴|裁罰|罰|火災|爆炸|停工|罷工|下市"),
    ("人事/股東會/經營權", r"董事長|總經理|執行長|董座|接班|改選|股東會|經營權|辭"),
]
say("\n===== 4) 新聞類別（事先寫定；一則只歸第一個符合的類別；同一股票日同類別算一次）=====")
N["cat"] = "其他"
for n_, pat in reversed(CATS):
    N.loc[N.title.str.contains(pat, regex=True), "cat"] = n_
C = N.drop_duplicates(key + ["cat"])
say(f"  {'類別':<16s}{'則數':>7s}｜大波動機率（漲/跌）｜影響日超額(t)｜2015-19 → 2020-25｜只看開盤前已發布(t)｜之後4天(t)｜前一天")
tab = []
for c in [n_ for n_, _ in CATS] + ["其他"]:
    s = C[C["cat"] == c]
    if len(s) < 30:
        continue
    a, t = tm(s.x, s.ym)
    f = s[s["first"]]
    af, tf = tm(f.x, f.ym)
    p, tp_ = tm(s.post1_4, s.ym)
    tab.append((c, len(s), s.big.mean(), s.up.mean(), s.dn.mean(), a, t, s[s.half == "2015-19"].x.mean(), s[s.half == "2020-25"].x.mean(), af, tf, p, tp_, s.x_m1.mean()))
for r in sorted(tab, key=lambda z: -z[2]):
    say(f"  {r[0]:<16s}{r[1]:>7,}｜{r[2]:.1%}（{r[3]:.1%}/{r[4]:.1%}）｜{r[5]:+.2%}（{r[6]:+.1f}）｜{r[7]:+.2%} → {r[8]:+.2%}｜{r[9]:+.2%}（{r[10]:+.1f}）｜{r[11]:+.2%}（{r[12]:+.1f}）｜{r[13]:+.2%}")

# ---------------- 5) 關鍵字：2015-19 找 → 2020-25 驗（只在探索模式做）----------------
if not OUTS:
    names = [n for n in df.stock_name.dropna().astype(str).unique() if len(n) >= 2]
    name_re = re.compile("|".join(sorted((re.escape(n) for n in names), key=len, reverse=True)))

    def grams(title):
        out = set()
        for seg in re.split(r"[^一-鿿]+", name_re.sub("", title)):
            for n in (2, 3, 4):
                for k in range(len(seg) - n + 1):
                    out.add(seg[k:k + n])
        return out

    K = N.drop_duplicates(key + ["id"]).copy()
    K["g"] = K.title.map(grams)
    A, B = K[K.half == "2015-19"], K[K.half == "2020-25"]
    found = {}
    for lab, col in [("大漲日", "up"), ("大跌日", "dn")]:
        a, b = A[A[col]].g.tolist(), A[~A.big].g.tolist()
        ca, cb = Counter(w for s_ in a for w in s_), Counter(w for s_ in b for w in s_)
        rows_ = []
        for w, c in ca.items():
            if c >= 12:
                pa, pb = (c + 0.5) / (len(a) + 1), (cb.get(w, 0) + 0.5) / (len(b) + 1)
                if pa / pb >= 2:
                    rows_.append((w, c, pa / pb, (np.log(pa / (1 - pa)) - np.log(pb / (1 - pb))) / np.sqrt(1 / (c + 0.5) + 1 / (cb.get(w, 0) + 0.5))))
        rows_.sort(key=lambda r: -r[3])
        keep = []
        for r in rows_:
            if not any((r[0] in k[0]) or (k[0] in r[0]) for k in keep):
                keep.append(r)
            if len(keep) >= 20:
                break
        found[col] = [k[0] for k in keep]
        say(f"\n===== 5) {lab}：2015~2019 找到的詞（{len(a):,} 篇 vs 一般日 {len(b):,} 篇）→ 2020~2025 驗證 =====")
        for w, c, mult, z in keep:
            b_ = B[B.g.map(lambda s_: w in s_)]
            a_ = A[A.g.map(lambda s_: w in s_)]
            if len(b_) < 15:
                continue
            say(f"  「{w}」2015-19：{len(a_):,} 則、{mult:.1f} 倍、當天 {a_.x.mean():+.2%}｜2020-25：{len(b_):,} 則、當天 {b_.x.mean():+.2%}（t={tm(b_.x, b_.ym)[1]:+.1f}）、大漲 {b_.up.mean():.1%}/大跌 {b_.dn.mean():.1%}｜前一天 {b_.x_m1.mean():+.2%}")
    pd.Series(found).to_json("cnyes_keywords_found.json", force_ascii=False)
    say("\n===== 6) 例子：2020 年以後波動最大、重大訊息沒有、但有媒體新聞的 20 個股票日 =====")
    ex = N[N.big & (N.Date >= "2020-01-01")].merge(V[key + ["mops"]], on=key)
    ex = ex[ex.mops == 0].sort_values("x", key=lambda s_: s_.abs(), ascending=False).drop_duplicates(key).head(20)
    for _, r in ex.iterrows():
        say(f"  {r.Date.date()} {r.stock_code} {r.stock_name} 超額 {r.x:+.1%}｜[{r.slot} {str(r['dt'])[5:16]}] {r.title[:44]}")
N.drop(columns=["kw"]).to_csv(f"cnyes_bigmove_matched{TAG}.csv", index=False, encoding="utf-8-sig")
open(f"cnyes_bigmove{TAG}.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
