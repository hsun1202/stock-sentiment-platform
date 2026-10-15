# -*- coding: utf-8 -*-
"""
鉅亨網台股新聞 → 清理 → 對到個股與交易日（給 cnyes_bigmove.py 用）
輸入：cnyes_news/*.jsonl（crawl_cnyes_news.py，188,441 則，2015~2025）、final_dataset_v3_tejall25（股名、交易日曆）
清理：
  1. 去掉轉貼的公司公告（標題是「公司名：公告／代子公司／澄清…」且沒有來源）——那是重大訊息的重複，已在 mops_bigmove.py 分析過。
  2. 去掉券商/期貨商的制式盤後評論（來源或標題含投顧、期貨、盤後分析）與自動報價（《股市》…期指）。
對到個股：
  a. 鉅亨自己標的 stock 欄（2017 年起約 85% 有標）。
  b. 標題裡的「股名(代號)」。
  c. 標題含股名：先用「有標 stock 的新聞」量每個股名的精確率（標題含該名字時，stock 欄真的有這檔的比例），
     只採用精確率 ≥ 0.8 且出現 ≥ 5 次的股名；沒有足夠證據的，只採用 3 個字以上的股名。（避免「傳奇投資人」→ 4994 傳奇這類誤配）
  一則新聞對到超過 3 檔的（盤勢整理、族群新聞）不用。
交易日：開盤前（<09:00）與盤中（09:00~13:30）的新聞，影響日＝當天；收盤後的，影響日＝下一個交易日。slot 記錄發布時段。
輸出：cnyes_matched.pkl（id, dt, title, kw, stock_code, slot, Date, how）、cnyes_build.log
python cnyes_build.py
"""
import glob
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


rows = []
for f in glob.glob("cnyes_news/*.jsonl"):
    rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
d = pd.DataFrame(rows).drop_duplicates("id")
d["dt"] = pd.to_datetime(d.t, unit="s") + pd.Timedelta(hours=8)
say(f"[原始] {len(d):,} 則（{d.dt.min().date()}~{d.dt.max().date()}）")

px = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"], usecols=["stock_code", "stock_name", "Date"])
names = px.drop_duplicates("stock_code", keep="last").set_index("stock_code").stock_name.astype(str).str.strip()
names = names[names.str.len() >= 2]
name2code = defaultdict(set)
for c, n in names.items():
    name2code[n].add(c)
    if n.endswith("-KY"):
        name2code[n[:-3]].add(c)
valid = set(names.index)
allnames = sorted(name2code, key=len, reverse=True)
name_re = re.compile("|".join(re.escape(n) for n in allnames))

pre = d.title.str.extract(r"^([^\s：:]{2,12})[：:]")[0]
is_ann = pre.isin(set(name2code)) & (d.src == "") | d.title.str.contains(r"^[^\s：:]{2,12}[：:]\s*(?:公告|代子公司|代重要子公司|本公司|澄清|更正|補充)", regex=True)
is_brk = d.src.str.contains("投顧|期貨|證券|投信") | d.title.str.contains(r"投顧|盤後分析|期貨.{0,8}盤後|選擇權盤後|^《股市》|^《期貨》", regex=True)
say(f"  去掉轉貼公告 {int(is_ann.sum()):,}、券商制式評論與自動報價 {int((is_brk & ~is_ann).sum()):,}")
d = d[~is_ann & ~is_brk].copy()
say(f"[清理後] {len(d):,} 則；有鉅亨 stock 標記的 {(d.stock.str.len() > 0).mean():.0%}")

# 股名精確率（用有標記的新聞）
tag = d[d.stock.str.len() > 0]
hit, tot = Counter(), Counter()
for title, st in zip(tag.title, tag.stock):
    st = set(map(str, st))
    for n in set(name_re.findall(title)):
        tot[n] += 1
        if name2code[n] & st:
            hit[n] += 1
prec = {n: hit[n] / tot[n] for n in tot}
good = {n for n in name2code if (tot[n] >= 5 and prec[n] >= 0.8) or (tot[n] < 5 and len(n) >= 3)}
bad = sorted([(n, tot[n], round(prec[n], 2)) for n in tot if tot[n] >= 20 and prec[n] < 0.8], key=lambda x: -x[1])
say(f"  股名 {len(name2code):,} 個：可用於標題比對 {len(good):,} 個；精確率不足而排除的（出現≥20 次）{len(bad)} 個，例：" + "、".join(f"{n}({p})" for n, _, p in bad[:25]))
good_re = re.compile("|".join(re.escape(n) for n in sorted(good, key=len, reverse=True)))
code_re = re.compile(r"[\(（]([1-9]\d{3})(?:-TW)?[\)）]")

out = []
for nid, dt_, title, kw, st in zip(d.id, d.dt, d.title, d.kw, d.stock):
    codes, how = set(), ""
    s1 = {str(x) for x in st if str(x) in valid}
    if s1:
        codes, how = s1, "tag"
    s2 = {c for c in code_re.findall(title) if c in valid}
    if s2:
        codes |= s2
        how = how or "code"
    if not codes:
        for n in set(good_re.findall(title)):
            codes |= name2code[n]
        how = "name" if codes else ""
    if codes and len(codes) <= 3:
        for c in codes:
            out.append((nid, dt_, title, kw, c, how))
M = pd.DataFrame(out, columns=["id", "dt", "title", "kw", "stock_code", "how"])
cal = np.array(sorted(px.Date.unique()), dtype="datetime64[ns]")
d0 = M.dt.dt.normalize().values.astype("datetime64[ns]")
i = np.searchsorted(cal, d0)
is_td = (i < len(cal)) & (cal[np.minimum(i, len(cal) - 1)] == d0)
hm = (M.dt.dt.hour * 60 + M.dt.dt.minute).values
slot = np.where(is_td & (hm < 540), "盤前", np.where(is_td & (hm < 810), "盤中", "收盤後"))
imp = np.where((slot == "收盤後") & is_td, i + 1, i)
M["slot"] = slot
M = M[imp < len(cal)].copy()
M["Date"] = cal[imp[imp < len(cal)]]
M.to_pickle("cnyes_matched.pkl")
say(f"[對到個股] {M.id.nunique():,} 則新聞 → {len(M):,} 個 新聞×股票、{M.stock_code.nunique():,} 檔；來源：" + "、".join(f"{k} {v:,}" for k, v in M.how.value_counts().items()))
say("  每年則數：" + " ".join(f"{y}:{n:,}" for y, n in M.groupby(M.dt.dt.year).id.nunique().items()))
say("  發布時段：" + "、".join(f"{k} {v:.0%}" for k, v in M.slot.value_counts(normalize=True).items()))
top = M.stock_code.value_counts().head(10)
say("  新聞最多的股票：" + "、".join(f"{names.get(c, c)} {n:,}" for c, n in top.items()) + f"；每檔中位數 {int(M.stock_code.value_counts().median())} 則")
open("cnyes_build.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
