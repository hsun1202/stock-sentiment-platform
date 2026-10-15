# -*- coding: utf-8 -*-
"""
CMoney 同學會資料清理規則（2026-10-04；依 Claude 逐則閱讀 1,764 篇後歸納）
  bot    ：自動貼文（券商報告、盤後複習/盤中訊號、分點買賣超、三大法人/融資券數字、每日精選新聞、各式排行）
  promo  ：廣告/導流（加賴、拉群組、求追蹤按讚、推 App、直播/社團、短網址導流）
  link   ：只有網址（去掉網址後不到 8 個字）
  news   ：整段貼新聞（有記者/報導/通訊社字樣且超過 100 字）——不是廣告，但不是發文者自己的看法
  dup    ：同一段內容貼到 ≥3 檔股票的討論區，或同一人重複貼 ≥3 次
  account：某帳號 ≥20 篇且上述比例 ≥50% → 該帳號其餘文章也一併移除（機器人/行銷帳號）
flag(df) 回傳每列的類別（'' 表示保留）。df 需有 stock_code, creator, content 欄。
"""
import re

import pandas as pd

URL_RE = re.compile(r"https?://\S+")
BOT_RE = re.compile(
    r"剛剛發布了一份|盤後複習|盤中即時|訊號來源|券商分點買賣超排名|今天關鍵分點|以下是今天的精選新聞|【數報】|熱門排名NO|"
    r"三大法人.{0,3}(買超|賣超).{0,4}[+\-－]?\d|三大法人買賣超|融資融券餘額|資券增減|買賣超前\d+|外資.{0,3}[+\-]\d+.{0,6}投信|"
    r"大數據總結|量化總結|【強勢股】|波羅的海航運各型指數|航海指數|籌碼較上周集中|贏勢股 ?APP|跟著法人走|小羽報報|警示股|"
    r"成交值排行|發言日期.{0,20}發言時間|倫[銅鎳鋁]CFD|【.{0,6}收市】|盤前快報|盤中熱門焦點股")
PROMO_RE = re.compile(
    r"加我[賴癩]|[賴癩] ?ID|LINE ?ID|加入群組|歡迎.{0,6}追蹤|追蹤我|請.{0,3}按讚|按讚\+追蹤|五星好評|加討論組|直播時間|社團內|"
    r"下載.{0,14}(APP|App|app|K線)|用《.{0,8}APP》|免責聲明|絕對主力|咪咪.{0,6}攻略|粉絲.{0,6}(見證|們)|VIP ?權限|限時優惠|"
    r"教學組|上千人的社員|搶先看分析|有緣人|歡迎大家一起來討論切磋|想找好標的")
LINK_RE = re.compile(r"page\.link|cmy\.tw|onelink\.me|cmoney\.tw/(r|notes)/|pixnet\.net|reurl\.cc|ssur\.cc|youtu\.be|youtube\.com")
NEWS_RE = re.compile(r"記者.{1,12}報導|／台北報導|台北電】|中央社|經濟日報|工商時報|財訊快報|時報新聞|MoneyDJ|FX168|鉅亨網|即時報導|〔記者|【時報")


def rule(text):
    t = text or ""
    if BOT_RE.search(t):
        return "bot"
    if PROMO_RE.search(t):
        return "promo"
    if len(URL_RE.sub("", t).strip()) < 8 and URL_RE.search(t):
        return "link"
    if LINK_RE.search(t):
        return "promo"
    if NEWS_RE.search(t) and len(t) >= 100:
        return "news"
    return ""


def flag(df):
    f = df.content.map(rule)
    nstk = df.groupby("content").stock_code.transform("nunique")
    nsame = df.groupby(["creator", "content"]).content.transform("size")
    long_enough = df.content.str.len() >= 12                 # 很短的句子（如「加油」）本來就會重複，不算
    f = f.mask((f == "") & long_enough & ((nstk >= 3) | (nsame >= 3)), "dup")
    g = pd.DataFrame({"c": df.creator, "bad": f != ""}).groupby("c").bad.agg(["size", "mean"])
    bad_acc = set(g[(g["size"] >= 20) & (g["mean"] >= 0.5)].index)
    f = f.mask((f == "") & df.creator.isin(bad_acc), "account")
    return f
