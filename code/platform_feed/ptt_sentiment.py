# -*- coding: utf-8 -*-
"""
PTT 每日情緒：分類（train_ptt_bert.py 訓練的 BERT）→ 依交易日彙整 → 市場與個股情緒。
  交易日：留言時間落在（前一交易日 13:30, 當天 13:30]，同研究（build_dataset_v3.py）。
  市場情緒：標題含「閒聊」的文章（盤中／盤後閒聊）。
  個股：標題比對到股票（四碼代號或股票名稱）的文章；一篇提到多檔就每檔都算。
  淨情緒＝(正 − 負) ÷ (正 + 負)。
研究結論（PTT）：情緒能描述市場，但不能預測隔天或之後的報酬 → 平台只當「描述」，不當買賣訊號。
輸出 out/sentiment_market.json、out/sentiment_stocks.json；分類結果快取在 data/ptt/labels.csv。
cd D:/專題claude && python platform_feed/ptt_sentiment.py
"""
import datetime as dt
import json
import os
import re

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(BASE, "data", "ptt")
LAB = os.path.join(DIR, "labels.csv")
MODEL = os.path.join(BASE, "models", "ptt_bert_context")
OUT = os.path.join(BASE, "out")
L = ["positive", "negative", "neutral"]
STOP = {"統一", "中華", "大同", "國產", "東元", "台開", "力麗", "大成", "國泰", "富邦", "永大", "全國", "中興", "亞洲"}   # 常見詞，只認代號
NOTE = "研究結論：PTT 情緒能描述市場氣氛，但不能預測之後的報酬（2015–2025 多種方法檢驗）。這裡只當參考，不是買賣訊號。"


def stock_names():
    px = pd.read_csv(os.path.join(BASE, "data", "prices.csv"), dtype={"code": str}, usecols=["code", "name", "date"])
    px = px[px.code.str.match(r"^[1-9]\d{3}$")]
    return px.drop_duplicates("code", keep="last").set_index("code").name, sorted(pd.to_datetime(px.date.unique()))


def match_titles(titles: pd.Series, names: pd.Series) -> dict:
    """{title: [code, ...]}：四碼代號，或股票名稱（去掉 -KY、*；兩字常見詞只認代號）"""
    codes = set(names.index)
    pats = [(c, re.sub(r"[-*].*$|-KY$", "", n).strip()) for c, n in names.items()]
    pats = [(c, n) for c, n in pats if len(n) >= 2 and n not in STOP]
    out = {}
    for t in titles.unique():
        hit = [c for c in re.findall(r"(?<!\d)(\d{4})(?!\d)", t) if c in codes]
        byname = [(c, n) for c, n in pats if n in t and c not in hit]
        hit += [c for c, n in byname if not any(n != n2 and n in n2 for _, n2 in byname)]   # 「聯發」被「聯發科」包含 → 只留長的
        if hit:
            out[t] = hit
    return out


def trading_day(times: pd.Series, days: list) -> pd.Series:
    """留言時間 → 歸屬交易日。過去沒有行情的日子＝休市（例：颱風假）；今天以後還沒有行情的用平日估。"""
    today = pd.Timestamp.today().normalize()
    d = list(days) + [x for x in pd.bdate_range(max(days[-1] + pd.Timedelta(days=1), today), periods=10)]
    cut = pd.DatetimeIndex([x + pd.Timedelta(hours=13, minutes=30) for x in d])
    i = cut.searchsorted(times.values, side="left")
    return pd.Series([d[k] if k < len(d) else pd.NaT for k in i], index=times.index)


def classify(df: pd.DataFrame, ctx: pd.Series, log=print) -> pd.Series:
    """只分類還沒分過的留言；結果快取在 labels.csv（aid, idx, label）"""
    cache = pd.read_csv(LAB, dtype=str) if os.path.exists(LAB) else pd.DataFrame(columns=["aid", "idx", "label"])
    key = df.aid + "#" + df.idx.astype(str)
    have = dict(zip(cache.aid + "#" + cache.idx.astype(str), cache.label))
    todo = df[~key.isin(have)]
    if len(todo):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
        a, b = ctx.loc[todo.index].tolist(), todo.content.fillna("").astype(str).tolist()
        order = np.argsort([len(x) for x in b])
        pred = np.empty(len(b), dtype=object)
        with torch.no_grad():
            for s in range(0, len(b), 256):
                idx = order[s:s + 256]
                enc = tok([a[i] for i in idx], [b[i] for i in idx], truncation="only_first", max_length=128, padding=True, return_tensors="pt").to(dev)
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    p = model(**enc).logits.float().argmax(-1).cpu().numpy()
                pred[idx] = [L[k] for k in p]
        new = pd.DataFrame({"aid": todo.aid.values, "idx": todo.idx.astype(str).values, "label": pred})
        pd.concat([cache, new]).to_csv(LAB, index=False, encoding="utf-8-sig")
        have.update(dict(zip(new.aid + "#" + new.idx, new.label)))
        log(f"[情緒分類] 新分類 {len(todo):,} 則（{dev}）")
    return key.map(have)


def summarize(g: pd.DataFrame) -> pd.DataFrame:
    s = g.groupby("day").agg(n=("label", "size"), pos=("label", lambda x: (x == "positive").sum()), neg=("label", lambda x: (x == "negative").sum()))
    s["net"] = (s.pos - s.neg) / (s.pos + s.neg).where(s.pos + s.neg > 0)
    return s


def build(log=print):
    df = pd.read_csv(os.path.join(DIR, "comments.csv"), dtype=str, keep_default_na=False)
    names, days = stock_names()
    df["time"] = pd.to_datetime(df.time, errors="coerce")
    pt = pd.to_datetime(df.post_time, errors="coerce")
    bad = (df.time < pt - pd.Timedelta(minutes=1)) | (df.time > pd.Timestamp.now() + pd.Timedelta(hours=1))   # 年份推錯、時間異常（同研究：早於發文就排除）
    df = df[~bad].dropna(subset=["time"])
    df = df[df.content.str.strip() != ""]
    m = match_titles(df.title, names)
    df["chat"] = df.title.str.contains("閒聊")
    first = df.title.map(lambda t: names.get(m[t][0], "") if t in m else "")
    ctx = np.where(df.chat, "大盤", first) + "｜" + df.title + "｜" + df.tag
    df["label"] = classify(df, pd.Series(ctx, index=df.index), log).values
    df["day"] = trading_day(df.time, days)
    asof = days[-1]

    mk = summarize(df[df.chat])
    mk = mk[mk.index <= asof + pd.Timedelta(days=7)]
    mk["net_avg20"] = mk.net.shift(1).rolling(20, min_periods=5).mean()
    mk["n_avg20"] = mk.n.shift(1).rolling(20, min_periods=5).mean()
    mk["net_z"] = (mk.net - mk.net_avg20) / mk.net.shift(1).rolling(20, min_periods=5).std()

    rows = []
    for t, cs in m.items():
        for c in cs:
            rows.append((t, c))
    tm = pd.DataFrame(rows, columns=["title", "code"])
    st = df[~df.chat].merge(tm, on="title")
    per = st.groupby(["code", "day"]).agg(n=("label", "size"), pos=("label", lambda x: (x == "positive").sum()),
                                           neg=("label", lambda x: (x == "negative").sum()), arts=("aid", "nunique")).reset_index()
    per["net"] = (per.pos - per.neg) / (per.pos + per.neg).where(per.pos + per.neg > 0)
    per = per.sort_values(["code", "day"])
    per["n_avg20"] = per.groupby("code").n.transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())

    os.makedirs(OUT, exist_ok=True)
    f = lambda x: None if pd.isna(x) else round(float(x), 4)  # noqa: E731
    market = {"asof": f"{asof:%Y-%m-%d}", "generated_at": dt.datetime.now().isoformat(timespec="seconds"), "source": "PTT Stock 板 盤中／盤後閒聊留言",
              "days": [{"date": f"{d:%Y-%m-%d}", "n": int(r.n), "pos": int(r.pos), "neg": int(r.neg), "net": f(r.net), "net_avg20": f(r.net_avg20),
                        "n_avg20": f(r.n_avg20), "net_z": f(r.net_z)} for d, r in mk.iterrows()], "note": NOTE}
    json.dump(market, open(os.path.join(OUT, "sentiment_market.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    last = per.day.max() if len(per) else asof
    top = per[per.day == last].sort_values("n", ascending=False).head(50)
    stocks = {"asof": f"{asof:%Y-%m-%d}", "latest_day": f"{last:%Y-%m-%d}", "generated_at": market["generated_at"],
              "top": [{"code": r.code, "name": names.get(r.code, ""), "n": int(r.n), "arts": int(r.arts), "net": f(r.net),
                       "attention": f(r.n / r.n_avg20) if r.n_avg20 else None} for r in top.itertuples()],
              "series": {c: [{"date": f"{r.day:%Y-%m-%d}", "n": int(r.n), "net": f(r.net)} for r in g.itertuples()] for c, g in per.groupby("code")},
              "note": NOTE}
    json.dump(stocks, open(os.path.join(OUT, "sentiment_stocks.json"), "w", encoding="utf-8"), ensure_ascii=False)
    lt = mk.iloc[-1] if len(mk) else None
    log(f"[PTT 情緒] 留言 {len(df):,} 則（閒聊 {int(df.chat.sum()):,}、個股文章 {len(st):,}）；交易日 {len(mk)} 天；"
        + (f"最新 {mk.index[-1]:%m-%d} 市場淨情緒 {lt.net:+.3f}（{int(lt.n):,} 則）" if lt is not None else "")
        + f"；個股 {per.code.nunique()} 檔")
    return mk, per


if __name__ == "__main__":
    build()
