# -*- coding: utf-8 -*-
"""
庫藏股公告後續漲：實際進出場回測（扣成本、排除買不到的）
事件：tej_raw/tej_events.csv 的「庫藏股」（公告日）；股價：2001~2013 用 tej_raw/tej_price_daily_pre2014.pkl，2015~2025 用 final_dataset_v3_tejall25。
第 0 天＝公告日（非交易日順延，最多 7 天）。公告日沒有時間，所以最早只允許在「第 +1 天收盤」進場（保守：假設都是第 0 天收盤後才公告）。

事先寫定的規則（2026-10-09，跑之前寫下；不做參數搜尋，只報這兩條）：
  規則 A（主要）：第 +1 天收盤買、第 +5 天收盤賣（持有 4 個交易日）。第 +1 天收在漲停 → 視為買不到，剔除。
  規則 B（保守）：第 +2 天開盤買、第 +5 天收盤賣。第 +2 天開盤價比前一天收盤高 ≥ 漲停幅度−0.5% → 視為買不到，剔除。
  成本：來回 0.47%（手續費六折）與 0.585%（無折扣）；另報毛報酬。
  報：每筆平均原始報酬、超額報酬（減同期全市場等權）、勝率、扣成本後；
      日曆時間投資組合（每天等權持有所有在場部位，沒部位的日子報酬 0）：年化報酬、年化波動、Sharpe、最大回落，與同期全市場等權比較；分年。
  主要判準：規則 A、兩段期間（2001~2013、2015~2025）扣 0.47% 後每筆平均原始報酬與超額報酬是否都 > 0（t 依公告月群聚）。
python buyback_backtest.py
"""
import numpy as np
import pandas as pd

LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def load_pre():
    p = pd.read_pickle("tej_raw/tej_price_daily_pre2014.pkl")
    p = p[p.date <= "2013-12-31"]
    d = pd.DataFrame({"code": p.code, "date": p.date, "open": p["開盤價(元)"], "close": p["收盤價(元)"], "ret": p["報酬率％"] / 100, "lim": p["漲跌停"]})
    return d


def load_post():
    p = pd.read_csv("final_dataset_v3_tejall25/all_stocks_v3.csv", dtype={"stock_code": str, "limit_flag": str}, parse_dates=["Date"],
                    usecols=["stock_code", "Date", "Open", "Close", "Volume", "limit_flag"])
    p = p.sort_values(["stock_code", "Date"])
    d = pd.DataFrame({"code": p.stock_code, "date": p.Date, "open": p.Open.where(p.Volume > 0), "close": p.Close, "lim": p.limit_flag})
    d["ret"] = d.close / d.groupby("code").close.shift(1) - 1
    return d


ev = pd.read_csv("tej_raw/tej_events.csv", dtype={"stock_code": str}, parse_dates=["ann_date"])
ev = ev[(ev.kind == "庫藏股") & ev.ann_date.notna() & ev.stock_code.str.match(r"^[1-9]\d{3}$", na=False)]


def tstat(v, ym):
    d = pd.DataFrame({"v": v, "m": ym}).dropna()
    mm = d.groupby("m").v.mean()
    return mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm)))


def run(d, label):
    d = d.sort_values(["code", "date"]).reset_index(drop=True)
    d["mkt"] = d.groupby("date").ret.transform("mean")
    d["pos"] = d.groupby("code").cumcount()
    g = d.groupby("code")
    d["gap"] = d.open / g.close.shift(1) - 1
    limit = np.where(d.date < "2015-06-01", 0.07, 0.10)
    d["open_locked"] = d.gap >= limit - 0.005
    E = ev[(ev.ann_date >= d.date.min()) & (ev.ann_date <= d.date.max())].sort_values("ann_date")
    m = pd.merge_asof(E, d[["code", "date", "pos"]].rename(columns={"code": "stock_code"}).sort_values("date"),
                      left_on="ann_date", right_on="date", by="stock_code", direction="forward")
    m = m[m.pos.notna() & ((m.date - m.ann_date).dt.days <= 7)].drop_duplicates(["stock_code", "date"]).copy()
    m["pos"] = m.pos.astype(int)
    X = d.set_index(["code", "pos"])

    def at(col, k):
        return X[col].reindex(pd.MultiIndex.from_arrays([m.stock_code, m.pos + k])).values

    c1, c5, o2 = at("close", 1), at("close", 5), at("open", 2)
    mk = np.column_stack([at("mkt", k) for k in (2, 3, 4, 5)])
    mcum = np.prod(1 + mk, axis=1) - 1
    m["ym"] = m.date.dt.to_period("M")
    m["year"] = m.date.dt.year
    m["A_ok"] = (at("lim", 1) != "+") & np.isfinite(c1) & np.isfinite(c5)
    m["A_raw"] = np.where(m.A_ok, c5 / c1 - 1, np.nan)
    m["A_x"] = m.A_raw - mcum
    m["B_ok"] = (~pd.Series(at("open_locked", 2)).fillna(True).astype(bool).values) & np.isfinite(o2) & np.isfinite(c5)
    r2 = at("ret", 2)
    m["B_raw"] = np.where(m.B_ok, c5 / o2 - 1, np.nan)
    m["B_x"] = m.B_raw - (mcum - 0)          # 市場用第 2~5 天（含第 2 天隔夜，略高估市場）
    say(f"\n===== {label}：庫藏股公告 {len(m):,} 件（{m.date.min().date()}~{m.date.max().date()}）=====")
    say(f"  第 +1 天收漲停（規則 A 買不到）{(at('lim', 1) == '+').mean():.1%}；第 +2 天開盤即漲停（規則 B 買不到）{pd.Series(at('open_locked', 2)).fillna(False).astype(bool).mean():.1%}")
    res = {}
    for R, nm in [("A", "規則 A：第+1天收盤買→第+5天收盤賣"), ("B", "規則 B：第+2天開盤買→第+5天收盤賣")]:
        s = m[m[f"{R}_ok"]]
        raw, x = s[f"{R}_raw"], s[f"{R}_x"]
        say(f"  {nm}｜可交易 {len(s):,} 筆")
        say(f"    毛報酬：原始 {raw.mean():+.2%}（t={tstat(raw, s.ym):+.2f}）、超額 {x.mean():+.2%}（t={tstat(x, s.ym):+.2f}）｜勝率（原始>0）{(raw > 0).mean():.1%}｜中位數 {raw.median():+.2%}")
        for c in (0.0047, 0.00585):
            say(f"    扣 {c:.3%}：原始 {(raw - c).mean():+.2%}（t={tstat(raw - c, s.ym):+.2f}）、超額 {(x - c).mean():+.2%}（t={tstat(x - c, s.ym):+.2f}）｜勝率 {(raw - c > 0).mean():.1%}")
        yr = s.groupby("year").agg(n=(f"{R}_raw", "size"), raw=(f"{R}_raw", "mean"), x=(f"{R}_x", "mean"))
        say("    分年 超額(毛)：" + " ".join(f"{y}:{r.x:+.1%}({int(r.n)})" for y, r in yr.iterrows()))
        say(f"    超額(毛)為正的年數 {(yr.x > 0).sum()}/{len(yr)}；扣 0.47% 後為正 {((yr.x - 0.0047) > 0).sum()}/{len(yr)}")
        res[R] = s
    # 日曆時間投資組合（規則 A）：持有第 +2~+5 天的日報酬
    s = res["A"]
    rows = []
    for k in (2, 3, 4, 5):
        rows.append(pd.DataFrame({"date": at("date", k)[m.A_ok.values], "r": at("ret", k)[m.A_ok.values], "first": k == 2}))
    H = pd.concat(rows).dropna(subset=["date", "r"])
    days = pd.Series(sorted(d.date.unique()))
    port = H.groupby("date").r.mean().reindex(days.values).fillna(0.0)
    npos = H.groupby("date").size().reindex(days.values).fillna(0)
    # 成本：每筆來回成本攤在進場日（等權下，當天新進部位占比 × 成本）
    newshare = (H[H["first"]].groupby("date").size() / H.groupby("date").size()).reindex(days.values).fillna(0.0)
    mkt = d.groupby("date").mkt.first().reindex(days.values)

    def stats_(r):
        ann = (1 + r).prod() ** (252 / len(r)) - 1
        vol = r.std() * np.sqrt(252)
        eq = (1 + r).cumprod()
        return ann, vol, (r.mean() * 252) / vol if vol > 0 else np.nan, (eq / eq.cummax() - 1).min()

    say(f"  日曆時間投資組合（規則 A，每天等權持有在場部位；有部位的日子占 {(npos > 0).mean():.0%}，在場部位中位數 {int(npos[npos > 0].median())} 檔）")
    for nm, r in [("毛報酬", port), ("扣 0.47%", port - newshare * 0.0047), ("扣 0.585%", port - newshare * 0.00585), ("同期全市場等權（買進持有）", mkt.fillna(0))]:
        a, v, sh, dd = stats_(r)
        say(f"    {nm:<14s} 年化 {a:+.1%}｜年化波動 {v:.1%}｜Sharpe {sh:.2f}｜最大回落 {dd:.1%}")
    ex = (port - newshare * 0.0047) - mkt.fillna(0) * (npos > 0).values
    say(f"    扣 0.47% 後、有部位的日子相對全市場的超額：日平均 {ex[npos.values > 0].mean() * 1e4:+.1f} bp（t={ex[npos.values > 0].mean() / (ex[npos.values > 0].std() / np.sqrt((npos > 0).sum())):+.2f}）")
    return m


a = run(load_pre(), "2001~2013（驗證期）")
b = run(load_post(), "2015~2025（探索期）")
pd.concat([a.assign(sample="2001-2013"), b.assign(sample="2015-2025")]).to_csv("buyback_backtest_trades.csv", index=False, encoding="utf-8-sig")
open("buyback_backtest.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
