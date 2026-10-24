"""今日名單：營收驚喜、盈餘驚喜、壞消息避開（platform_feed 每天 18:50 產生）。"""
import pandas as pd
import streamlit as st

import common

rev, ern = common.signals("revenue_signals"), common.signals("earnings_signals")
if not rev or not ern:
    common.empty("還沒有名單：先執行 python platform_feed/run_feed.py")
    st.stop()

common.page_title("今日名單", f"資料日 {rev['asof']}・財報公布後第一個交易日開盤進場，月營收持有 20 個交易日、季報持有 60 個交易日")
common.kpis([
    ("營收驚喜（持有中）", f"{len(rev['surprise_active'])} 檔", f"本月門檻 SUR ≥ {rev['thresholds']['p90']}", "accent"),
    ("盈餘驚喜（持有中）", f"{len(ern['surprise_active'])} 檔", f"本月門檻 SUE ≥ {ern['thresholds']['p90']}", "accent"),
    ("營收壞消息（避開）", f"{len(rev['bad_news_active'])} 檔", f"SUR ≤ {rev['thresholds']['p10']}", "calm"),
    ("盈餘壞消息（避開）", f"{len(ern['bad_news_active'])} 檔", f"SUE ≤ {ern['thresholds']['p10']}", "calm"),
    ("回測：贏一般股票", "+1.26%／月", "2018–2025 兩個策略皆是（t 3.8～4.8）"),
    ("對 0050 的 β", "0.56～0.65", "α +16～18%／年（顯著）"),
])


def table(rows: list[dict], kind: str, key: str):
    if not rows:
        common.empty("目前沒有")
        return
    ret = common.since_entry(rows)
    asof = pd.Timestamp(rev["asof"])
    df = pd.DataFrame({
        "代號": [r["code"] for r in rows], "名稱": [r["name"] for r in rows], "期別": [r["period"] for r in rows],
        "公布日": [r["announce_date"] for r in rows], "進場日": [r["entry_date"] for r in rows], "出場日": [r["exit_date"] for r in rows],
        "驚喜分數": [r["surprise"] for r in rows]})
    if kind == "rev":
        df["營收年增"] = [r.get("yoy") for r in rows]
        df["當月營收（億）"] = [r["revenue_k"] / 1e5 for r in rows]
    else:
        df["單季淨利（億）"] = [r["net_income_k"] / 1e5 for r in rows]
        df["去年同季（億）"] = [None if r.get("net_income_last_year_k") is None else r["net_income_last_year_k"] / 1e5 for r in rows]
    df["進場以來"] = ret
    df["剩餘交易日"] = [max(0, len(pd.bdate_range(asof, r["exit_date"])) - 1) for r in rows]
    fmt = {"驚喜分數": "{:+.2f}", "營收年增": "{:+.1%}", "當月營收（億）": "{:,.2f}", "單季淨利（億）": "{:,.2f}",
           "去年同季（億）": "{:,.2f}", "進場以來": "{:+.1%}"}
    sty = df.style.format({k: v for k, v in fmt.items() if k in df.columns}, na_rep="—").map(
        common.color_signed, subset=[c for c in ["營收年增", "進場以來", "驚喜分數"] if c in df.columns])
    ev = st.dataframe(sty, hide_index=True, on_select="rerun", selection_mode="single-row", key=key, height=min(38 * len(df) + 40, 560))
    st.caption("點一列看個股。進場以來＝進場日開盤到最新收盤（未含息）。")
    if ev.selection.rows:
        common.goto_stock(df.iloc[ev.selection.rows[0]]["代號"])


t1, t2, t3 = st.tabs([f"營收驚喜 {len(rev['surprise_active'])}", f"盈餘驚喜 {len(ern['surprise_active'])}",
                      f"壞消息（避開）{len(rev['bad_news_active']) + len(ern['bad_news_active'])}"])
with t1:
    st.markdown("**這個月營收比自己過去常態好最多的前 10%**（月營收驚喜 SUR）。回測：2018–2025 年化 +29.8%、每年都贏一般股票；"
                "效果集中在中小型股。")
    table(rev["surprise_active"], "rev", "rev_top")
with t2:
    st.markdown("**這一季淨利比自己過去常態好最多的前 10%**（盈餘驚喜 SUE）。回測：2018–2025 年化 +29.4%；大型股也有效。")
    table(ern["surprise_active"], "ern", "ern_top")
with t3:
    st.markdown("**驚喜最低 10%**：回測中之後每月比一般股票差 1.1～1.4%（2026 年 9 個月幾乎每月都輸）。持有中的股票若出現在這裡，要特別注意。")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 營收壞消息")
        table(rev["bad_news_active"], "rev", "rev_bad")
    with c2:
        st.markdown("##### 盈餘壞消息")
        table(ern["bad_news_active"], "ern", "ern_bad")
