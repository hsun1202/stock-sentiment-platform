"""市場情緒：PTT Stock 板盤中／盤後閒聊的每日淨情緒、討論量；今天最熱門的個股（附財報標籤）。只當描述，不是買賣訊號。"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import common

mk, stk = common.signals("sentiment_market"), common.signals("sentiment_stocks")
if not mk or not mk.get("days"):
    common.empty("還沒有情緒資料：先執行 python platform_feed/ptt_daily.py --days 60 再 python platform_feed/ptt_sentiment.py")
    st.stop()

d = pd.DataFrame(mk["days"])
d["date"] = pd.to_datetime(d.date)
d = d[d.date <= pd.Timestamp(mk["asof"]) + pd.Timedelta(days=7)]
last = d.iloc[-1]
live_day = last.date > pd.Timestamp(mk["asof"])
common.page_title("市場情緒", f"PTT Stock 板 盤中／盤後閒聊・最新交易日 {last.date:%Y-%m-%d}{'（還沒收盤，到目前為止）' if live_day else ''}・"
                  "前一交易日 13:30 到當天 13:30 的留言")
st.info(mk["note"])


def pct_html(x):
    return "—" if x is None or pd.isna(x) else common.signed_html(x, 1)


z = last.get("net_z")
mood = "—" if pd.isna(z) else ("比平常樂觀" if z > 1 else "比平常悲觀" if z < -1 else "和平常差不多")
common.kpis([
    ("淨情緒", pct_html(last.net), "（正 − 負）÷（正 + 負）", "accent" if (last.net or 0) > 0 else "calm"),
    ("近 20 日平均", pct_html(last.net_avg20), f"今天 {mood}" + ("" if pd.isna(z) else f"（z＝{z:+.1f}）")),
    ("留言數", f"{int(last.n):,}", "" if pd.isna(last.n_avg20) else f"近 20 日平均 {last.n_avg20:,.0f}"),
    ("看多／看空留言", f"{int(last.pos):,}／{int(last.neg):,}", "其餘為中立"),
    ("資料天數", f"{len(d)} 天", f"{d.date.min():%m-%d} 起"),
])

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=d.date, y=d.n, name="留言數", marker_color="rgba(148,163,184,.45)"), secondary_y=True)
fig.add_trace(go.Scatter(x=d.date, y=d.net * 100, name="淨情緒（%）", mode="lines+markers",
                         line=dict(color=common.C_MAIN, width=2), marker=dict(size=5,
                         color=[common.UP if v > 0 else common.DOWN for v in d.net.fillna(0)])), secondary_y=False)
fig.add_trace(go.Scatter(x=d.date, y=d.net_avg20 * 100, name="近 20 日平均", line=dict(color=common.C_ETF, width=1.2, dash="dot")), secondary_y=False)
fig.update_yaxes(title_text="淨情緒（%）", ticksuffix="%", secondary_y=False)
fig.update_yaxes(title_text="留言數", showgrid=False, secondary_y=True)
common.chart(common.fig_layout(fig, "每日市場淨情緒與討論量", 380))

st.markdown(f"#### 今天討論最多的個股（{stk.get('latest_day', '—')}，標題提到該股的文章底下的留言）")
top = stk.get("top", [])
if not top:
    common.empty("今天沒有個股討論")
else:
    rev, ern = common.signals("revenue_signals"), common.signals("earnings_signals")
    tags = {}
    for sig in (rev, ern):
        for k in ("surprise_active", "bad_news_active"):
            for r in sig.get(k, []):
                tags.setdefault(r["code"], []).append(r["label"])
    t = pd.DataFrame(top)
    t["attention"] = pd.to_numeric(t.attention, errors="coerce")
    t["net"] = pd.to_numeric(t.net, errors="coerce")
    t = pd.DataFrame({"代號": t.code, "名稱": t.name, "留言數": t.n, "文章數": t.arts, "淨情緒": t.net,
                      "熱度（相對近 20 日）": t.attention, "財報標籤": [("、".join(tags.get(c, [])) or "—") for c in t.code]})

    def warn(v):
        return "color: #e5484d; font-weight: 600" if "壞消息" in str(v) else ""
    ev = st.dataframe(t.style.format({"留言數": "{:,}", "淨情緒": "{:+.0%}", "熱度（相對近 20 日）": "{:.1f} 倍"}, na_rep="—")
                      .map(common.color_signed, subset=["淨情緒"]).map(warn, subset=["財報標籤"]),
                      hide_index=True, on_select="rerun", selection_mode="single-row", key="sent_top", height=min(38 * len(t) + 40, 560))
    st.caption("點一列看個股。熱度＝今天留言數 ÷ 該股近 20 個有討論的交易日平均。分類器：中文 BERT 以 6,300 則人工金標微調（與金標一致率約 69%）。")
    if ev.selection.rows:
        common.goto_stock(t.iloc[ev.selection.rows[0]]["代號"])
