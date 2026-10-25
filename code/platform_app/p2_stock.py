"""個股：股價、月營收、季報、目前的驚喜／壞消息標籤；情緒之後補。"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common

names = common.stock_names()
opts = [f"{c} {n}" for c, n in names.items()]
code0 = st.session_state.pop("goto_code", None) or st.session_state.get("stock_code", "2330")
idx = next((i for i, o in enumerate(opts) if o.split()[0] == code0), 0)
pick = st.selectbox("股票", opts, index=idx, label_visibility="collapsed", placeholder="輸入代號或名稱")
code = pick.split()[0]
st.session_state["stock_code"] = code

px = common.prices()
p = px[px.code == code].set_index("date")
rev, ern = common.signals("revenue_signals"), common.signals("earnings_signals")
labels = []
for nm, sig in [("營收", rev), ("盈餘", ern)]:
    for k in ("surprise_active", "bad_news_active"):
        for r in sig.get(k, []):
            if r["code"] == code:
                labels.append((r["label"], f"{r['period']}・{r['announce_date']} 公布・持有到 {r['exit_date']}・驚喜分數 {r['surprise']:+.2f}"))

sent = common.signals("sentiment_stocks")
sser = pd.DataFrame(sent.get("series", {}).get(code, []))
if len(sser):
    sser["date"] = pd.to_datetime(sser.date)
    s_last = sser.iloc[-1]
    sent_txt = (f"{s_last.date:%m-%d} 留言 {int(s_last.n)} 則、淨情緒 " + ("—" if pd.isna(s_last.net) else f"{s_last.net:+.0%}")
                + f"；近期有討論 {len(sser)} 天")
else:
    sent_txt = "近期 PTT 沒有以這檔為標題的討論"

# ---------------------------------------------------------------- 標頭
close = p.close.dropna()
chg = close.iloc[-1] / close.iloc[-2] - 1 if len(close) >= 2 else None
y1 = close.loc[close.index[-1] - pd.Timedelta(days=365):]
lo, hi = y1.min(), y1.max()
pos = 0 if hi <= lo else (close.iloc[-1] - lo) / (hi - lo) * 100
lab_html = "".join(f'<div class="row">{common.badge(l)}<span style="opacity:.65;font-size:.78rem">{common.esc(d)}</span></div>'
                   for l, d in labels) or '<div class="row" style="opacity:.6">目前沒有驚喜或壞消息標籤</div>'
st.markdown(
    f'<div class="hero"><div><div><span class="nm">{common.esc(names.get(code, ""))}</span><span class="cd">{code} · 上市</span></div>'
    f'<div><span class="px">{close.iloc[-1]:,.2f}</span><span class="chg">{common.signed_html(chg, 2)}</span></div>'
    f'<div class="meta">資料日 {close.index[-1]:%Y-%m-%d}・證交所收盤價（未還原）</div>'
    f'<div class="row"><span class="k">近一年</span>{lo:,.2f} – {hi:,.2f}<div class="range"><i style="left:calc({pos:.0f}% - 1px)"></i></div></div></div>'
    f'<div class="right"><div class="row"><b>財報訊號</b></div>{lab_html}'
    f'<div class="row"><span class="badge muted">PTT 情緒</span><span style="opacity:.65;font-size:.78rem">{common.esc(sent_txt)}</span></div></div></div>',
    unsafe_allow_html=True)

t1, t2, t3, t4 = st.tabs(["股價", "月營收", "季報", "情緒"])
with t1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, name="收盤", line=dict(color=common.C_MAIN, width=1.8)))
    fig.add_trace(go.Scatter(x=close.index, y=close.rolling(20).mean(), name="20 日均線", line=dict(color=common.C_ETF, width=1.1, dash="dot")))
    fig.add_trace(go.Scatter(x=close.index, y=close.rolling(60).mean(), name="60 日均線", line=dict(color=common.C_ALT, width=1.1, dash="dot")))
    marks = sorted(((pd.Timestamp(r["entry_date"]), r) for sig in (rev, ern) for r in sig.get("all_recent", [])
                    if r["code"] == code and r["label"] and pd.Timestamp(r["entry_date"]) <= close.index[-1]), key=lambda z: z[0])
    for k, (d, r) in enumerate(marks):                                   # 標籤上下錯開，避免日期接近時重疊
        col = common.UP if "驚喜" in r["label"] else common.DOWN
        fig.add_vline(x=d, line=dict(color=col, width=1, dash="dash"))
        fig.add_annotation(x=d, y=1 - 0.07 * (k % 3), yref="paper", text=f"{r['label'].replace('（避開）', '')} {r['period']}",
                           showarrow=False, font=dict(size=11, color=col), xanchor="right", bgcolor="rgba(255,255,255,.75)")
    fig = common.fig_layout(fig, "收盤價（2025-09 起，證交所公開資料；虛線＝驚喜／壞消息的進場日）", 420)
    fig.update_layout(legend=dict(x=0, xanchor="left", y=-0.12, yanchor="top"), margin=dict(t=56))
    common.chart(fig)
    vol = p.value.dropna() / 1e8
    fv = go.Figure(go.Bar(x=vol.index, y=vol, marker_color=common.C_POOL, name="成交金額（億）"))
    common.chart(common.fig_layout(fv, "成交金額（億元）", 200, legend=False))

with t2:
    r = common.revenue()
    r = r[r.code == code].set_index("per").R
    if r.empty:
        common.empty("沒有月營收資料")
    else:
        r = r.iloc[-37:]
        yoy = (r / r.shift(12) - 1).iloc[-24:]
        r24 = r.iloc[-24:] / 1e5
        fig = go.Figure()
        fig.add_trace(go.Bar(x=r24.index.astype(str), y=r24, name="單月營收（億）", marker_color=common.C_MAIN))
        fig.add_trace(go.Scatter(x=yoy.index.astype(str), y=yoy * 100, name="年增率（%）", yaxis="y2", line=dict(color=common.UP, width=1.8)))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", ticksuffix="%", showgrid=False))
        common.chart(common.fig_layout(fig, "近 24 個月營收與年增率（以合併為主）", 380))
        mine = [x for x in rev.get("all_recent", []) if x["code"] == code]
        if mine:
            x = mine[0]
            common.kpis([("最新月份", x["period"], f"{x['announce_date']} 公布"), ("年增率", common.signed_html(x.get("yoy")), ""),
                         ("驚喜分數 SUR", f"{x['surprise']:+.2f}", f"第 {x['decile']} 組（10＝最好）"),
                         ("本月門檻", f"{rev['thresholds']['p90']:+.2f}", "高於此＝營收驚喜")])

with t3:
    e = common.earnings()
    e = e[e.code == code].set_index("per").N
    if e.empty:
        common.empty("沒有季報資料")
    else:
        e12 = e.iloc[-12:] / 1e5
        fig = go.Figure(go.Bar(x=e12.index.astype(str), y=e12, name="單季歸屬母公司淨利（億）",
                               marker_color=[common.UP if v >= 0 else common.DOWN for v in e12]))
        common.chart(common.fig_layout(fig, "近 12 季單季淨利（億元，歸屬母公司）", 360, legend=False))
        mine = [x for x in ern.get("all_recent", []) if x["code"] == code]
        if mine:
            x = mine[0]
            ly = x.get("net_income_last_year_k")
            common.kpis([("最新一季", x["period"], f"{x['announce_date']} 公布"),
                         ("單季淨利", f"{x['net_income_k'] / 1e5:,.2f} 億", "" if ly is None else f"去年同季 {ly / 1e5:,.2f} 億"),
                         ("驚喜分數 SUE", f"{x['surprise']:+.2f}", f"第 {x['decile']} 組（10＝最好）"),
                         ("本季門檻", f"{ern['thresholds']['p90']:+.2f}", "高於此＝盈餘驚喜")])

with t4:
    st.info(sent.get("note") or "情緒只當描述，不是買賣訊號。")
    if not len(sser):
        common.empty("近期 PTT Stock 板沒有標題提到這檔的文章（每日爬蟲從 2026-08 起）。")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sser.date, y=sser.n, name="留言數", marker_color="rgba(148,163,184,.55)", yaxis="y2"))
        fig.add_trace(go.Scatter(x=sser.date, y=sser.net * 100, name="淨情緒（%）", mode="lines+markers", line=dict(color=common.C_MAIN, width=1.8),
                                 marker=dict(size=6, color=[common.UP if (v or 0) > 0 else common.DOWN for v in sser.net])))
        fig.update_layout(yaxis=dict(ticksuffix="%", title="淨情緒"), yaxis2=dict(overlaying="y", side="right", showgrid=False, title="留言數"))
        common.chart(common.fig_layout(fig, "PTT 討論（標題提到這檔的文章底下的留言，依交易日）", 360))
        mkd = pd.DataFrame(common.signals("sentiment_market").get("days", []))
        common.kpis([("最近有討論的一天", f"{s_last.date:%m-%d}", f"留言 {int(s_last.n)} 則"),
                     ("淨情緒", "—" if pd.isna(s_last.net) else common.signed_html(s_last.net, 0), "（正 − 負）÷（正 + 負）"),
                     ("同日市場淨情緒", "—" if mkd.empty else common.signed_html(mkd.set_index("date").net.get(f"{s_last.date:%Y-%m-%d}"), 0), "盤中／盤後閒聊"),
                     ("近期有討論的天數", f"{len(sser)} 天", f"共 {int(sser.n.sum()):,} 則留言")])
    st.caption("CMoney「散戶極度悲觀」標籤需要 CMoney 的每日資料（token 每天過期），目前沒有接上。")
