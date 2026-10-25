"""虛擬交易：帳戶、下單（台股規則）、持股與損益；持股旁邊標出財報壞消息。引擎見 trading.py。"""
from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common
import trading as T

px = common.prices()
trade_days = set(px.date.dt.date.unique())
now = datetime.now()
in_session = T.is_session(now, trade_days)
codes_ok = px.code.str.match(r"^([1-9]\d{3}|00\d{2,4}[A-Z]?)$")
names = px[codes_ok].drop_duplicates("code", keep="last").set_index("code").name.sort_index()
last_close = px.groupby("code").close.last()

# ---------------------------------------------------------------- 財報標籤（持股警示、下單提醒）
rev, ern = common.signals("revenue_signals"), common.signals("earnings_signals")
tags: dict[str, list[str]] = {}
for sig in (rev, ern):
    for k in ("surprise_active", "bad_news_active"):
        for r in sig.get(k, []):
            tags.setdefault(r["code"], []).append(f"{'⚠ ' if '壞消息' in r['label'] else ''}{r['label']}（到 {r['exit_date'][5:]}）")

# ---------------------------------------------------------------- 帳戶與撮合
acct = T.load()


@st.cache_data(ttl=20, show_spinner=False)
def live(codes: tuple) -> dict:
    out = {}
    for i in range(0, len(codes), 20):
        out.update(T.quotes(list(codes[i:i + 20])))
    return out


def bars_for(codes: set, q: dict) -> dict:
    """{code: {date: (開, 高, 低, 收, 前一日收)}}：證交所日行情，再補上即時報價裡「今天已收盤」的那一天"""
    out = {}
    sub = px[px.code.isin(codes)].sort_values(["code", "date"])
    for c, g in sub.groupby("code"):
        prev = g.close.shift(1)
        out[c] = {d.date(): (o, h, l, cl, pv) for d, o, h, l, cl, pv in zip(g.date, g.open, g.high, g.low, g.close, prev) if pd.notna(pv) and pd.notna(o)}
    if now.time() >= T.CLOSE_T:
        for c in codes:
            v = q.get(c) or {}
            if v.get("date") == now.strftime("%Y%m%d") and v.get("open") and v.get("prev"):
                out.setdefault(c, {})[now.date()] = (v["open"], v["high"], v["low"], v["last"] or v["prev"], v["prev"])
    return out


pend = {o["code"] for o in acct["orders"] if o["status"] == "pending"}
held = set(acct["positions"])
q_all = live(tuple(sorted(pend | held)))
if pend:
    b = bars_for(pend, q_all)
    if T.settle(acct, b, max(max(v) for v in b.values()) if b else max(trade_days)):
        T.save(acct)


def mark(code: str) -> float:
    v = q_all.get(code) or {}
    return v.get("last") or v.get("prev") or float(last_close.get(code, 0) or 0)


# ---------------------------------------------------------------- 總覽
mv = sum(p["qty"] * mark(c) for c, p in acct["positions"].items())
cost = sum(p["cost"] for p in acct["positions"].values())
real = sum(r["pnl"] for r in acct["realized"])
equity = acct["cash"] + mv
common.page_title("虛擬交易", f"{'盤中（09:00–13:30）' if in_session else '已收盤'}・下一個撮合日 {T.next_session(now, trade_days)}・"
                  f"開戶 {acct['created'][:10]}・手續費 {acct['fee_discount'] * 10:g} 折")
common.kpis([
    ("總資產", f"{equity:,.0f}", f"本金 {acct['initial']:,.0f}", "accent"),
    ("總報酬", common.signed_html(equity / acct["initial"] - 1, 2), ""),
    ("現金", f"{acct['cash']:,.0f}", f"委託中預留 {T.pending_buy_cash(acct):,.0f}"),
    ("股票市值", f"{mv:,.0f}", f"{len(acct['positions'])} 檔"),
    ("未實現損益", common.signed_html(mv - cost, 0, 1, " 元"), "已扣買進手續費、未扣賣出成本"),
    ("已實現損益", common.signed_html(real, 0, 1, " 元"), f"{len(acct['realized'])} 筆賣出"),
])

left, right = st.columns([5, 7], gap="large")

# ---------------------------------------------------------------- 下單
with left:
    st.markdown("#### 下單")
    opts = [f"{c} {n}" for c, n in names.items()]
    c0 = st.session_state.get("trade_code") or st.session_state.get("stock_code") or "2330"
    pick = st.selectbox("股票", opts, index=next((i for i, o in enumerate(opts) if o.split()[0] == c0), 0))
    code = pick.split()[0]
    st.session_state["trade_code"] = code
    name = names.get(code, "")
    etf = T.is_etf(code)
    q = live((code,)).get(code) or {}
    prev = q.get("prev") or float(last_close.get(code, 0) or 0)
    lastp = q.get("last") or prev
    up, dn = (q["up"], q["down"]) if q.get("up") else T.limits(prev, etf)
    chg = lastp / prev - 1 if prev else None
    qtime = f"{q['date'][4:6]}/{q['date'][6:]} {q['time']}" if q.get("date") else "（即時報價取得失敗，改用最近收盤）"
    asks = "".join(f"<div class='row'><span class='k'>賣{i + 1}</span><span class='up'>{p:g}</span> <span style='opacity:.6'>{s}</span></div>"
                   for i, (p, s) in enumerate(zip(q.get("asks", [])[:3], q.get("ask_sz", [])[:3] + [""] * 3)))
    bids = "".join(f"<div class='row'><span class='k'>買{i + 1}</span><span class='down'>{p:g}</span> <span style='opacity:.6'>{s}</span></div>"
                   for i, (p, s) in enumerate(zip(q.get("bids", [])[:3], q.get("bid_sz", [])[:3] + [""] * 3)))
    st.markdown(f'<div class="hero"><div><div><span class="nm">{common.esc(name)}</span><span class="cd">{code}</span></div>'
                f'<div><span class="px">{lastp:,.2f}</span><span class="chg">{common.signed_html(chg, 2)}</span></div>'
                f'<div class="meta">證交所即時報價 {common.esc(qtime)}</div>'
                f'<div class="row"><span class="k">漲停</span><span class="up">{up:g}</span>　<span class="k">跌停</span><span class="down">{dn:g}</span></div></div>'
                f'<div class="right">{asks}{bids}</div></div>', unsafe_allow_html=True)
    for t in tags.get(code, []):
        (st.warning if "壞消息" in t else st.info)(f"財報訊號：{t}")

    c1, c2 = st.columns(2)
    side = c1.radio("買賣", ["買進", "賣出"], horizontal=True)
    board = c2.radio("單位", ["整股（張）", "零股（股）"], horizontal=True)
    c3, c4 = st.columns(2)
    kind = c3.radio("價格", ["限價", "市價"], horizontal=True)
    qty = c4.number_input("數量（張）" if board.startswith("整股") else "數量（股）", min_value=1, max_value=999 if board.startswith("零股") else 500,
                          value=1, step=1)
    tk = T.tick_c(lastp, etf) / 100
    limit = None
    if kind == "限價":
        default = min(max(round(round(lastp / tk) * tk, 2), dn), up)
        limit = st.number_input(f"委託價（這個價位每檔 {tk:g} 元）", min_value=float(dn), max_value=float(up), value=float(default), step=float(tk), format="%.2f", key=f"limit_{code}")
    shares = qty * T.LOT if board.startswith("整股") else qty
    px_est = limit if kind == "限價" else ((q.get("asks") or [up])[0] if side == "買進" else (q.get("bids") or [dn])[0])
    amt = px_est * shares
    f = T.fee(amt, acct["fee_discount"])
    tx = T.tax(amt, code) if side == "賣出" else 0
    st.caption(f"預估：{shares:,} 股 × {px_est:g} ＝ {amt:,.0f} 元，手續費 {f:,} 元" + (f"、證交稅 {tx:,} 元，入帳約 {amt - f - tx:,.0f} 元" if tx else f"，共約 {amt + f:,.0f} 元"))
    if st.button("送出委託", type="primary", use_container_width=True):
        ok, msg = T.place(acct, code=code, name=name, side="buy" if side == "買進" else "sell", board="lot" if board.startswith("整股") else "odd",
                          qty=int(qty), kind="limit" if kind == "限價" else "market", limit=limit, quote=q, prev_close=prev, now=datetime.now(),
                          trading_days=trade_days)
        if ok:
            T.save(acct)
            st.session_state["trade_msg"] = ("success", msg)
            st.rerun()
        st.error(msg)
    if "trade_msg" in st.session_state:
        kind_, m = st.session_state.pop("trade_msg")
        st.success(m)
    st.caption("規則：09:00–13:30 盤中可立即成交；盤後的單排到下一個交易日開盤。價格要符合升降單位、在漲跌停內；鎖漲停買不到、鎖跌停賣不掉。"
               "手續費 0.1425%×折扣（最低 20 元），賣出證交稅股票 0.3%、ETF 0.1%。簡化：成交即交割、不計除權息、不做當沖與融資券。")

# ---------------------------------------------------------------- 持股與損益
with right:
    st.markdown("#### 持股")
    if not acct["positions"]:
        common.empty("還沒有持股")
    else:
        rows = []
        for c, p in acct["positions"].items():
            m = mark(c)
            rows.append({"代號": c, "名稱": p.get("name") or names.get(c, ""), "股數": p["qty"], "均價（含費）": p["cost"] / p["qty"], "現價": m,
                         "市值": p["qty"] * m, "未實現損益": p["qty"] * m - p["cost"], "報酬率": p["qty"] * m / p["cost"] - 1,
                         "財報標籤": "、".join(tags.get(c, [])) or "—"})
        h = pd.DataFrame(rows)

        def warn(v):
            return "color: #e5484d; font-weight: 600" if "壞消息" in str(v) else ""
        st.dataframe(h.style.format({"股數": "{:,}", "均價（含費）": "{:,.2f}", "現價": "{:,.2f}", "市值": "{:,.0f}", "未實現損益": "{:+,.0f}", "報酬率": "{:+.2%}"})
                     .map(common.color_signed, subset=["未實現損益", "報酬率"]).map(warn, subset=["財報標籤"]), hide_index=True)
        bad = [r for r in rows if "壞消息" in r["財報標籤"]]
        if bad:
            st.warning("持股中有財報壞消息：" + "、".join(f"{r['代號']} {r['名稱']}" for r in bad) +
                       "。研究結果：壞消息最低 10% 的股票之後每月平均比一般股票差 1.1～1.4%（不保證未來）。")

    st.markdown("#### 委託紀錄")
    if not acct["orders"]:
        common.empty("還沒有委託")
    else:
        stt = {"filled": "已成交", "pending": "委託中", "cancelled": "已取消", "expired": "未成交失效"}
        od = pd.DataFrame([{"#": o["id"], "時間": o["time"].replace("T", " ")[5:16], "代號": o["code"], "名稱": o["name"],
                            "買賣": "買" if o["side"] == "buy" else "賣", "股數": o["qty"], "條件": "市價" if o["kind"] == "market" else f"{o['limit']:g}",
                            "狀態": stt[o["status"]], "成交價": o.get("fill_price"), "手續費": o.get("fee"), "證交稅": o.get("tax"),
                            "撮合日": o["session"]} for o in reversed(acct["orders"])])
        st.dataframe(od.style.format({"股數": "{:,}", "成交價": "{:g}", "手續費": "{:,.0f}", "證交稅": "{:,.0f}"}, na_rep="—"), hide_index=True, height=260)
        pend_ids = [o["id"] for o in acct["orders"] if o["status"] == "pending"]
        if pend_ids:
            cc1, cc2 = st.columns([3, 1])
            oid = cc1.selectbox("取消委託", pend_ids, format_func=lambda i: f"#{i}", label_visibility="collapsed")
            if cc2.button("取消"):
                T.cancel(acct, oid)
                T.save(acct)
                st.rerun()

    if acct["realized"]:
        st.markdown("#### 已實現損益")
        rz = pd.DataFrame(acct["realized"])[["date", "code", "name", "qty", "price", "proceeds", "basis", "pnl"]]
        rz.columns = ["日期", "代號", "名稱", "股數", "賣價", "入帳", "成本", "損益"]
        st.dataframe(rz.iloc[::-1].style.format({"股數": "{:,}", "賣價": "{:g}", "入帳": "{:,.0f}", "成本": "{:,.0f}", "損益": "{:+,.0f}"})
                     .map(common.color_signed, subset=["損益"]), hide_index=True)

    # 帳戶淨值 vs 0050（從開戶日起，用每日收盤估）
    start = pd.Timestamp(acct["created"][:10])
    days = sorted(d for d in px.date.unique() if d >= start)
    fills = sorted((o for o in acct["orders"] if o["status"] == "filled"), key=lambda o: o["filled_at"])
    if len(days) >= 2 and fills:
        cl = px[px.date.isin(days)].pivot(index="date", columns="code", values="close").ffill()
        cash, pos, i, nav = acct["initial"], {}, 0, []
        for d in days:
            while i < len(fills) and pd.Timestamp(fills[i]["filled_at"][:10]) <= d:
                o = fills[i]
                amt = o["fill_price"] * o["qty"]
                sgn = 1 if o["side"] == "buy" else -1
                cash -= sgn * amt + o["fee"] + o["tax"]
                pos[o["code"]] = pos.get(o["code"], 0) + sgn * o["qty"]
                i += 1
            nav.append(cash + sum(q_ * cl.at[d, c] for c, q_ in pos.items() if c in cl.columns and pd.notna(cl.at[d, c])))
        nav = pd.Series(nav, index=days) / acct["initial"]
        e50 = cl["0050"] / cl["0050"].iloc[0] if "0050" in cl else None
        fig = go.Figure(go.Scatter(x=nav.index, y=nav, name="我的帳戶", line=dict(color=common.UP, width=2)))
        if e50 is not None:
            fig.add_trace(go.Scatter(x=e50.index, y=e50, name="0050（未含息）", line=dict(color=common.C_ETF, width=1.4)))
        common.chart(common.fig_layout(fig, "帳戶淨值（開戶＝1，每日收盤估）", 300))

    with st.expander("重設帳戶"):
        ini = st.number_input("本金（元）", min_value=100_000, max_value=100_000_000, value=int(acct["initial"]), step=100_000)
        disc = st.number_input("手續費折扣（0.6＝6 折）", min_value=0.1, max_value=1.0, value=float(acct["fee_discount"]), step=0.05, format="%.2f")
        sure = st.checkbox("我確定要清空持股與紀錄")
        if st.button("重設", disabled=not sure):
            T.save(T.new_account(int(ini), float(disc)))
            st.rerun()
