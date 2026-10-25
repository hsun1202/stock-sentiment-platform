# -*- coding: utf-8 -*-
"""虛擬交易引擎的檢查：漲跌停價和證交所即時資料比對、升降單位、手續費與證交稅、下單與撮合情境。
cd D:/專題claude/platform_app && python test_trading.py"""
import datetime as dt
import random

import pandas as pd

import trading as T

px = pd.read_csv("../platform_feed/data/prices.csv", dtype={"code": str})
last = px[px.date == px.date.max()]
codes = list(last[last.code.str.match(r"^[1-9]\d{3}$")].sample(80, random_state=1).code) + ["0050", "0056", "00878", "006208", "00632R"]
bad = 0
q = {}
for i in range(0, len(codes), 20):
    q.update(T.quotes(codes[i:i + 20]))
for c, v in q.items():
    if v["prev"] and v["up"]:
        up, dn = T.limits(v["prev"], T.is_etf(c))
        if abs(up - v["up"]) > 1e-9 or abs(dn - v["down"]) > 1e-9:
            bad += 1
            print("不一致", c, v["prev"], (up, dn), (v["up"], v["down"]))
print(f"漲跌停比對：{len(q)} 檔，不一致 {bad} 檔")

assert T.on_tick(103.5) and not T.on_tick(103.3) and T.on_tick(2385) and not T.on_tick(2386) and T.on_tick(9.99) and not T.on_tick(10.01)
assert T.on_tick(65.55, etf=True) and not T.on_tick(65.53, etf=True) and T.on_tick(45.01, etf=True)
assert T.fee(103_500, 0.6) == 88 and T.fee(5_000, 0.6) == 20 and T.tax(103_500, "4938") == 310 and T.tax(100_000, "0050") == 100
print("升降單位、手續費、證交稅：通過")

# 下單與撮合情境
days = {dt.date(2026, 10, 21), dt.date(2026, 10, 22)}
a = T.new_account(1_000_000, 0.6)
quote = {"prev": 105.0, "asks": [104.0], "bids": [103.5], "last": 103.5, "up": 115.5, "down": 94.5}
intraday = dt.datetime(2026, 10, 22, 10, 0)
ok, msg = T.place(a, code="4938", name="和碩", side="buy", board="lot", qty=2, kind="limit", limit=104.0, quote=quote, prev_close=105, now=intraday, trading_days=days)
assert ok and a["orders"][-1]["status"] == "filled" and a["orders"][-1]["fill_price"] == 104.0, msg
assert a["positions"]["4938"]["qty"] == 2000 and a["cash"] == 1_000_000 - 208_000 - 177
ok, msg = T.place(a, code="4938", name="和碩", side="buy", board="lot", qty=1, kind="limit", limit=103.3, quote=quote, prev_close=105, now=intraday, trading_days=days)
assert not ok and "升降單位" in msg
ok, msg = T.place(a, code="4938", name="和碩", side="buy", board="lot", qty=1, kind="limit", limit=116.0, quote=quote, prev_close=105, now=intraday, trading_days=days)
assert not ok and "漲跌停" in msg
ok, msg = T.place(a, code="4938", name="和碩", side="sell", board="lot", qty=3, kind="market", limit=None, quote=quote, prev_close=105, now=intraday, trading_days=days)
assert not ok and "庫存不足" in msg
ok, msg = T.place(a, code="4938", name="和碩", side="buy", board="lot", qty=1, kind="limit", limit=103.0, quote=quote, prev_close=105, now=intraday, trading_days=days)
assert ok and a["orders"][-1]["status"] == "pending"                    # 低於賣價：掛著
ok, msg = T.place(a, code="4938", name="和碩", side="buy", board="lot", qty=50, kind="limit", limit=110.0, quote=quote, prev_close=105, now=intraday, trading_days=days)
assert not ok and "現金不足" in msg
# 收盤後撮合：當天最低 102.5 ≤ 103 → 以 103 成交
T.settle(a, {"4938": {dt.date(2026, 10, 22): (107, 108, 102.5, 103.5, 105)}}, dt.date(2026, 10, 22))
assert a["orders"][-1]["status"] == "filled" and a["orders"][-1]["fill_price"] == 103.0
# 盤後下單（週六）→ 下週一開盤撮合；賣 1 張零股 500 股
sat = dt.datetime(2026, 10, 24, 15, 0)
ok, msg = T.place(a, code="4938", name="和碩", side="sell", board="odd", qty=500, kind="limit", limit=100.0, quote=None, prev_close=103.5, now=sat, trading_days=days)
assert ok and a["orders"][-1]["session"] == "2026-10-26", a["orders"][-1]
T.settle(a, {"4938": {dt.date(2026, 10, 26): (101, 102, 99, 100, 103.5)}}, dt.date(2026, 10, 26))
o = a["orders"][-1]
assert o["status"] == "filled" and o["fill_price"] == 101 and o["tax"] == T.tax(50_500, "4938"), o
# 鎖漲停買不到
ok, msg = T.place(a, code="4938", name="和碩", side="buy", board="lot", qty=1, kind="market", limit=None, quote=None, prev_close=100, now=sat, trading_days=days)
T.settle(a, {"4938": {dt.date(2026, 10, 26): (110, 110, 110, 110, 100)}}, dt.date(2026, 10, 26))
assert a["orders"][-1]["status"] == "expired"
r = a["realized"][-1]
print(f"情境測試：通過（持股 {a['positions']['4938']['qty']} 股、現金 {a['cash']:,.0f}、已實現 {r['pnl']:+,.0f}）")
