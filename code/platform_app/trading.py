"""
虛擬交易引擎（台股規則）。帳戶存在 platform_app/data/account.json（本機、單一使用者）。

規則（簡化處標「簡化」）：
- 交易時段 09:00–13:30（週一至週五、交易日）。盤後下的單排到下一個交易日。
- 整股以張（1,000 股）為單位；零股 1–999 股（簡化：盤中零股與整股同樣撮合）。
- 價格要符合升降單位，且在漲跌停（前一日收盤 ±10%，再依升降單位取整）之內。市價單以漲停價（買）／跌停價（賣）檢查資金。
- 撮合：
  - 盤中：買單限價 ≥ 最佳賣價就立刻以最佳賣價成交；賣單限價 ≤ 最佳買價就以最佳買價成交；漲停鎖死（沒有賣價）買不到、跌停鎖死賣不到。
    沒成交的掛著（當日有效），收盤後用當天最高／最低價判斷：買單 最低價 ≤ 限價 就以限價成交（簡化：不考慮排隊順序）。
  - 盤前／盤後下的單：用下一個交易日的開盤價撮合（限價優於開盤就以開盤價成交），否則同上以最高／最低判斷；
    開盤就鎖漲停（開＝高＝低＝漲停）的買單、鎖跌停的賣單不成交。當天沒成交就失效。
- 手續費 0.1425% × 折扣（無條件捨去），每筆最低 20 元；賣出另扣證交稅：股票 0.3%、ETF 0.1%（無條件捨去）。
- 簡化：成交後立刻扣款／入帳（實際是 T+2 交割）；不計除權息、不做當沖減半證交稅、不做融資融券。
"""
import json
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path

import requests

FEE, MIN_FEE, TAX_STOCK, TAX_ETF, LOT = 0.001425, 20, 0.003, 0.001, 1000
OPEN_T, CLOSE_T = time(9, 0), time(13, 30)
ACCT = Path(__file__).resolve().parent / "data" / "account.json"
MIS = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={}&json=1&delay=0"


# ---------------------------------------------------------------- 價格規則（以「分」整數計算，避免浮點誤差）

def is_etf(code: str) -> bool:
    return code.startswith("00")


def tick_c(price: float, etf: bool = False) -> int:
    """升降單位（分）"""
    if etf:
        return 1 if price < 50 else 5
    for lim, t in ((10, 1), (50, 5), (100, 10), (500, 50), (1000, 100)):
        if price < lim:
            return t
    return 500


def on_tick(price: float, etf: bool = False) -> bool:
    c = round(price * 100)
    return abs(price * 100 - c) < 1e-6 and c % tick_c(price, etf) == 0


def limits(prev_close: float, etf: bool = False) -> tuple[float, float]:
    """漲停、跌停價"""
    up_raw, dn_raw = prev_close * 110, prev_close * 90                  # 分
    tu, td = tick_c(up_raw / 100, etf), tick_c(dn_raw / 100, etf)
    up = math.floor(round(up_raw, 6) / tu) * tu
    dn = math.ceil(round(dn_raw, 6) / td) * td
    return up / 100, dn / 100


def fee(amount: float, discount: float) -> int:
    return max(MIN_FEE, math.floor(amount * FEE * discount)) if amount > 0 else 0


def tax(amount: float, code: str) -> int:
    return math.floor(amount * (TAX_ETF if is_etf(code) else TAX_STOCK))


# ---------------------------------------------------------------- 即時報價

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def quotes(codes: list[str]) -> dict:
    """證交所基本市況報導（mis.twse.com.tw，延遲約數秒）。回傳 {code: {...}}；失敗回傳 {}"""
    if not codes:
        return {}
    ex = "|".join(f"tse_{c}.tw" for c in codes)
    try:
        j = requests.get(MIS.format(ex), headers={"User-Agent": "Mozilla/5.0"}, timeout=10).json()
    except (requests.RequestException, ValueError):
        return {}
    out = {}
    for m in j.get("msgArray", []):
        asks = [_f(x) for x in (m.get("a") or "").split("_") if _f(x)]
        bids = [_f(x) for x in (m.get("b") or "").split("_") if _f(x)]
        asz = [int(x) for x in (m.get("f") or "").split("_") if x.isdigit()]
        bsz = [int(x) for x in (m.get("g") or "").split("_") if x.isdigit()]
        out[m.get("c")] = {"last": _f(m.get("z")), "open": _f(m.get("o")), "high": _f(m.get("h")), "low": _f(m.get("l")),
                           "prev": _f(m.get("y")), "up": _f(m.get("u")), "down": _f(m.get("w")), "asks": asks, "bids": bids,
                           "ask_sz": asz, "bid_sz": bsz, "date": m.get("d"), "time": m.get("t"), "vol": _f(m.get("v"))}
    return out


# ---------------------------------------------------------------- 交易日

def is_session(now: datetime, trading_days: set) -> bool:
    d = now.date()
    known = d in trading_days or (d.weekday() < 5 and d > max(trading_days))   # 資料還沒到的今天：平日先當交易日
    return known and OPEN_T <= now.time() < CLOSE_T


def next_session(now: datetime, trading_days: set) -> date:
    """下一個可撮合的交易日（盤中＝今天）。未來的假日不知道，先用平日估；撮合時用「該日或之後第一個有行情的日子」。"""
    d = now.date()
    if is_session(now, trading_days) or (d.weekday() < 5 and now.time() < OPEN_T and (d in trading_days or d > max(trading_days))):
        return d
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# ---------------------------------------------------------------- 帳戶

def new_account(initial: int = 1_000_000, discount: float = 0.6) -> dict:
    return {"created": datetime.now().isoformat(timespec="seconds"), "initial": initial, "cash": initial, "fee_discount": discount,
            "orders": [], "positions": {}, "realized": [], "next_id": 1}


def load() -> dict:
    if ACCT.exists():
        return json.loads(ACCT.read_text(encoding="utf-8"))
    a = new_account()
    save(a)
    return a


def save(a: dict) -> None:
    ACCT.parent.mkdir(parents=True, exist_ok=True)
    tmp = ACCT.with_suffix(".tmp")
    tmp.write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(ACCT)


def pending_sell_qty(a: dict, code: str) -> int:
    return sum(o["qty"] for o in a["orders"] if o["code"] == code and o["side"] == "sell" and o["status"] == "pending")


def pending_buy_cash(a: dict) -> float:
    return sum(o["reserve"] for o in a["orders"] if o["side"] == "buy" and o["status"] == "pending")


def _fill(a: dict, o: dict, price: float, when: str) -> None:
    amt = price * o["qty"]
    f = fee(amt, a["fee_discount"])
    pos = a["positions"].setdefault(o["code"], {"qty": 0, "cost": 0.0, "name": o.get("name", "")})
    if o["side"] == "buy":
        a["cash"] -= amt + f
        pos["qty"] += o["qty"]
        pos["cost"] += amt + f                                          # 成本含手續費
        t = 0
    else:
        t = tax(amt, o["code"])
        a["cash"] += amt - f - t
        basis = pos["cost"] * o["qty"] / pos["qty"]
        a["realized"].append({"date": when[:10], "code": o["code"], "name": o.get("name", ""), "qty": o["qty"], "price": price,
                              "proceeds": amt - f - t, "basis": round(basis, 2), "pnl": round(amt - f - t - basis, 2)})
        pos["qty"] -= o["qty"]
        pos["cost"] -= basis
        if pos["qty"] == 0:
            del a["positions"][o["code"]]
    o.update(status="filled", fill_price=price, fee=f, tax=t, filled_at=when)


def place(a: dict, *, code: str, name: str, side: str, board: str, qty: int, kind: str, limit: float | None,
          quote: dict | None, prev_close: float, now: datetime, trading_days: set) -> tuple[bool, str]:
    """下單。回傳（是否接受, 訊息）。board＝'lot'（整股，qty 為張數）或 'odd'（零股，qty 為股數）"""
    etf = is_etf(code)
    shares = qty * LOT if board == "lot" else qty
    if board == "lot" and qty < 1:
        return False, "整股至少 1 張"
    if board == "odd" and not 1 <= qty <= 999:
        return False, "零股要 1–999 股"
    q = quote or {}
    if is_session(now, trading_days) and q.get("up") and q.get("down"):
        up, dn = q["up"], q["down"]                                     # 盤中直接用證交所公布的漲跌停（少數 ETF 的取整規則不同）
    else:
        up, dn = limits(prev_close, etf)
    if kind == "limit":
        if limit is None or not on_tick(limit, etf):
            return False, f"價格不符升降單位（這個價位每檔 {tick_c(limit or 0, etf) / 100:g} 元）"
        if not dn <= limit <= up:
            return False, f"價格超出漲跌停範圍（{dn:g}–{up:g}）"
    px_chk = limit if kind == "limit" else up
    if side == "buy":
        need = px_chk * shares + fee(px_chk * shares, a["fee_discount"])
        if need > a["cash"] - pending_buy_cash(a):
            return False, f"可用現金不足（需要約 {need:,.0f} 元，可用 {a['cash'] - pending_buy_cash(a):,.0f} 元）"
    else:
        have = a["positions"].get(code, {}).get("qty", 0) - pending_sell_qty(a, code)
        if shares > have:
            return False, f"庫存不足（可賣 {have:,} 股）"
    o = {"id": a["next_id"], "time": now.isoformat(timespec="seconds"), "code": code, "name": name, "side": side, "board": board,
         "qty": shares, "kind": kind, "limit": limit, "status": "pending", "reserve": need if side == "buy" else 0.0,
         "session": next_session(now, trading_days).isoformat(), "intraday": is_session(now, trading_days)}
    a["next_id"] += 1
    a["orders"].append(o)
    if o["intraday"] and quote:
        if side == "buy" and quote["asks"] and (kind == "market" or limit >= quote["asks"][0]):
            _fill(a, o, quote["asks"][0], now.isoformat(timespec="seconds"))
        elif side == "sell" and quote["bids"] and (kind == "market" or limit <= quote["bids"][0]):
            _fill(a, o, quote["bids"][0], now.isoformat(timespec="seconds"))
    if o["status"] == "filled":
        o["reserve"] = 0.0
        return True, f"已成交：{'買進' if side == 'buy' else '賣出'} {name} {shares:,} 股 @ {o['fill_price']:g}"
    when = "今天收盤後依當日價格撮合" if o["intraday"] else f"{o['session']} 開盤撮合"
    return True, f"已送出委託（{when}，當日有效）"


def cancel(a: dict, oid: int) -> bool:
    for o in a["orders"]:
        if o["id"] == oid and o["status"] == "pending":
            o.update(status="cancelled", reserve=0.0)
            return True
    return False


def settle(a: dict, bars: dict, last_data_day: date) -> int:
    """用日行情撮合還沒成交的委託。bars＝{code: {date: (open, high, low, close, prev_close)}}。回傳處理筆數。"""
    n = 0
    for o in a["orders"]:
        if o["status"] != "pending":
            continue
        days = sorted(d for d in bars.get(o["code"], {}) if d >= date.fromisoformat(o["session"]))
        if not days:
            if date.fromisoformat(o["session"]) < last_data_day - timedelta(days=7):
                o.update(status="expired", reserve=0.0)                 # 停牌等：一直沒有行情
                n += 1
            continue
        d = days[0]
        op, hi, lo, cl, prev = bars[o["code"]][d]
        up, dn = limits(prev, is_etf(o["code"]))
        P, buy = o["limit"], o["side"] == "buy"
        price = None
        locked = (buy and lo >= up) or ((not buy) and hi <= dn)          # 整天鎖漲停（買不到）／鎖跌停（賣不掉）
        if not locked:
            if not o["intraday"] and (o["kind"] == "market" or (op <= P if buy else op >= P)):
                price = op                                              # 盤前的單：開盤就成交
            elif o["kind"] == "limit" and (lo <= P if buy else hi >= P):
                price = P
        if price is None:
            o.update(status="expired", reserve=0.0, note=f"{d} 未成交")
        else:
            o["reserve"] = 0.0
            _fill(a, o, price, f"{d.isoformat()}T13:30:00")
        n += 1
    return n
