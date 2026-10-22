# -*- coding: utf-8 -*-
"""模擬券商：每個策略一份帳戶（現金、持股、待成交委託），逐個交易日處理：開盤成交 → 除權息 → 收盤記淨值 → 必要時出新訊號。"""
import json
import math
import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(BASE, "state")
CAPITAL = 10_000_000          # 虛擬本金；1,000 萬才不會因為高價股（一股數千元）買不到整數股而閒置現金
FEE = 0.001425 * 0.6          # 手續費（六折），買賣都收
TAX_STOCK, TAX_ETF = 0.003, 0.001
LIMIT = 0.095                 # 開盤漲跌幅超過這個比例，視為漲停買不到／跌停賣不掉（漲跌幅上限 10%）


def tax_rate(code):
    return TAX_ETF if code.startswith("00") else TAX_STOCK


class Account:
    def __init__(self, name):
        self.name = name
        self.path = os.path.join(STATE, f"{name}.json")
        os.makedirs(STATE, exist_ok=True)
        self.state_dir = STATE
        if os.path.exists(self.path):
            self.s = json.load(open(self.path, encoding="utf-8"))
        else:
            self.s = {"cash": CAPITAL, "pos": {}, "pending": None, "last_day": None, "start": None, "last_signal": None}

    def save(self):
        json.dump(self.s, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def _append(self, fname, rows):
        if not rows:
            return
        p = os.path.join(self.state_dir, f"{self.name}_{fname}.csv")
        pd.DataFrame(rows).to_csv(p, mode="a", header=not os.path.exists(p), index=False, encoding="utf-8-sig")

    # ---------- 開盤成交 ----------
    def execute(self, day, opn, base, last_px):
        p = self.s["pending"]
        if not p:
            return
        w = p["weights"]
        only = set(p.get("retry") or [])          # 前一天被漲跌停擋住、今天重試的代號
        px = lambda c: opn.get(c) if (opn.get(c) == opn.get(c) and opn.get(c)) else last_px.get(c)  # noqa: E731
        nav = self.s["cash"] + sum(sh * (px(c) or 0) for c, sh in self.s["pos"].items())
        target = {}
        for c in set(w) | set(self.s["pos"]):
            if only and c not in only:
                continue
            o = opn.get(c)
            target[c] = math.floor(nav * w.get(c, 0) * (1 - 0.002) / o) if (o and o == o and w.get(c, 0) > 0) else 0
        trades, blocked = [], []
        # 先賣
        for c, tgt in target.items():
            have = self.s["pos"].get(c, 0)
            if have > tgt:
                o, b = opn.get(c), base.get(c)
                if not (o and o == o) or (b and o / b - 1 <= -LIMIT):
                    blocked.append(c)
                    continue
                q = have - tgt
                amt = q * o
                cost = amt * (FEE + tax_rate(c))
                self.s["cash"] += amt - cost
                self.s["pos"][c] = tgt
                trades.append({"date": day, "code": c, "side": "賣", "shares": q, "price": o, "amount": round(amt), "cost": round(cost, 1)})
        # 再買（現金不夠就依序縮量）
        for c, tgt in sorted(target.items(), key=lambda kv: -w.get(kv[0], 0)):
            have = self.s["pos"].get(c, 0)
            if tgt > have:
                o, b = opn.get(c), base.get(c)
                if b and o / b - 1 >= LIMIT:
                    blocked.append(c)
                    continue
                q = min(tgt - have, math.floor(self.s["cash"] / (o * (1 + FEE))))
                if q <= 0:
                    continue
                amt = q * o
                cost = amt * FEE
                self.s["cash"] -= amt + cost
                self.s["pos"][c] = have + q
                trades.append({"date": day, "code": c, "side": "買", "shares": q, "price": o, "amount": round(amt), "cost": round(cost, 1)})
        self.s["pos"] = {c: sh for c, sh in self.s["pos"].items() if sh > 0}
        self.s["pending"] = {"weights": w, "signal_day": p["signal_day"], "retry": blocked} if blocked else None
        self._append("trades", trades)
        return trades, blocked

    # ---------- 除權息 ----------
    def dividends(self, day, exr_today):
        rows = []
        for c, sh in self.s["pos"].items():
            if c in exr_today:
                prev, ref = exr_today[c]
                if prev == prev and ref == ref and prev > ref:
                    cash = (prev - ref) * sh
                    self.s["cash"] += cash
                    rows.append({"date": day, "code": c, "side": "除權息", "shares": sh, "price": round(prev - ref, 4), "amount": round(cash), "cost": 0})
        self._append("trades", rows)

    # ---------- 收盤淨值 ----------
    def mark(self, day, last_px):
        mv = sum(sh * last_px.get(c, 0) for c, sh in self.s["pos"].items())
        nav = self.s["cash"] + mv
        self._append("nav", [{"date": day, "nav": round(nav, 1), "cash": round(self.s["cash"], 1), "n_pos": len(self.s["pos"])}])
        self.s["last_day"] = day
        return nav

    def signal(self, day, weights):
        self.s["pending"] = {"weights": weights, "signal_day": day, "retry": []}
        self.s["last_signal"] = day
