# -*- coding: utf-8 -*-
"""策略：給「到某天收盤為止」的資料，回傳目標權重 {代號: 權重}。新增策略只要在 STRATEGIES 加一筆。"""
import numpy as np
import pandas as pd


def returns_panel(prices, exr):
    """收盤價面板與日報酬；除權息日用參考價當基準（不把配息當成下跌）。"""
    close = prices.pivot(index="date", columns="code", values="close").sort_index()
    value = prices.pivot(index="date", columns="code", values="value").sort_index()
    base = close.shift(1).ffill()
    if len(exr):
        ref = exr.pivot_table(index="date", columns="code", values="ref_price", aggfunc="last")
        ref = ref.reindex(index=close.index, columns=close.columns)
        base = ref.where(ref.notna(), base)
    ret = close / base - 1
    return close, value, ret


def universe(value, asof, n=250, win=60, min_obs=40):
    """上市普通股（四碼、非 0 開頭）中，近 win 日平均成交金額前 n 名。"""
    v = value.loc[:asof].tail(win)
    cols = [c for c in v.columns if len(c) == 4 and c.isdigit() and c[0] != "0"]
    v = v[cols]
    avg = v.mean().where(v.notna().sum() >= min_obs).dropna()
    return list(avg.nlargest(n).index)


def vol_rank(ret, codes, asof, win=60, min_obs=40):
    r = ret.loc[:asof, codes].tail(win)
    return r.std().where(r.notna().sum() >= min_obs).dropna()


def highvol30(ctx, asof):
    vol = vol_rank(ctx["ret"], universe(ctx["value"], asof), asof)
    top = vol.nlargest(30).index
    return {c: 1 / len(top) for c in top}


def lowvol30(ctx, asof):
    vol = vol_rank(ctx["ret"], universe(ctx["value"], asof), asof)
    top = vol.nsmallest(30).index
    return {c: 1 / len(top) for c in top}


def ew250(ctx, asof):
    u = universe(ctx["value"], asof)
    return {c: 1 / len(u) for c in u}


def etf0050(ctx, asof):
    return {"0050": 1.0}


# 代號：(說明, 函式, 換股頻率)；頻率 "Q"＝每季第一個交易日出訊號、隔天開盤成交；"once"＝只在開始時建倉
STRATEGIES = {
    "highvol30": ("高波動 30 檔（前 250 名，季換）", highvol30, "Q"),
    "lowvol30": ("低波動 30 檔（對照組，季換）", lowvol30, "Q"),
    "ew250": ("前 250 名等權（一般股票基準，季換）", ew250, "Q"),
    "etf0050": ("0050 買進持有（大盤基準）", etf0050, "once"),
}
