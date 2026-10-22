# -*- coding: utf-8 -*-
"""產生 report/index.html 與 report/nav.png：各策略淨值、報酬、最大回落、與 0050 的差、目前持股、最近成交。"""
import html
import json
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import broker  # noqa: E402
from broker import CAPITAL  # noqa: E402
from strategies import STRATEGIES  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "report")


def set_out(d):
    global OUT
    OUT = d
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def build(names_map=None):
    os.makedirs(OUT, exist_ok=True)
    navs, rows = {}, []
    for k, (desc, _, freq) in STRATEGIES.items():
        p = os.path.join(broker.STATE, f"{k}_nav.csv")
        if not os.path.exists(p):
            continue
        n = pd.read_csv(p).drop_duplicates("date", keep="last").set_index("date").nav
        navs[k] = n
    if not navs:
        return None
    ref = navs.get("etf0050")
    for k, n in navs.items():
        st = json.load(open(os.path.join(broker.STATE, f"{k}.json"), encoding="utf-8"))
        ret = n.iloc[-1] / CAPITAL - 1
        dd = (n / n.cummax() - 1).min()
        vs = (ret - (ref.iloc[-1] / CAPITAL - 1)) if ref is not None else float("nan")
        rows.append({"策略": STRATEGIES[k][0], "開始": st["start"], "最新": n.index[-1], "淨值": f"{n.iloc[-1]:,.0f}",
                     "累積報酬": f"{ret:+.2%}", "相對 0050": f"{vs:+.2%}" if k != "etf0050" else "—",
                     "最大回落": f"{dd:.2%}", "持股數": len(st["pos"]), "現金": f"{st['cash']:,.0f}",
                     "待成交": ("有（訊號日 " + st["pending"]["signal_day"] + "）") if st.get("pending") else "無"})
    # 圖
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for k, n in navs.items():
        ax.plot(pd.to_datetime(n.index), n / CAPITAL, label=STRATEGIES[k][0], lw=2.2 if k in ("highvol30", "etf0050") else 1.3)
    ax.axhline(1, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("淨值（起點 = 1）")
    ax.set_title("紙上模擬：各策略淨值")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "nav.png"), dpi=130)
    plt.close(fig)
    # 持股與成交
    hold_html, trade_html = [], []
    names = names_map or {}
    for k in navs:
        st = json.load(open(os.path.join(broker.STATE, f"{k}.json"), encoding="utf-8"))
        pos = sorted(st["pos"].items())
        hold_html.append(f"<h3>{html.escape(STRATEGIES[k][0])}（{len(pos)} 檔）</h3><p class='codes'>" +
                         "、".join(f"{c} {html.escape(names.get(c, ''))}（{sh:,}股）" for c, sh in pos) + "</p>")
        tp = os.path.join(broker.STATE, f"{k}_trades.csv")
        if os.path.exists(tp):
            t = pd.read_csv(tp, dtype={"code": str}).tail(12)
            t.insert(2, "名稱", t.code.map(lambda c: names.get(c, "")))
            trade_html.append(f"<h3>{html.escape(STRATEGIES[k][0])}</h3>" + t.to_html(index=False, border=0))
    css = ("body{font-family:'Microsoft JhengHei',sans-serif;margin:24px;max-width:1100px;color:#222;background:#fff}"
           "table{border-collapse:collapse;font-size:14px;margin:8px 0 20px}th,td{padding:5px 10px;border-bottom:1px solid #ddd;text-align:right}"
           "th{background:#f3efe8}td:first-child,th:first-child{text-align:left}.codes{font-size:13px;line-height:1.7}"
           ".note{color:#666;font-size:13px}img{max-width:100%}")
    doc = (f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>紙上模擬交易</title><style>{css}</style></head><body>"
           f"<h1>紙上模擬交易</h1><p class='note'>虛擬本金每個策略 {CAPITAL:,} 元；訊號日收盤出訊號、隔天開盤成交；含手續費（六折）與證交稅；"
           f"除權息以（除權息前收盤價 − 參考價）× 股數計入現金。不動真錢、不下單。</p>"
           f"{pd.DataFrame(rows).to_html(index=False, border=0)}<img src='nav.png' alt='各策略淨值走勢'>"
           f"<h2>目前持股</h2>{''.join(hold_html)}<h2>最近成交</h2>{''.join(trade_html)}</body></html>")
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(doc)
    return pd.DataFrame(rows)
