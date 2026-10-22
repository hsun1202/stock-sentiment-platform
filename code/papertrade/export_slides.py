# -*- coding: utf-8 -*-
"""把紙上模擬的最新結果匯出給簡報用：ppt_build/assets/papertrade_nav.png（淨值圖，簡報配色）、papertrade_summary.json（數字）。
run_daily.py 每次跑完都會呼叫；報告前重建簡報（node build_deck_vN.js）就會帶入最新結果。"""
import json
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import broker  # noqa: E402
from strategies import STRATEGIES  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(BASE), "ppt_build", "assets")
COLORS = {"highvol30": "#C15B2A", "lowvol30": "#3E6B4F", "ew250": "#9C9184", "etf0050": "#26221D"}
SHORT = {"highvol30": "高波動 30 檔", "lowvol30": "低波動 30 檔", "ew250": "一般股票（前 250 名等權）", "etf0050": "0050（大盤）"}
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def export():
    os.makedirs(ASSETS, exist_ok=True)
    navs = {}
    for k in STRATEGIES:
        p = os.path.join(broker.STATE, f"{k}_nav.csv")
        if os.path.exists(p):
            navs[k] = pd.read_csv(p).drop_duplicates("date", keep="last").set_index("date").nav / broker.CAPITAL
    rows, asof, start = [], None, None
    ref = navs.get("etf0050")
    for k, n in navs.items():
        st = json.load(open(os.path.join(broker.STATE, f"{k}.json"), encoding="utf-8"))
        start, asof = st["start"], n.index[-1]
        ret = n.iloc[-1] - 1
        rows.append({"key": k, "name": SHORT[k], "ret": ret, "mdd": float((n / n.cummax() - 1).min()),
                     "vs0050": (ret - (ref.iloc[-1] - 1)) if (ref is not None and k != "etf0050") else None, "n_pos": len(st["pos"])})
    n_days = max((len(n) for n in navs.values()), default=0) - 1      # 開帳日之後的交易日數
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    if n_days >= 2:
        for k, n in navs.items():
            ax.plot(pd.to_datetime(n.index), (n - 1) * 100, color=COLORS[k], lw=2.4 if k in ("highvol30", "etf0050") else 1.6,
                    ls="--" if k == "etf0050" else "-", label=SHORT[k])
        ax.axhline(0, color="#C9BBA6", lw=0.8)
        ax.set_ylabel("累積報酬（%）", color="#6B6157")
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")
        fig.autofmt_xdate()
    else:
        ax.text(0.5, 0.5, f"{start} 開帳，隔天開盤建倉\n之後每個交易日自動累積", ha="center", va="center", fontsize=13, color="#6B6157", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors="#6B6157", labelsize=8.5)
    ax.grid(axis="y", color="#E4DACB", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "papertrade_nav.png"), dpi=200, facecolor="white")
    plt.close(fig)
    out = {"start": start, "asof": asof, "n_days": n_days, "rows": rows}
    json.dump(out, open(os.path.join(ASSETS, "papertrade_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    print(export())
