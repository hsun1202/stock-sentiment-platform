"""研究結果：財報驚喜策略的淨值（2018–2025 回測＋2026 樣本外）與所有事先登記檢定的結果總表。"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common

common.page_title("研究結果", "所有策略都先寫好假說（PREREG）再跑；通過與沒通過的都列出來")
A = common.ROOT / "ai_strategies"
bt = pd.read_csv(A / "surprise_event_returns.csv", index_col=0)
oo = pd.read_csv(A / "oos2026_surprise_returns.csv", index_col=0)
bt.index, oo.index = pd.PeriodIndex(bt.index, freq="M"), pd.PeriodIndex(oo.index, freq="M")
oo = oo[oo.index <= pd.Period("2026-09", "M")]
cols = {"rev_top10": "月營收驚喜", "ern_top10": "盈餘驚喜", "ew": "一般股票（等權）", "etf0050": "0050 含息"}
M = pd.concat([bt[list(cols)], oo[list(cols)]]).rename(columns=cols)


def stats(r):
    eq = (1 + r).cumprod()
    beta = np.cov(r, M["0050 含息"].loc[r.index])[0, 1] / M["0050 含息"].loc[r.index].var()
    return (1 + r).prod() ** (12 / len(r)) - 1, r.mean() * 12 / (r.std() * np.sqrt(12)), (eq / eq.cummax() - 1).min(), beta


common.kpis([(f"{nm}", common.signed_html(stats(M[nm])[0]), f"Sharpe {stats(M[nm])[1]:.2f}・最大回落 {stats(M[nm])[2]:.0%}・β {stats(M[nm])[3]:.2f}",
              "accent" if "驚喜" in nm else "") for nm in cols.values()])
fig = go.Figure()
colors = {"月營收驚喜": common.UP, "盈餘驚喜": common.C_ALT, "一般股票（等權）": common.C_POOL, "0050 含息": common.C_ETF}
for nm in cols.values():
    eq = (1 + M[nm]).cumprod()
    fig.add_trace(go.Scatter(x=eq.index.to_timestamp(), y=eq, name=nm, line=dict(color=colors[nm], width=2 if "驚喜" in nm else 1.4)))
fig.add_vrect(x0="2026-01-01", x1="2026-09-30", fillcolor="rgba(229,72,77,.07)", line_width=0)
fig.add_annotation(x="2026-05-15", y=0.02, yref="paper", text="2026 樣本外", showarrow=False)
fig.update_yaxes(type="log")
common.chart(common.fig_layout(fig, "累積淨值（2018-01～2026-09，扣成本，對數座標；2026 為沒看過的資料）", 440))
st.caption("2018–2025 用 TEJ 股價回測（ai_strategies/surprise_event_backtest.py）；2026 用證交所公開資料（oos2026_surprise.py）。"
           "策略扣手續費 0.0855%×2＋證交稅 0.3%；一般股票與 0050 不扣成本。")

st.markdown("#### 事先登記的檢定總表")
rows = [
    ("情緒（PTT、CMoney）預測隔天漲跌、選股", "多組", "❌ 能描述、不能預測；CMoney 統計上測得到一點，但賺不到錢"),
    ("機器學習月頻選股 vs 一般股票", "M1", "✅ +0.58%／月（t=2.93）"),
    ("機器學習月頻選股 vs 0050", "M2", "❌ 打平"),
    ("情緒過濾（剔除最悲觀）", "M3", "❌"),
    ("財報水準加進模型", "F1", "❌ IC 變好，報酬沒有顯著增加"),
    ("法人籌碼加進模型／投信買超前 30", "I1／I2", "❌ 投信買超還輸一般股票"),
    ("市值前 50 大＋AI 調權重 vs 0050", "T1／T2", "❌ 打平；風險較小"),
    ("**月營收驚喜**（公布隔天買、持有 20 天）", "U1", "✅ +1.26%／月（t=4.81），8 年都贏"),
    ("**盈餘驚喜**（公布隔天買、持有 60 天）", "U2", "✅ +1.26%／月（t=3.75），8 年都贏"),
    ("**2026 樣本外**：月營收驚喜／盈餘驚喜", "O1／O2", "✅ +2.70%／月（t=2.09）／+1.66%（t=3.86）；都輸 0050"),
    ("對 0050 的 α（事後檢查）", "I 節", "月營收 α +15.8%／年（t=2.69）、β 0.65；盈餘 α +17.5%（t=2.51）、β 0.56"),
    ("雙重驚喜（營收＋盈餘都最好）比只有盈餘驚喜好", "D1", "❌ +2.58%（t=1.16）；台股季營收早已由月營收公布"),
]
st.markdown("| 檢定 | 代號 | 結果 |\n|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows))
st.caption("新因子的 t 值要超過 3.0 才可信（Harvey, Liu & Zhu 2016）；兩個財報驚喜策略在 2018–2025 都超過。詳細數字見 ai_strategies/PREREG_ai_strategies.md。")
