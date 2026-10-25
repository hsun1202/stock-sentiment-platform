"""
平台試作版的共用部分：樣式、資料狀態列、讀資料（快取）、圖表。
樣式（CSS、紅漲綠跌配色、指標卡、標籤）參考組員 jameschen108/tw-stock-quant 的 frontend/common.py；資料讀取是自己寫的，
只讀 platform_feed 的輸出（證交所行情、公開資訊觀測站營收／季報快照、兩份驚喜名單）與 TEJ 匯出的歷史。
"""
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
FEED = ROOT / "platform_feed"
DISCLAIMER = "學術研究用。名單依歷史統計規則產生（2018–2025 回測與 2026 樣本外贏一般股票，但沒有贏 0050），不保證未來報酬，不構成投資建議。"
UP, DOWN = "#e5484d", "#18a058"                    # 台股慣例：漲紅跌綠
C_MAIN, C_POOL, C_ETF, C_ALT = "#3b82f6", "#94a3b8", "#f59e0b", "#8b5cf6"

CSS = """
<style>
.block-container {padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1440px;}
div[data-testid="stMetricValue"] {font-size: 1.45rem;}
.sbar {display:flex; flex-wrap:wrap; gap:.35rem; align-items:center; font-size:.8rem; margin:.1rem 0 .2rem;}
.chip {padding:.12rem .6rem; border-radius:999px; background:rgba(128,128,128,.10); border:1px solid rgba(128,128,128,.22); white-space:nowrap;}
.chip b {font-weight:600;}
.chip.ok {border-color:rgba(24,160,88,.55);}
.chip.warn {background:rgba(234,179,8,.14); border-color:rgba(234,179,8,.55);}
.chip.bad {background:rgba(229,72,77,.14); border-color:rgba(229,72,77,.55);}
.disc {font-size:.76rem; opacity:.62; margin:.15rem 0 .6rem;}
.ptitle {display:flex; align-items:baseline; gap:.8rem; flex-wrap:wrap; margin:.4rem 0 .6rem;}
.ptitle h1 {font-size:1.9rem; margin:0; padding:0;}
.ptitle .sub {opacity:.65; font-size:.9rem;}
.kgrid {display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:.7rem; margin:.3rem 0 1rem;}
.kpi {border:1px solid rgba(128,128,128,.22); border-radius:.8rem; padding:.7rem .95rem; background:rgba(128,128,128,.05);}
.kpi .t {font-size:.78rem; opacity:.68;}
.kpi .v {font-size:1.5rem; font-weight:650; line-height:1.35; font-variant-numeric:tabular-nums;}
.kpi .s {font-size:.76rem; opacity:.62;}
.kpi.accent {border-color:rgba(229,72,77,.5); background:rgba(229,72,77,.06);}
.kpi.calm {border-color:rgba(24,160,88,.5); background:rgba(24,160,88,.06);}
.badge {display:inline-block; padding:.05rem .5rem; border-radius:.4rem; font-size:.78rem; font-weight:600; margin:0 .25rem .2rem 0; white-space:nowrap;}
.badge.bull {background:rgba(229,72,77,.15); color:#e5484d;}
.badge.bear {background:rgba(24,160,88,.15); color:#18a058;}
.badge.muted {background:rgba(128,128,128,.14);}
.up {color:#e5484d;} .down {color:#18a058;}
.hero {display:flex; justify-content:space-between; gap:1.5rem; flex-wrap:wrap; border:1px solid rgba(128,128,128,.22);
       border-radius:.9rem; padding:1rem 1.2rem; margin:.4rem 0 .9rem; background:rgba(128,128,128,.04);}
.hero .nm {font-size:1.5rem; font-weight:650;}
.hero .cd {opacity:.6; font-size:.95rem; margin-left:.4rem;}
.hero .px {font-size:2.1rem; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.2;}
.hero .chg {font-size:1.05rem; font-weight:600; margin-left:.5rem;}
.hero .meta {opacity:.6; font-size:.8rem;}
.hero .right {display:flex; flex-direction:column; gap:.35rem; align-items:flex-start; min-width:280px;}
.hero .row {font-size:.85rem;}
.hero .row .k {opacity:.6; display:inline-block; width:4.6rem;}
.range {position:relative; height:6px; border-radius:3px; background:rgba(128,128,128,.25); width:220px; margin-top:.3rem;}
.range i {position:absolute; top:-4px; width:3px; height:14px; background:currentColor; border-radius:2px;}
.empty {border:1px dashed rgba(128,128,128,.35); border-radius:.8rem; padding:1rem 1.2rem; opacity:.8; font-size:.9rem;}
</style>
"""


# ---------------------------------------------------------------- 版面

def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def signed_html(x, nd=1, scale=100.0, suffix="%") -> str:
    v = _num(x)
    if v is None:
        return "—"
    cls = "up" if v > 0 else ("down" if v < 0 else "")
    return f'<span class="{cls}">{v * scale:+.{nd}f}{suffix}</span>'


def badge(label: str) -> str:
    kind = "bear" if "壞消息" in label else ("bull" if "驚喜" in label else "muted")
    return f'<span class="badge {kind}">{esc(label)}</span>'


def page_title(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="ptitle"><h1>{esc(title)}</h1><span class="sub">{esc(sub)}</span></div>', unsafe_allow_html=True)


def kpis(items: list[tuple]) -> None:
    """items: (標題, 值 HTML, 小字, 樣式 ''/'accent'/'calm')"""
    cells = []
    for it in items:
        t, v, s = it[0], it[1], it[2] if len(it) > 2 else ""
        cls = f" {it[3]}" if len(it) > 3 and it[3] else ""
        cells.append(f'<div class="kpi{cls}"><div class="t">{esc(t)}</div><div class="v">{v}</div><div class="s">{esc(s)}</div></div>')
    st.markdown(f'<div class="kgrid">{"".join(cells)}</div>', unsafe_allow_html=True)


def empty(msg: str) -> None:
    st.markdown(f'<div class="empty">{esc(msg)}</div>', unsafe_allow_html=True)


def color_signed(v) -> str:
    v = _num(v)
    if v is None or v == 0:
        return ""
    return f"color: {UP}" if v > 0 else f"color: {DOWN}"


def last_run() -> tuple[str, bool]:
    """platform_feed/run.log 最後一次執行時間、是否成功（有 DONE 且沒有錯誤）"""
    p = FEED / "run.log"
    if not p.exists():
        return "—", False
    lines = p.read_text(encoding="utf-8").splitlines()
    done = [i for i, ln in enumerate(lines) if ln.endswith(" DONE")]
    if not done:
        return "—", False
    start = max([i for i in range(done[-1]) if "[行情]" in lines[i]] or [0])
    ok = not any("錯誤" in ln for ln in lines[start:done[-1]])
    return lines[done[-1]][5:16], ok


def header() -> None:
    rev, ern = signals("revenue_signals"), signals("earnings_signals")
    run_at, ok = last_run()
    chips = [f'<span class="chip"><b>行情</b> {esc(rev.get("asof", "—"))}</span>',
             f'<span class="chip"><b>名單更新</b> {esc(run_at)}</span>',
             f'<span class="chip"><b>營收門檻</b> {esc(rev.get("thresholds", {}).get("month", "—"))}</span>',
             f'<span class="chip"><b>季報門檻</b> {esc(ern.get("thresholds", {}).get("month", "—"))}</span>',
             '<span class="chip ok">每日 18:50 自動更新</span>' if ok else '<span class="chip bad">最近一次更新有問題</span>']
    mk = signals("sentiment_market")
    if mk.get("days"):
        chips.append(f'<span class="chip"><b>PTT 情緒</b> {esc(mk["days"][-1]["date"])}</span>')
    else:
        chips.append('<span class="chip warn">還沒有情緒資料</span>')
    st.markdown(f'<div class="sbar">{"".join(chips)}</div><div class="disc">{DISCLAIMER}</div>', unsafe_allow_html=True)


def goto_stock(code: str) -> None:
    st.session_state["goto_code"] = code
    st.switch_page("p2_stock.py")


# ---------------------------------------------------------------- 資料（依檔案修改時間快取）

def _sig(p: Path) -> int:
    return p.stat().st_mtime_ns if p.exists() else 0


@st.cache_data(show_spinner=False, max_entries=4)
def _signals(name: str, _sig: int) -> dict:
    p = FEED / "out" / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def signals(name: str) -> dict:
    return _signals(name, _sig(FEED / "out" / f"{name}.json"))


@st.cache_data(show_spinner="讀取行情…", max_entries=2)
def _prices(_sig: int) -> pd.DataFrame:
    px = pd.read_csv(FEED / "data" / "prices.csv", dtype={"code": str}, parse_dates=["date"])
    return px.drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"])


def prices() -> pd.DataFrame:
    return _prices(_sig(FEED / "data" / "prices.csv"))


def stock_names() -> pd.Series:
    px = prices()
    s = px[px.code.str.match(r"^[1-9]\d{3}$")].drop_duplicates("code", keep="last").set_index("code").name
    return s.sort_index()


@st.cache_data(show_spinner="讀取月營收…", max_entries=2)
def _revenue(_sig: int) -> pd.DataFrame:
    rv = pd.read_csv(ROOT / "tej_raw/rev/20261022103241.csv", encoding="utf-16", sep="\t", on_bad_lines="skip", low_memory=False,
                     na_values=["."], usecols=["證券代碼", "年月", "以合併為主單月營收(千元)"])
    rv["code"] = rv["證券代碼"].astype(str).str.split().str[0]
    rv = rv[rv.code.str.match(r"^[1-9]\d{3}$")]
    T = pd.DataFrame({"code": rv.code, "per": pd.PeriodIndex(pd.to_datetime(rv["年月"].astype(str) + "01", format="%Y%m%d"), freq="M"),
                      "R": rv["以合併為主單月營收(千元)"]})
    P = pd.read_csv(FEED / "data/revenue_public.csv", dtype={"code": str})
    P = pd.DataFrame({"code": P.code, "per": pd.PeriodIndex(P.per, freq="M"), "R": P.R})
    A = pd.concat([T, P[P.per > T.per.max()]], ignore_index=True).drop_duplicates(["code", "per"], keep="first")
    return A.dropna(subset=["R"]).sort_values(["code", "per"])


def revenue() -> pd.DataFrame:
    return _revenue(_sig(FEED / "data/revenue_public.csv"))


@st.cache_data(show_spinner="讀取季報…", max_entries=2)
def _earnings(_sig: int) -> pd.DataFrame:
    """單季歸屬母公司淨利：TEJ（單季）＋觀測站快照（年初累計 − 同年前幾季）"""
    fn = pd.read_csv(ROOT / "tej_raw/fin/20261022104526.csv", encoding="utf-16", sep="\t", low_memory=False, na_values=["."],
                     usecols=["證券代碼", "年月", "歸屬母公司淨利（損）"])
    fn["code"] = fn["證券代碼"].astype(str).str.split().str[0]
    fn = fn[fn.code.str.match(r"^[1-9]\d{3}$")]
    qend = pd.to_datetime(fn["年月"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)
    T = pd.DataFrame({"code": fn.code, "per": pd.PeriodIndex(qend, freq="Q"), "N": fn["歸屬母公司淨利（損）"]}).drop_duplicates(["code", "per"], keep="last")
    single = T.set_index(["code", "per"]).N.to_dict()
    P = pd.read_csv(FEED / "data/earnings_public.csv", dtype={"code": str})
    add = []
    for r in P.itertuples():
        per = pd.Period(year=int(r.year), quarter=int(r.season), freq="Q")
        if (r.code, per) in single:
            continue
        prev = [single.get((r.code, pd.Period(year=per.year, quarter=k, freq="Q")), np.nan) for k in range(1, per.quarter)]
        if np.all(np.isfinite(prev)):
            add.append({"code": r.code, "per": per, "N": r.ni_ytd - np.sum(prev)})
    return pd.concat([T, pd.DataFrame(add, columns=["code", "per", "N"])], ignore_index=True).dropna(subset=["N"]).sort_values(["code", "per"])


def earnings() -> pd.DataFrame:
    return _earnings(_sig(FEED / "data/earnings_public.csv"))


def since_entry(rows: list[dict]) -> list:
    """進場日開盤（未還原）到最新收盤的報酬；還沒進場的回傳 None"""
    px = prices()
    last = px.groupby("code").close.last()
    op = px.set_index(["code", "date"]).open
    out = []
    for r in rows:
        o = op.get((r["code"], pd.Timestamp(r["entry_date"])))
        c = last.get(r["code"])
        out.append(None if o is None or not o or c is None or pd.isna(o) else c / o - 1)
    return out


# ---------------------------------------------------------------- 圖表

def fig_layout(fig, title: str = "", height: int = 340, legend: bool = True):
    fig.update_layout(title=dict(text=title, font=dict(size=14)) if title else None, height=height,
                      margin=dict(l=8, r=8, t=40 if title else 10, b=8), showlegend=legend, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1) if legend else None)
    return fig


def chart(fig) -> None:
    st.plotly_chart(fig, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
