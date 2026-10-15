# -*- coding: utf-8 -*-
"""
CMoney 同學會 vs PTT：今天收盤前的淨情緒 → 隔天報酬（2020-2024，同一套方法，與 ptt_cc_deepdive.py 相同的變數定義）
  來源：p20＝PTT(Breeze)；cm＝CMoney×微調 BERT；cmbb＝CMoney×作者自標多空
  報酬用超額（個股 − 當天全體平均）；迴歸依交易日群聚標準誤；控制：今天超額、|今天超額|、漲停、跌停、量能、文章量、規模
  另：收盤買「淨情緒>0」隔天開盤/收盤賣的每筆報酬與成本；逐年係數；兩個平台淨情緒的相關
輸出：cmoney_daily_test.log
python cmoney_daily_test.py
"""
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
LOG = []
OUT = "cmoney_daily_test.log"


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


COLS = ["stock_code", "Date", "Open", "Close", "Volume", "close_unadj", "abnormal_price",
        "ptt_pre_pos", "ptt_pre_neg", "ptt_pre_n", "ptt_intra_pos", "ptt_intra_neg", "ptt_intra_n"]
CTRL = ["x_r_today", "x_abs_today", "lim_up", "lim_dn", "volr", "att", "size_r"]


def load(tag):
    df = pd.read_csv(f"final_dataset_v3_{tag}/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"], usecols=COLS)
    df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
    g = df.groupby("stock_code")
    for c in COLS[7:]:
        df[c] = df[c].fillna(0)
    pos, neg = df.ptt_pre_pos + df.ptt_intra_pos, df.ptt_pre_neg + df.ptt_intra_neg
    df["n_expr"] = pos + neg
    df["att"] = np.log1p(df.ptt_pre_n + df.ptt_intra_n)
    df["net"] = (pos - neg) / df.n_expr.replace(0, np.nan)
    bad = (df.abnormal_price.fillna(0) != 0) | g.abnormal_price.shift(1).fillna(0).ne(0)
    C = df.Close
    okn = g.abnormal_price.shift(-1).fillna(0).eq(0)
    df["r_today"] = (C / g.Close.shift(1) - 1).where(~bad)
    df["r1"] = (g.Close.shift(-1) / C - 1).where(okn)
    df["r_on"] = (g.Open.shift(-1) / C - 1).where(okn)
    df["r_id"] = (g.Close.shift(-1) / g.Open.shift(-1) - 1).where(okn)
    for k in range(2, 6):
        df[f"r{k}"] = (g.Close.shift(-k) / g.Close.shift(-k + 1) - 1).where(g.abnormal_price.shift(-k).fillna(0).eq(0))
    limit = np.where(df.Date < "2015-06-01", 0.07, 0.10)
    df["lim_up"] = (df.r_today >= limit - 0.005).astype(int)
    df["lim_dn"] = (df.r_today <= -limit + 0.005).astype(int)
    vol20 = g.Volume.transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["volr"] = np.log((df.Volume + 1) / (vol20 + 1))
    dv = df.close_unadj * df.Volume.where(df.Volume > 0)
    df["size"] = dv.groupby(df.stock_code).transform(lambda s: s.rolling(60, min_periods=20).median())
    df["size_r"] = df.groupby("Date")["size"].rank(pct=True)
    for col in ["r_today", "r1", "r_on", "r_id", "r2", "r3", "r4", "r5"]:
        df["x_" + col] = df[col] - df.groupby("Date")[col].transform("mean")
    df["x_abs_today"] = df.x_r_today.abs()
    df["x_r2_5"] = df[[f"x_r{k}" for k in range(2, 6)]].sum(axis=1, min_count=4)
    return df[(df.Date >= "2020-01-01") & (df.Date <= "2024-12-31")]


def reg(d, ycol, xcols):
    d = d.dropna(subset=[ycol] + xcols)
    if len(d) < 200:
        return None
    r = sm.OLS(d[ycol] * 1e4, sm.add_constant(d[xcols])).fit(cov_type="cluster", cov_kwds={"groups": d.Date.dt.strftime("%Y%m%d")})
    return r.params["net"], r.tvalues["net"], len(d)


def fmt(x):
    return "  —  " if x is None else f"{x[0]:+6.2f} bp（t={x[1]:+.2f}，n={x[2]:,}）"


NAMES = {"p20": "PTT（Breeze）", "cm": "CMoney × 微調BERT", "cmbb": "CMoney × 作者自標多空"}
if len(sys.argv) > 1 and sys.argv[1] == "v2":   # 清理後的版本
    NAMES = {"p20": "PTT（Breeze）", "cm": "CMoney 未清理 × 舊模型（只用PTT標註）", "cmc": "CMoney 清理後 × 舊模型", "cm2": "CMoney 清理後 × 新模型（PTT+CMoney標註）",
             "cmbb2": "CMoney 清理後 × 作者自標多空"}
    OUT = "cmoney_daily_test_v2.log"
if len(sys.argv) > 1 and sys.argv[1] == "tej":  # 股價改用 TEJ 的版本
    NAMES = {"tejp20": "PTT（Breeze）｜TEJ股價", "tejcmc": "CMoney 清理後 × 舊模型｜TEJ股價", "tejcm2": "CMoney 清理後 × 新模型｜TEJ股價",
             "tejcmbb2": "CMoney 清理後 × 作者自標多空｜TEJ股價"}
    OUT = "cmoney_daily_test_tej.log"
DATA = {}
for tag in NAMES:
    df = load(tag)
    S = df[(df.n_expr >= 1) & df.x_r_today.notna() & df.x_r1.notna()].copy()
    DATA[tag] = S
    say(f"\n===== {NAMES[tag]}｜2020-2024 =====")
    say(f"有表態的股票日 {len(S):,}（{S.stock_code.nunique()} 檔）；淨情緒>0 {int((S.net > 0).sum()):,}、<0 {int((S.net < 0).sum()):,}；每日表態則數中位數 {S.n_expr.median():.0f}")
    say(f"描述力：淨情緒與今天超額報酬相關 {S.net.corr(S.x_r_today):+.3f}")
    say(f"  隔天收盤→收盤｜只放淨情緒      {fmt(reg(S, 'x_r1', ['net']))}")
    say(f"  隔天收盤→收盤｜全部控制變數    {fmt(reg(S, 'x_r1', ['net'] + CTRL))}")
    say(f"  隔夜（收盤→開盤）｜全部控制    {fmt(reg(S, 'x_r_on', ['net'] + CTRL))}")
    say(f"  盤中（開盤→收盤）｜全部控制    {fmt(reg(S, 'x_r_id', ['net'] + CTRL))}")
    say(f"  第 2~5 天累積｜全部控制        {fmt(reg(S, 'x_r2_5', ['net'] + CTRL))}")
    say(f"  表態≥5 則｜隔夜｜全部控制      {fmt(reg(S[S.n_expr >= 5], 'x_r_on', ['net'] + CTRL))}")
    say(f"  排除漲跌停日｜隔夜             {fmt(reg(S[(S.lim_up == 0) & (S.lim_dn == 0)], 'x_r_on', ['net', 'x_r_today', 'x_abs_today', 'volr', 'att', 'size_r']))}")
    say("  逐年隔夜係數（全部控制）：" + "｜".join(f"{y}: " + ("—" if (x := reg(S[S.Date.dt.year == y], 'x_r_on', ['net'] + CTRL)) is None else f"{x[0]:+.1f}(t={x[1]:+.1f})") for y in range(2020, 2025)))
    for lab, m in [("淨情緒>0", S.net > 0), ("淨情緒>0 且今天非漲停", (S.net > 0) & (S.lim_up == 0)), ("淨情緒<0", S.net < 0)]:
        d = S[m]
        say(f"  收盤買[{lab}] {len(d):,} 筆｜隔天開盤賣：原始 {d.r_on.mean() * 1e4:+.1f} bp、超額 {d.x_r_on.mean() * 1e4:+.1f}｜隔天收盤賣：原始 {d.r1.mean() * 1e4:+.1f}、超額 {d.x_r1.mean() * 1e4:+.1f}"
            f"｜（成本 47~59 bp）")

CM = next(k for k in ("tejcm2", "cm2", "cm") if k in DATA)
BB = next(k for k in ("tejcmbb2", "cmbb2", "cmbb") if k in DATA)
P20 = "tejp20" if "tejp20" in DATA else "p20"
m = DATA[P20][["stock_code", "Date", "net"]].merge(DATA[CM][["stock_code", "Date", "net"]], on=["stock_code", "Date"], suffixes=("_ptt", "_cm"))
say("")
say(f"兩個平台同一股票日都有表態的 {len(m):,} 個：淨情緒相關 {m.net_ptt.corr(m.net_cm):+.3f}、同號 {(np.sign(m.net_ptt) == np.sign(m.net_cm)).mean():.0%}")
m2 = DATA[CM][["stock_code", "Date", "net"]].merge(DATA[BB][["stock_code", "Date", "net"]], on=["stock_code", "Date"], suffixes=("_bert", "_bb"))
say(f"CMoney 內：模型淨情緒與作者自標淨情緒（{len(m2):,} 個股票日）相關 {m2.net_bert.corr(m2.net_bb):+.3f}")
open(OUT, "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
