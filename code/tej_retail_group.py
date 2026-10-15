# -*- coding: utf-8 -*-
"""
TEJ 新檢驗 (2)：依「散戶持股比例」分組，看情緒（PTT、CMoney）與融資融券的效果在散戶多的股票是不是比較強。
資料：tej_raw/tej_股權分散_年.csv（年資料）；股票日資料 final_dataset_v3_tej（PTT、融資融券）、final_dataset_v3_tejcm2（CMoney 清理後×新模型）。
散戶持股比例＝持股未滿 400 張的股東合計持股比率（＝100 − 400 張以上各級距比率）；Y 年一律用「年度 Y−1」那筆（多為 Y−1 年 3~4 月的資料），
每年在樣本股票內分三組（低/中/高）。

事先寫定的假說（2026-10-05，跑之前寫下；主要檢定 3 個，Bonferroni 門檻 p<0.0167；預期方向都是「散戶多的組效果較強」）：
  G1（PTT，2015~2024）：隔天收盤→收盤超額報酬 ~ 淨情緒 + 控制變數（同 ptt_cc_deepdive.py），高組的淨情緒係數 > 低組
      （只取高、低兩組，檢定 淨情緒×高組 的交互項，依交易日群聚）。
  G2（CMoney，2020~2024）：同 G1。
  G3（融資融券，2018~2024）：券資比與隔天開盤→收盤報酬的每日橫斷面 Rank IC（全樣本為負），高組比低組更負（逐日配對 t）。
  探索性：隔夜/盤中拆解、各組個別係數、其他門檻（未滿 50 張、未滿 1000 張）、融資增減與融券增減、與規模的重疊。
python tej_retail_group.py
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
ALPHA = 0.05 / 3
LOG = []
GN = {0: "低", 1: "中", 2: "高"}


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


# ---------------- 散戶持股比例（年）----------------
h = pd.read_csv("tej_raw/tej_股權分散_年.csv", encoding="cp950", dtype=str)
h.columns = [c.replace(" ", "") for c in h.columns]
ratio = [c for c in h.columns if c.endswith("(比率)") and not c.startswith("集保")]
for c in ratio:
    h[c] = pd.to_numeric(h[c].str.replace(",", ""), errors="coerce")
BIG400 = ["400-600張(比率)", "600-800張(比率)", "800-1000張(比率)", "1000張以上(比率)"]
SMALL50 = ["1張以下(比率)", "1-5張(比率)", "5-10張(比率)", "10-15張(比率)", "15-20張(比率)", "20-30張(比率)", "30-50張(比率)"]
assert all(c in h.columns for c in BIG400 + SMALL50), ratio
tot = h[ratio].sum(axis=1)
h = h[(tot > 99) & (tot < 101)].copy()
h["rh400"] = 100 - h[BIG400].sum(axis=1)
h["rh50"] = h[SMALL50].sum(axis=1)
h["rh1000"] = 100 - h["1000張以上(比率)"]
h["year"] = h["年度"].astype(int) + 1
h = h.rename(columns={"代號": "stock_code"}).drop_duplicates(["stock_code", "year"], keep="last")[["stock_code", "year", "rh400", "rh50", "rh1000"]]
say(f"[股權分散] {len(h):,} 筆（級距比率合計≈100 才保留；原始合計中位數 {tot.median():.1f}）；未滿400張持股比率中位數 {h.rh400.median():.1f}%")

COLS = ["stock_code", "Date", "Open", "Close", "Volume", "close_unadj", "abnormal_price", "mktcap", "margin_eligible",
        "margin_chg", "short_chg", "short_margin_ratio",
        "ptt_pre_pos", "ptt_pre_neg", "ptt_pre_n", "ptt_intra_pos", "ptt_intra_neg", "ptt_intra_n"]
CTRL = ["x_r_today", "x_abs_today", "lim_up", "lim_dn", "volr", "att", "size_r"]


def load(tag):
    df = pd.read_csv(f"final_dataset_v3_{tag}/all_stocks_v3.csv", dtype={"stock_code": str}, parse_dates=["Date"], usecols=COLS)
    df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
    g = df.groupby("stock_code")
    for c in COLS[12:]:
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
    limit = np.where(df.Date < "2015-06-01", 0.07, 0.10)
    df["lim_up"] = (df.r_today >= limit - 0.005).astype(int)
    df["lim_dn"] = (df.r_today <= -limit + 0.005).astype(int)
    vol20 = g.Volume.transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["volr"] = np.log((df.Volume + 1) / (vol20 + 1))
    dv = df.close_unadj * df.Volume.where(df.Volume > 0)
    df["size"] = dv.groupby(df.stock_code).transform(lambda s: s.rolling(60, min_periods=20).median())
    df["size_r"] = df.groupby("Date")["size"].rank(pct=True)
    for col in ["r_today", "r1", "r_on", "r_id"]:
        df["x_" + col] = df[col] - df.groupby("Date")[col].transform("mean")
    df["x_abs_today"] = df.x_r_today.abs()
    df["year"] = df.Date.dt.year
    df = df.merge(h, on=["stock_code", "year"], how="left")
    yr = df.drop_duplicates(["stock_code", "year"])[["stock_code", "year", "rh400", "rh50", "rh1000"]].dropna()
    for c in ("rh400", "rh50", "rh1000"):
        yr["g_" + c] = yr.groupby("year")[c].transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
    return df.merge(yr[["stock_code", "year", "g_rh400", "g_rh50", "g_rh1000"]], on=["stock_code", "year"], how="left")


def reg(d, ycol, xcols, key="net"):
    d = d.dropna(subset=[ycol] + xcols)
    if len(d) < 200:
        return None
    r = sm.OLS(d[ycol] * 1e4, sm.add_constant(d[xcols])).fit(cov_type="cluster", cov_kwds={"groups": d.Date.dt.strftime("%Y%m%d")})
    return r.params[key], r.tvalues[key], r.pvalues[key], len(d)


def fmt(x):
    return "—" if x is None else f"{x[0]:+.2f}（t={x[1]:+.2f}，n={x[3]:,}）"


def sentiment(tag, name, a, b, label):
    df = load(tag)
    df = df[(df.Date >= a) & (df.Date <= b)]
    S = df[(df.n_expr >= 1) & df.x_r_today.notna() & df.x_r1.notna() & df.g_rh400.notna()].copy()
    yr = df.drop_duplicates(["stock_code", "year"])
    say(f"\n===== {label}：{name}｜{a[:4]}~{b[:4]} =====")
    say(f"  有表態且有分組的股票日 {len(S):,}；各組（低/中/高）筆數 " + "/".join(f"{int((S.g_rh400 == k).sum()):,}" for k in range(3))
        + f"；散戶持股比例與規模排名相關 {yr.rh400.corr(yr.size_r):+.2f}")
    for y, yn in [("x_r1", "隔天收→收"), ("x_r_on", "隔夜"), ("x_r_id", "盤中")]:
        say(f"  {yn:<6s}（bp，全部控制）｜全部 {fmt(reg(S, y, ['net'] + CTRL))}｜"
            + "｜".join(f"{GN[k]}組 {fmt(reg(S[S.g_rh400 == k], y, ['net'] + CTRL))}" for k in range(3)))
    out = None
    for gc, gname in [("g_rh400", "未滿400張★"), ("g_rh50", "未滿50張"), ("g_rh1000", "未滿1000張")]:
        d = S[S[gc].isin([0, 2])].copy()
        d["hi"] = (d[gc] == 2).astype(int)
        d["net_hi"] = d.net * d.hi
        for y, yn in [("x_r1", "隔天收→收"), ("x_r_on", "隔夜")]:
            x = reg(d, y, ["net", "net_hi", "hi"] + CTRL, key="net_hi")
            say(f"  交互項 淨情緒×高組［{gname}］{yn}：{x[0]:+.2f} bp（t={x[1]:+.2f}，p={x[2]:.4f}）")
            if gc == "g_rh400" and y == "x_r1":
                out = x
    say(f"  → {label}：高組−低組 {out[0]:+.2f} bp，t={out[1]:+.2f}，p={out[2]:.4f}（門檻 {ALPHA:.4f}）→ {'通過' if (out[2] < ALPHA and out[0] > 0) else '未通過'}")
    return df


D = sentiment("tej", "PTT（Breeze）", "2015-01-01", "2024-12-31", "G1")
sentiment("tejcm2", "CMoney 清理後×新模型", "2020-01-01", "2024-12-31", "G2")

# ---------------- G3 融資融券 ----------------
say("\n===== G3：融資融券｜2018~2024｜每日橫斷面 Rank IC（與隔天開盤→收盤報酬），依散戶持股比例分組 =====")
M = D[(D.Date >= "2018-01-01") & (D.margin_eligible == 1) & D.r_id.notna() & D.g_rh400.notna()]


def daily_ic(d, col):
    d = d[d[col].notna()]
    d = d.assign(a=d.groupby("Date")[col].rank(), b=d.groupby("Date").r_id.rank())
    n = d.groupby("Date").size()
    ic = d.groupby("Date").apply(lambda z: z.a.corr(z.b))
    return ic[n >= 20]


def tp(s):
    s = pd.Series(s).dropna()
    t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
    return t, 2 * stats.t.sf(abs(t), len(s) - 1)


g3 = None
for col, name in [("short_margin_ratio", "券資比★"), ("margin_chg", "融資增減"), ("short_chg", "融券增減")]:
    ics = {k: daily_ic(M[M.g_rh400 == k], col) for k in range(3)}
    diff = (ics[2] - ics[0]).dropna()
    t, p = tp(diff)
    say(f"  {name:<6s} 全部 {daily_ic(M, col).mean():+.4f}｜" + "｜".join(f"{GN[k]}組 {ics[k].mean():+.4f}（t={tp(ics[k])[0]:+.2f}）" for k in range(3))
        + f"｜高−低 {diff.mean():+.4f}（t={t:+.2f}，p={p:.4f}，{len(diff)} 天）")
    if col == "short_margin_ratio":
        g3 = (diff.mean(), t, p)
say("  每日各組可融資融券檔數中位數：" + "/".join(f"{int(M[M.g_rh400 == k].groupby('Date').size().median())}" for k in range(3)))
say(f"  → G3：高組−低組 IC {g3[0]:+.4f}，t={g3[1]:+.2f}，p={g3[2]:.4f}（門檻 {ALPHA:.4f}）→ {'通過' if (g3[2] < ALPHA and g3[0] < 0) else '未通過'}")
open("tej_retail_group.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
say("DONE")
