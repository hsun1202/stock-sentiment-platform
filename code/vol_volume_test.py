# -*- coding: utf-8 -*-
"""
預測目標換成「波動」與「成交量」：PTT 情緒／留言量對隔天個股波動、成交量有沒有增量？（文獻：Antweiler & Frank 2004 留言板對波動的預測力強於對報酬）
資料：final_dataset_v3_tej（265 檔、PTT=Breeze、TEJ 股價）＋ tej_raw/tej_chip_daily.pkl（散戶成交比重）；CMoney 用 final_dataset_v3_tejcm2（2020-24，只做迴歸）。
目標（t+1 日）：波動＝log(最高/最低)；成交量＝log(隔天成交量 / 到 t 日為止的 20 日均量)。
可用資訊：t 日收盤後的技術面與籌碼、到 t+1 開盤前的 PTT 留言（同 features_v4 的 ptt_open_*）。
基準：技術面（features_v4）＋波動與量能的歷史（過去 1~4 天振幅、5/20 日平均振幅、|報酬|、過去量能、市值、星期幾）——波動與量本來就高度持續，基準很強。
模型：XGBoost 迴歸（顯示卡），逐年滾動 2018~2024（切分同 retail_test_v4.py）；評估＝每日橫斷面 Rank IC 的平均，以及樣本外 R²。

事先寫定的假說（2026-10-05，跑之前寫下；主要檢定 2 個，Bonferroni 門檻 p<0.025；7 個測試年的 ΔIC 做 t 檢定，自由度 6）：
  V1：基準＋PTT 特徵 對隔天波動的 Rank IC 高於基準。
  V2：基準＋PTT 特徵 對隔天成交量的 Rank IC 高於基準。
  探索性：+融資融券、+散戶成交比重、+法人；逐日配對 t；只看有留言的股票日；
          迴歸（依交易日群聚）：隔天波動/成交量 ~ 留言量、異常留言量、淨情緒、|淨情緒| + 控制；CMoney 同一迴歸；依異常留言量分五組的隔天波動。
python vol_volume_test.py
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from scipy import stats

from features_v4 import build

warnings.filterwarnings("ignore")
PATH = "final_dataset_v3_tej/all_stocks_v3.csv"
SEEDS, TEST_YEARS, ALPHA = 3, range(2018, 2025), 0.05 / 2
LOG = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def tp(s):
    s = pd.Series(s).dropna()
    t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
    return t, 2 * stats.t.sf(abs(t), len(s) - 1)


def prep(path):
    """回傳 feat（含基準與各組特徵）、raw（含目標）、各組欄位"""
    feat, meta, A = build(path)
    raw = pd.read_csv(path, dtype={"stock_code": str}, parse_dates=["Date"],
                      usecols=["stock_code", "Date", "Open", "High", "Low", "Close", "Volume", "abnormal_price", "mktcap",
                               "ptt_pre_n", "ptt_intra_n", "ptt_pre_pos", "ptt_pre_neg", "ptt_intra_pos", "ptt_intra_neg"])
    raw = raw.sort_values(["stock_code", "Date"]).reset_index(drop=True)
    assert (raw.stock_code.values == meta.stock_code.values).all() and (raw.Date.values == meta.date.values).all()
    g = raw.groupby("stock_code")
    traded = raw.Volume > 0
    raw["rng"] = np.log(raw.High / raw.Low).where(traded & (raw.abnormal_price.fillna(0) == 0))
    vol20 = g.Volume.transform(lambda s: s.rolling(20, min_periods=10).mean())
    okn = (g.abnormal_price.shift(-1).fillna(0) == 0) & (g.Volume.shift(-1) > 0) & (raw.abnormal_price.fillna(0) == 0)
    raw["y_vol"] = g.rng.shift(-1).where(okn)
    raw["y_volu"] = np.log((g.Volume.shift(-1) + 1) / (vol20 + 1)).where(okn & vol20.notna())

    def roll(col, w, mp):
        return g[col].transform(lambda x: x.rolling(w, min_periods=mp).mean())

    HIST = []
    for k in (1, 2, 3, 4):
        feat[f"rng_l{k}"] = g.rng.shift(k).values
        feat[f"volr_l{k}"] = feat.volr.groupby(raw.stock_code).shift(k).values
        HIST += [f"rng_l{k}", f"volr_l{k}"]
    feat["rng0"], feat["rng5"], feat["rng20"], feat["rng60"] = raw.rng, roll("rng", 5, 3), roll("rng", 20, 10), roll("rng", 60, 30)
    feat["absret"] = feat.ret1.abs()
    feat["absret5"] = feat.absret.groupby(raw.stock_code).transform(lambda x: x.rolling(5, min_periods=3).mean())
    feat["volr5"] = feat.volr.groupby(raw.stock_code).transform(lambda x: x.rolling(5, min_periods=3).mean())
    feat["logmv"] = np.log(raw.mktcap.where(raw.mktcap > 0))
    feat["dow"] = raw.Date.dt.dayofweek
    HIST += ["rng0", "rng5", "rng20", "rng60", "absret", "absret5", "volr5", "logmv", "dow"]
    # PTT：到 t+1 開盤前可得（features_v4 的 ptt_open_*）再加異常留言量、|淨情緒|
    feat["ptt_open_att20"] = feat.ptt_open_att.groupby(raw.stock_code).transform(lambda x: x.rolling(20, min_periods=10).mean().shift(1))
    feat["ptt_open_abn"] = feat.ptt_open_att - feat.ptt_open_att20
    feat["ptt_open_absfac"] = feat.ptt_open_fac.abs()
    PTT = ["ptt_open_fac", "ptt_open_att", "ptt_open_att5", "ptt_open_abn", "ptt_open_absfac"]
    return feat, meta, raw, A["技術面"] + HIST, PTT


feat, meta, raw, BASE, PTT = prep(PATH)
chip = pd.read_pickle("tej_raw/tej_chip_daily.pkl")[["code", "date", "法人成交比重"]].drop_duplicates(["code", "date"])
chip = chip.rename(columns={"code": "stock_code", "date": "Date"})
n0 = len(raw)
raw = raw.merge(chip, on=["stock_code", "Date"], how="left")
assert len(raw) == n0
raw["rs"] = ((100 - raw["法人成交比重"]) / 100).where(raw.Volume > 0)
g = raw.groupby("stock_code")
feat["rs"] = raw.rs
feat["rs_dev"] = raw.rs - g.rs.transform(lambda x: x.rolling(20, min_periods=10).mean().shift(1))
feat["rs_5"] = g.rs.transform(lambda x: x.rolling(5, min_periods=3).mean())
RS = ["rs", "rs_dev", "rs_5"]
MARGIN = ["margin_chg", "margin_chg5", "short_chg", "short_margin_ratio", "margin_eligible"]
INST = [f"{c}_{k}" for c in ("foreign", "trust", "dealer", "inst") for k in ("r1", "r5")]
ARMS = {"基準": BASE, "+PTT★": BASE + PTT, "+融資融券": BASE + MARGIN, "+散戶成交比重": BASE + RS, "+法人買賣超": BASE + INST}
date, end = meta["date"], meta["end_date"]
has_ptt = (feat.ptt_open_att > 0).values
say(f"[資料] {len(raw):,} 列、{raw.stock_code.nunique()} 檔；隔天振幅 log(高/低) 中位數 {raw.y_vol.median():.4f}；有留言（到隔天開盤前）的股票日 {has_ptt.mean():.1%}")
say(f"       持續性：今天與隔天振幅的相關 {raw.rng.corr(raw.y_vol):.2f}；20 日平均振幅與隔天振幅的相關 {feat.rng20.corr(raw.y_vol):.2f}")


def daily_ic(dates, y, p):
    d = pd.DataFrame({"d": dates, "y": y, "p": p})
    d["ry"], d["rp"] = d.groupby("d").y.rank(), d.groupby("d").p.rank()
    return d.groupby("d").apply(lambda z: z.ry.corr(z.rp))


res = {}
for tgt, tname in [("y_vol", "隔天波動"), ("y_volu", "隔天成交量")]:
    y = raw[tgt]
    keep = y.notna() & feat[BASE].notna().all(axis=1)
    say(f"\n===== {tname}（XGBoost 迴歸，逐年滾動；Rank IC＝每日橫斷面 Spearman 的年平均）=====")
    R = {a: {"ic": [], "r2": [], "icd": [], "ic_ptt": []} for a in ARMS}
    for Y in TEST_YEARS:
        tr = keep & (end <= f"{Y-1}-06-30"); va = keep & (date >= f"{Y-1}-07-01") & (end <= f"{Y-1}-12-31")
        te = keep & (date >= f"{Y}-01-01") & (date <= f"{Y}-12-31")
        yte, dte = y[te].values, date[te].values
        line = f"  {Y}:"
        for arm, cols in ARMS.items():
            ps = []
            for s in range(SEEDS):
                m = xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=5, min_child_weight=50, subsample=0.8,
                                     colsample_bytree=0.8, random_state=s, tree_method="hist", device="cuda", early_stopping_rounds=30)
                m.fit(feat.loc[tr, cols], y[tr], eval_set=[(feat.loc[va, cols], y[va])], verbose=False)
                m.set_params(device="cpu")
                ps.append(m.predict(feat.loc[te, cols]))
            p = np.mean(ps, axis=0)
            icd = daily_ic(dte, yte, p)
            k = has_ptt[te.values]
            R[arm]["icd"].append(icd)
            R[arm]["ic"].append(icd.mean())
            R[arm]["r2"].append(1 - ((yte - p) ** 2).sum() / ((yte - y[tr].mean()) ** 2).sum())
            R[arm]["ic_ptt"].append(np.corrcoef(pd.Series(yte[k]).rank(), pd.Series(p[k]).rank())[0, 1])
            line += f" {arm} IC {R[arm]['ic'][-1]:.4f}／R² {R[arm]['r2'][-1]:.4f}｜"
        say(line)
    say("  --- 7 年彙總（相對基準）---")
    say(f"  基準：Rank IC {np.mean(R['基準']['ic']):.4f}、樣本外 R² {np.mean(R['基準']['r2']):.4f}")
    res[tgt] = {}
    for arm in list(ARMS)[1:]:
        dic = np.array(R[arm]["ic"]) - np.array(R["基準"]["ic"])
        dr2 = np.array(R[arm]["r2"]) - np.array(R["基準"]["r2"])
        dd = (pd.concat(R[arm]["icd"]) - pd.concat(R["基準"]["icd"])).dropna()
        dp = np.array(R[arm]["ic_ptt"]) - np.array(R["基準"]["ic_ptt"])
        t, p = tp(dic)
        res[tgt][arm] = {"dic": float(dic.mean()), "t": float(t), "p": float(p), "dr2": float(dr2.mean())}
        say(f"  {arm:<10s} ΔIC {dic.mean():+.4f}（年 t={t:.2f}，p={p:.4f}，正 {(dic > 0).sum()}/7；逐日 t={tp(dd)[0]:.2f}）｜ΔR² {dr2.mean():+.4f}（正 {(dr2 > 0).sum()}/7）"
            f"｜只看有留言的股票日 Δ相關 {dp.mean():+.4f}（t={tp(dp)[0]:.2f}）")
    x = res[tgt]["+PTT★"]
    say(f"  → {'V1' if tgt == 'y_vol' else 'V2'}：ΔIC {x['dic']:+.4f}，t={x['t']:.2f}，p={x['p']:.4f}（門檻 {ALPHA}）→ {'通過' if (x['p'] < ALPHA and x['dic'] > 0) else '未通過'}")

# ---------------- 迴歸：看得懂的大小 ----------------
CTRL = ["rng0", "rng5", "rng20", "absret", "volr", "volr5", "logmv"]


def panel(feat, raw, label, a, b):
    d = pd.DataFrame({"Date": raw.Date, "y_vol": raw.y_vol, "y_volu": raw.y_volu, "rng20": feat.rng20,
                      "att": feat.ptt_open_att, "abn": feat.ptt_open_abn, "net": feat.ptt_open_fac.fillna(0), "absnet": feat.ptt_open_absfac.fillna(0)})
    for c in CTRL:
        d[c] = feat[c]
    d = d[(d.Date >= a) & (d.Date <= b)]
    d["y_volr"] = np.log(d.y_vol.clip(lower=1e-4) / d.rng20.clip(lower=1e-4))      # 隔天振幅相對自己 20 日平均（log）
    say(f"\n===== 迴歸｜{label}｜{a[:4]}~{b[:4]}｜每個變數先減掉當天全體平均；依交易日群聚 =====")
    for ycol, yn, unit in [("y_volr", "隔天振幅（相對自己20日平均，%）", 100), ("y_volu", "隔天成交量（相對20日均量，%）", 100)]:
        xs = ["att", "abn", "net", "absnet"] + CTRL
        z = d.dropna(subset=[ycol] + xs)
        zz = z[[ycol] + xs] - z.groupby("Date")[[ycol] + xs].transform("mean")
        r = sm.OLS(zz[ycol] * unit, zz[xs]).fit(cov_type="cluster", cov_kwds={"groups": z.Date.dt.strftime("%Y%m%d")})
        r0 = sm.OLS(zz[ycol] * unit, zz[CTRL]).fit()
        say(f"  {yn}｜n={len(z):,}｜留言量(log) {r.params['att']:+.2f}（t={r.tvalues['att']:+.1f}）｜異常留言量 {r.params['abn']:+.2f}（t={r.tvalues['abn']:+.1f}）"
            f"｜淨情緒 {r.params['net']:+.2f}（t={r.tvalues['net']:+.1f}）｜|淨情緒| {r.params['absnet']:+.2f}（t={r.tvalues['absnet']:+.1f}）｜R² {r0.rsquared:.4f} → {r.rsquared:.4f}")
    return d


d = panel(feat, raw, "PTT", "2015-01-01", "2024-12-31")
panel(feat, raw, "PTT（同期對照）", "2020-01-01", "2024-12-31")
fc, mc, rc, _, _ = prep("final_dataset_v3_tejcm2/all_stocks_v3.csv")
panel(fc, rc, "CMoney 同學會（清理後×新模型）", "2020-01-01", "2024-12-31")

# ---------------- 依異常留言量分組 ----------------
say("\n===== 依 PTT 異常留言量分組（只取當天有留言的股票日，2018~2024）：隔天振幅相對自己 20 日平均 =====")
s = d[(d.Date >= "2018-01-01") & (d.att > 0) & d.abn.notna() & d.y_volr.notna()].copy()
s["q"] = pd.qcut(s.abn.rank(method="first"), 5, labels=False) + 1
gq = s.groupby("q").agg(n=("y_volr", "size"), nxt=("y_volr", "mean"), today=("rng0", "mean"), base=("rng20", "mean"), volu=("y_volu", "mean"))
for q, r in gq.iterrows():
    say(f"  第 {q} 組（{'最低' if q == 1 else '最高' if q == 5 else '　'}）n={int(r.n):,}｜今天振幅/20日平均 {r.today / r.base:.2f} 倍｜隔天振幅 {np.exp(r.nxt):.2f} 倍｜隔天成交量 {np.exp(r.volu):.2f} 倍")
nop = d[(d.Date >= "2018-01-01") & (d.att == 0) & d.y_volr.notna()]
say(f"  沒有留言的股票日 n={len(nop):,}｜隔天振幅 {np.exp(nop.y_volr.mean()):.2f} 倍｜隔天成交量 {np.exp(nop.y_volu.mean()):.2f} 倍")
open("vol_volume_test.log", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
json.dump(res, open("vol_volume_test_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
say("DONE")
