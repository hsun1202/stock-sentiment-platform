# -*- coding: utf-8 -*-
"""
扣交易成本回測：用 LSTM 正式版（lstm_direction_v4.py，10 seeds 平均）的測試期預測，2018~2024 七年
============================================================
策略（每個交易日 t 收盤後決定，t+1 開盤買進、收盤賣出＝當沖，每檔等權）：
  做多 top k%：預測上漲機率最高的前 k%（k = 1, 5, 10）
  多空 k%：做多前 k%、放空後 k%（現股當沖先賣後買），兩邊各半資金
  對照：全部股票等權買進（大盤當沖基準）；只用技術面的 LSTM
交易成本（來回、以成交金額計）：
  手續費 0.1425% × 2（無折扣）或 × 0.6（常見網路下單六折）
  證交稅：當沖 0.15%（2017-04-28 起減半；2018 以後測試期都適用）
  → 無折扣 0.435%、六折 0.321%；另列「不扣成本」看訊號本身
實務限制：t+1 開盤即漲停（買不到）或跌停（賣不到/空不到）排除、t+1 成交量 0 排除
輸出：backtest_v4_results.json、backtest_v4.log
python backtest_v4.py
"""
import json
import numpy as np
import pandas as pd

from features_v4 import build

WIN = 20
SEEDS = range(10)
TEST_YEARS = range(2018, 2025)
COSTS = {"不扣成本": 0.0, "六折手續費(0.321%)": 2 * 0.001425 * 0.6 + 0.0015, "無折扣(0.435%)": 2 * 0.001425 + 0.0015}
ARMS = ["技術面", "+籌碼(當天盤後公布)"]

feat, meta, _ = build()
df = pd.read_csv("final_dataset_v3/all_stocks_v3.csv", dtype={"stock_code": str},
                 usecols=["stock_code", "Date", "Open", "Close", "Volume"])
df = df.sort_values(["stock_code", "Date"]).reset_index(drop=True)
g = df.groupby("stock_code")
nextO, nextC, nextV = g["Open"].shift(-1), g["Close"].shift(-1), g["Volume"].shift(-1)
prevC_next = df["Close"]  # t+1 的前一日收盤 = t 收盤
r_oc = (nextC / nextO - 1).values                      # t+1 開盤→收盤報酬
open_move = (nextO / prevC_next - 1).values            # t+1 開盤相對 t 收盤（判斷開盤漲跌停）
tradable = (nextV > 0).values & (np.abs(open_move) < 0.095)

code = meta["stock_code"].values
has_hist = np.zeros(len(meta), dtype=bool)
has_hist[WIN - 1:] = code[WIN - 1:] == code[:len(code) - WIN + 1]
base_ok = meta["valid"].values & has_hist
date, end_date = meta["date"], meta["end_date"]

rows = []
for Y in TEST_YEARS:
    te = np.where(base_ok & (date >= f"{Y}-01-01").values & (date <= f"{Y}-12-31").values
                  & (end_date <= f"{Y+1}-01-31").values)[0]
    d = pd.DataFrame({"date": date.values[te], "code": code[te], "r": r_oc[te], "ok": tradable[te]})
    for arm in ARMS:
        d[arm] = np.mean([np.load(f"lstm_v4/{Y}_{arm}_{s}.npy") for s in SEEDS], axis=0)
    rows.append(d)
bt = pd.concat(rows, ignore_index=True)
bt = bt[bt["ok"]].drop(columns="ok")
print(f"回測樣本 {len(bt):,} 個股票日、{bt['date'].nunique()} 個交易日，{bt['date'].min().date()}~{bt['date'].max().date()}")


def daily_returns(arm, k, mode):
    """回傳每日毛報酬（未扣成本）與當天交易檔數。"""
    out = []
    for dt, x in bt.groupby("date"):
        n = max(1, int(round(len(x) * k)))
        s = x.sort_values(arm, ascending=False)
        if mode == "long":
            ret = s["r"].iloc[:n].mean()
        else:  # 多空各半
            ret = 0.5 * s["r"].iloc[:n].mean() - 0.5 * s["r"].iloc[-n:].mean()
        out.append((dt, ret, n))
    return pd.DataFrame(out, columns=["date", "gross", "n"]).set_index("date")


def stats(ret):
    eq = (1 + ret).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    ann = eq.iloc[-1] ** (252 / len(ret)) - 1
    sharpe = ret.mean() / ret.std(ddof=1) * np.sqrt(252) if ret.std() > 0 else 0
    return {"年化報酬": ann, "Sharpe": sharpe, "日勝率": (ret > 0).mean(), "最大回撤": dd,
            "總報酬": eq.iloc[-1] - 1, "日均毛報酬": ret.mean()}


lines, res = [], {}


def log(s=""):
    print(s)
    lines.append(s)


bench = bt.groupby("date")["r"].mean()
log("\n【對照】全部股票等權當沖（每天都買）：日均毛報酬 {:+.3%}".format(bench.mean()))
for cname, c in COSTS.items():
    st = stats(bench - c)
    log(f"   {cname:18s} 年化 {st['年化報酬']:+8.1%}｜Sharpe {st['Sharpe']:+.2f}｜日勝率 {st['日勝率']:.1%}")

for mode, mname in (("long", "做多"), ("ls", "多空")):
    for k in (0.01, 0.05, 0.10):
        for arm in ARMS:
            dr = daily_returns(arm, k, mode)
            cost_mult = 1 if mode == "long" else 1  # 多空各半資金、兩邊都付來回成本 → 以總資金計仍是1倍
            key = f"{mname} top{int(k*100)}% [{arm}]"
            res[key] = {"平均每天檔數": float(dr["n"].mean()), "日均毛報酬": float(dr["gross"].mean())}
            log(f"\n{key}（每天約 {dr['n'].mean():.0f} 檔）日均毛報酬 {dr['gross'].mean():+.3%}")
            for cname, c in COSTS.items():
                net = dr["gross"] - c * cost_mult
                st = stats(net)
                yr = net.groupby(net.index.year).apply(lambda s: (1 + s).prod() - 1)
                res[key][cname] = {**{k2: float(v) for k2, v in st.items()},
                                   "各年報酬": {int(y): float(v) for y, v in yr.items()}}
                log(f"   {cname:18s} 年化 {st['年化報酬']:+8.1%}｜Sharpe {st['Sharpe']:+.2f}｜日勝率 {st['日勝率']:.1%}｜"
                    f"最大回撤 {st['最大回撤']:.1%}｜各年 " + " ".join(f"{y}:{v:+.0%}" for y, v in yr.items()))

json.dump(res, open("backtest_v4_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
open("backtest_v4.log", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n完成 → backtest_v4_results.json、backtest_v4.log")
