# -*- coding: utf-8 -*-
"""每天執行一次：更新資料 → 逐日處理每個策略帳戶 → 產生報告。漏跑的日子下次會自動補上。
python papertrade/run_daily.py [--report-only]"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd  # noqa: E402

import report  # noqa: E402
from broker import Account  # noqa: E402
from strategies import STRATEGIES, returns_panel  # noqa: E402
from update import update  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    open(os.path.join(BASE, "run.log"), "a", encoding="utf-8").write(line + "\n")


def broker_state_is_real():
    import broker
    return os.path.basename(broker.STATE) == "state"      # 測試模式（test_state）不覆蓋簡報素材


def quarter(d):
    t = pd.Timestamp(d)
    return (t.year, (t.month - 1) // 3)


def main(report_only=False, start=None):
    """start：測試用，假裝從過去某天開始（搭配另一個 state 資料夾），之後的日子照常逐日處理。"""
    prices, exr = update(log) if not report_only else (pd.read_csv(os.path.join(BASE, "data", "prices.csv"), dtype={"code": str}),
                                                       pd.read_csv(os.path.join(BASE, "data", "exright.csv"), dtype={"code": str}))
    names = prices.drop_duplicates("code", keep="last").set_index("code").name.to_dict()
    if not report_only:
        close, value, ret = returns_panel(prices, exr)
        opn = prices.pivot(index="date", columns="code", values="open").sort_index()
        base = close.shift(1).ffill()
        if len(exr):
            ref = exr.pivot_table(index="date", columns="code", values="ref_price", aggfunc="last").reindex(index=close.index, columns=close.columns)
            base = ref.where(ref.notna(), base)
        last_close = close.ffill()
        days = list(close.index)
        ctx = {"close": close, "value": value, "ret": ret}
        exr_by_day = {d: dict(zip(g.code, zip(g.prev_close, g.ref_price))) for d, g in exr.groupby("date")}
        for k, (desc, fn, freq) in STRATEGIES.items():
            a = Account(k)
            if a.s["last_day"] is None:                       # 第一次：最新一天出訊號，隔天開盤建倉
                d0 = start or days[-1]
                a.s["start"] = d0
                a.signal(d0, fn(ctx, d0))
                a.mark(d0, last_close.loc[d0].to_dict())
                log(f"[{k}] 開始（{d0}）：訊號 {len(a.s['pending']['weights'])} 檔，{d0} 之後的第一個交易日開盤建倉")
                a.save()
                if not start:
                    continue
            for i, d in enumerate(days):
                if d <= a.s["last_day"]:
                    continue
                a.dividends(d, exr_by_day.get(d, {}))          # 先：前一天收盤持有的股票才領得到除權息
                res = a.execute(d, opn.loc[d].to_dict(), base.loc[d].to_dict(), last_close.loc[d].to_dict())
                if res:
                    tr, bl = res
                    log(f"[{k}] {d} 成交 {len(tr)} 筆" + (f"；漲跌停擋住 {len(bl)} 檔，明天重試" if bl else ""))
                nav = a.mark(d, last_close.loc[d].to_dict())
                prev = days[i - 1] if i > 0 else None
                if freq == "Q" and prev and quarter(d) != quarter(prev) and a.s.get("last_signal") != d:
                    w = fn(ctx, d)
                    a.signal(d, w)
                    log(f"[{k}] {d} 新一季訊號：{len(w)} 檔，隔天開盤換股")
                a.save()
            log(f"[{k}] 處理到 {a.s['last_day']}")
    r = report.build(names)
    if r is not None:
        log("[報告] " + "｜".join(f"{x['策略'][:6]} {x['累積報酬']}" for _, x in r.iterrows()))
    if broker_state_is_real():
        try:
            import export_slides
            export_slides.export()                         # 給簡報用的淨值圖與數字（ppt_build/assets/）
        except Exception as e:                             # 匯出失敗不影響模擬本身
            log(f"[簡報匯出] 失敗：{type(e).__name__}: {e}")
    log("DONE")


if __name__ == "__main__":
    if "--test-from" in sys.argv:                          # 測試：python run_daily.py --test-from 2026-09-15（結果在 test_state/、test_report/）
        import broker
        broker.STATE = os.path.join(BASE, "test_state")
        report.set_out(os.path.join(BASE, "test_report"))
        main(start=sys.argv[sys.argv.index("--test-from") + 1])
    else:
        main(report_only="--report-only" in sys.argv)
