#!/bin/bash
# 原論文作法（per-stock LSTM 預測收盤價）十年版：4組（散戶情緒版）× 10 seeds，4個平行。已完成(有metrics.json)的自動跳過，可重複執行。
cd /d/專題claude
OUT=lstm_price_v4
mkdir -p $OUT
echo "$(date '+%Y-%m-%d %H:%M:%S') START" >> $OUT/_run.log
for s in 0 1 2 3 4 5 6 7 8 9; do
  for arm in "技術面" "+PTT" "+融資融券" "+PTT+融資融券"; do echo "$s|$arm"; done
done | PYTHONUTF8=1 xargs -P 4 -I{} bash -c '
  s="${1%%|*}"; arm="${1#*|}"; d="'$OUT'/${arm}_seed${s}"
  [ -f "$d/metrics.json" ] && exit 0
  mkdir -p "$d"
  python -u lstm_price_v4.py --arm "$arm" --seed "$s" --out-dir "$d" > "$d.log" 2>&1
  echo "$(date "+%Y-%m-%d %H:%M:%S") done $arm seed$s" >> '$OUT'/_run.log
' _ {}
echo "$(date '+%Y-%m-%d %H:%M:%S') ALL_DONE" >> $OUT/_run.log
PYTHONUTF8=1 python -u summarize_lstm_price_v4.py > summarize_lstm_price_v4.log 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') SUMMARY_DONE" >> $OUT/_run.log
