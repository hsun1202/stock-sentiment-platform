#!/bin/bash
# CMoney 完整資料：XGBoost（已跑完則跳過）→ LSTM 5 組 × 5 seeds（CPU、8 個平行、每個 2 執行緒）→ 彙總。已完成的自動跳過，可重複執行。
# 實測：這個小 LSTM（batch 10）在 GPU 上平行沒有加速，CPU 每檔約 6–7 秒，16 核可同時跑 8 個。
cd /d/專題claude
OUT=cmoney_full/lstm
mkdir -p $OUT
log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> cmoney_full/_run.log; }
log START
if [ ! -f cmoney_full/xgb_direction_cm.log ]; then
  PYTHONUTF8=1 python -u cmoney_full/xgb_direction_cm.py > cmoney_full/xgb_direction_cm.out 2>&1 && log XGB_DONE || log XGB_FAILED
fi
for s in 0 1 2 3 4; do
  for arm in "技術面" "+PTT" "+CMoney" "+PTT+CMoney" "+CMoney收盤前"; do echo "$s|$arm"; done
done | LSTM_DEVICE=cpu LSTM_THREADS=2 PYTHONUTF8=1 xargs -P 8 -I{} bash -c '
  s="${1%%|*}"; arm="${1#*|}"; d="'$OUT'/${arm}_seed${s}"
  [ -f "$d/metrics.json" ] && exit 0
  mkdir -p "$d"
  python -u cmoney_full/lstm_price_cm.py --arm "$arm" --seed "$s" --out-dir "$d" > "$d.log" 2>&1
  echo "$(date "+%Y-%m-%d %H:%M:%S") done $arm seed$s" >> cmoney_full/_run.log
' _ {}
log LSTM_ALL_DONE
PYTHONUTF8=1 python -u cmoney_full/summarize_cm.py > cmoney_full/summarize_cm.out 2>&1 && log SUMMARY_DONE || log SUMMARY_FAILED
