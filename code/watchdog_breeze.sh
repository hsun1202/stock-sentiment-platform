#!/usr/bin/env bash
# Breeze 情緒分類看門狗（2015-2024 版）：程式掛掉或卡住（log 20 分鐘沒更新）就自動重啟。
# breeze_infer_267_fast.py 每 4000 則寫一次 .partial.csv，已完成的股票會跳過，所以重啟不會重算。
# 事件寫在 watchdog_breeze.log（START / RESTART / STALL / DONE / GIVEUP）。
# 用法：cd /d/專題claude && nohup bash watchdog_breeze.sh > /dev/null 2>&1 &

cd "/d/專題claude" || exit 1
LOG=breeze_2015_2024.log
WLOG=watchdog_breeze.log
MAX_RESTARTS=20
restarts=0
pid=""

ts() { date '+%Y-%m-%d %H:%M:%S'; }
event() { echo "$(ts) $*" >> "$WLOG"; }

start_job() {
  PYTHONUTF8=1 python -u breeze_infer_267_fast.py --mode constrained \
    --batch-dir ptt_output_267_2015_2024 --out-dir sentiment_output_267_2015_2024 >> "$LOG" 2>&1 &
  pid=$!
  event "START pid=$pid"
}

touch "$LOG"
start_job
while true; do
  sleep 60
  if kill -0 "$pid" 2>/dev/null; then
    age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
    if [ "$age" -gt 1200 ]; then
      event "STALL log ${age}s 沒更新，重啟"
      kill "$pid" 2>/dev/null; sleep 5; kill -9 "$pid" 2>/dev/null
      restarts=$((restarts + 1))
      sleep 20
      start_job
    fi
    continue
  fi
  if tail -5 "$LOG" | grep -q "完成情緒推論"; then
    event "DONE 全部分類完成"
    exit 0
  fi
  restarts=$((restarts + 1))
  event "RESTART 程式意外結束，第 $restarts 次重啟；log 最後：$(tail -1 "$LOG" | tr -d '\r')"
  if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
    event "GIVEUP 重啟超過 $MAX_RESTARTS 次，需要人工除錯"
    exit 1
  fi
  sleep 30
  start_job
done
