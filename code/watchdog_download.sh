#!/usr/bin/env bash
# 2015-2019 籌碼/官方日行情下載看門狗：掛掉就重啟（已下載的檔案會跳過）；log 40分鐘沒更新視為卡住
# （下載程式被證交所擋時會退避最久15分鐘，所以門檻設寬一點）。事件寫在 watchdog_download.log
# 用法：cd /d/專題claude && nohup bash watchdog_download.sh > /dev/null 2>&1 &
cd "/d/專題claude" || exit 1
LOG=download_chip_2015.log; WLOG=watchdog_download.log; restarts=0
ts() { date '+%Y-%m-%d %H:%M:%S'; }
event() { echo "$(ts) $*" >> "$WLOG"; }
start_job() { PYTHONUTF8=1 python -u download_chip_data_2015.py >> "$LOG" 2>&1 & pid=$!; event "START pid=$pid"; }
touch "$LOG"; start_job
while true; do
  sleep 60
  if kill -0 "$pid" 2>/dev/null; then
    age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
    if [ "$age" -gt 2400 ]; then event "STALL log ${age}s 沒更新，重啟"; kill "$pid"; sleep 5; kill -9 "$pid" 2>/dev/null; restarts=$((restarts+1)); start_job; fi
    continue
  fi
  if tail -2 "$LOG" | grep -q "\[完成\]"; then event "DONE $(tail -1 "$LOG" | tr -d '\r')"; exit 0; fi
  restarts=$((restarts+1)); event "RESTART 第 $restarts 次；log 最後：$(tail -1 "$LOG" | tr -d '\r')"
  [ "$restarts" -gt 20 ] && { event "GIVEUP"; exit 1; }
  sleep 30; start_job
done
