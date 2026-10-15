#!/usr/bin/env bash
# pttweb 爬蟲看門狗：爬蟲掛掉或卡住（log 10 分鐘沒更新）就自動重啟；
# 爬完後若還有沒抓到的文章（失敗的），再補跑，最多補 3 次。
# 事件寫在 watchdog_*.log（START / RESTART / STALL / RETRY / DONE / GIVEUP）。
# 用法：
#   2015-2019初舊文章：cd /d/專題claude && nohup bash watchdog_pttweb.sh > /dev/null 2>&1 &
#   2019-04~2024 PTT上已刪文章：cd /d/專題claude && nohup bash watchdog_pttweb.sh deleted > /dev/null 2>&1 &

cd "/d/專題claude" || exit 1
if [ "$1" = "deleted" ]; then
  OUT=pttweb_stock_2019_2024_deleted; START=2019-04-01; END=2024-12-31
  SKIP="$OUT/ptt_existing_ids.txt"
  LOG=crawl_pttweb_deleted.log; WLOG=watchdog_pttweb_deleted.log
else
  OUT=pttweb_stock_2015_2019; START=2015-01-01; END=2019-03-31; SKIP=""
  LOG=crawl_pttweb.log; WLOG=watchdog_pttweb.log
fi
MAX_RESTARTS=30
restarts=0
retry_rounds=0
pid=""

ts() { date '+%Y-%m-%d %H:%M:%S'; }
event() { echo "$(ts) $*" >> "$WLOG"; }

start_crawler() {
  if [ -n "$SKIP" ]; then
    PYTHONUTF8=1 python -u crawl_pttweb_api.py --start "$START" --end "$END" \
      --out-dir "$OUT" --delay 0.5 --skip-ids "$SKIP" >> "$LOG" 2>&1 &
  else
    PYTHONUTF8=1 python -u crawl_pttweb_api.py --start "$START" --end "$END" \
      --out-dir "$OUT" --delay 0.5 >> "$LOG" 2>&1 &
  fi
  pid=$!
  event "START pid=$pid"
}

# 還剩幾篇沒抓（列表還沒翻完時回傳 list）
remaining() {
  [ -f "$OUT/articles.csv" ] || { echo "list"; return; }
  OUT="$OUT" SKIP="$SKIP" PYTHONUTF8=1 python -c '
import csv, json, os
out, skip = os.environ["OUT"], os.environ["SKIP"]
arts = {r["article_aid"] for r in csv.DictReader(open(out + "/articles.csv", encoding="utf-8-sig"))}
if skip:
    arts -= {l.strip() for l in open(skip, encoding="utf-8")}
done = set()
raw = out + "/raw_articles.jsonl"
if os.path.exists(raw):
    for line in open(raw, encoding="utf-8"):
        try:
            done.add(json.loads(line)["_aid"])
        except Exception:
            pass
print(len(arts - done))
'
}

touch "$LOG"
start_crawler
while true; do
  sleep 60
  if kill -0 "$pid" 2>/dev/null; then
    age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
    if [ "$age" -gt 600 ]; then
      event "STALL log ${age}s 沒更新，重啟"
      kill "$pid" 2>/dev/null; sleep 5; kill -9 "$pid" 2>/dev/null
      restarts=$((restarts + 1))
      start_crawler
    fi
    continue
  fi
  # 爬蟲已結束
  left=$(remaining)
  if [ "$left" = "0" ]; then
    event "DONE 全部抓完"
    exit 0
  fi
  if tail -3 "$LOG" | grep -q "階段二完成"; then
    retry_rounds=$((retry_rounds + 1))
    if [ "$retry_rounds" -gt 3 ]; then
      event "GIVEUP 補跑3次後仍有 $left 篇抓不到（見 $OUT/fetch_failed.txt）"
      exit 1
    fi
    event "RETRY 正常結束但還有 $left 篇沒抓到，第 $retry_rounds 次補跑"
  else
    restarts=$((restarts + 1))
    event "RESTART 爬蟲意外結束（剩 $left），第 $restarts 次重啟；log 最後：$(tail -1 "$LOG" | tr -d '\r')"
  fi
  if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
    event "GIVEUP 重啟超過 $MAX_RESTARTS 次，需要人工除錯"
    exit 1
  fi
  sleep 30
  start_crawler
done
