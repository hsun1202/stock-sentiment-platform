#!/bin/bash
# CMoney 完整資料：LSTM 5 組 × 5 seeds（GPU 1 條線＋CPU 8 個平行）→ 彙總。已完成的自動跳過，可重複執行。
# 實測：這個小 LSTM（batch 10）GPU 單一任務最快（每檔約 6 秒），但 GPU 上多開幾個不會變快；CPU 16 核可同時跑 8 個。
cd /d/專題claude
# CM_TAG=cm3 → 讀補抓後的資料集、結果放 cmoney_full/lstm_cm3/
CM_TAG=${CM_TAG:-cm2}; export CM_TAG
OUT=cmoney_full/lstm$([ "$CM_TAG" = cm2 ] || echo "_$CM_TAG"); export LSTM_DIR=$OUT
mkdir -p $OUT
log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> cmoney_full/_run.log; }
log "LSTM START ($CM_TAG)"
job='s="${1%%|*}"; arm="${1#*|}"; d="$LSTM_DIR/${arm}_seed${s}"
  [ -f "$d/metrics.json" ] && exit 0
  mkdir -p "$d"
  python -u cmoney_full/lstm_price_cm.py --arm "$arm" --seed "$s" --out-dir "$d" > "$d.log" 2>&1
  echo "$(date "+%Y-%m-%d %H:%M:%S") done $arm seed$s ($DEV)" >> cmoney_full/_run.log'
ARMS=("技術面" "+PTT" "+CMoney" "+PTT+CMoney" "+CMoney收盤前")
# 依 seed 分硬體（同一個 seed 的 5 組都在同一種硬體上，比較時不混入 CPU/GPU 差異）：
# GPU 一條線依序跑 seed 4 的 5 組；CPU 8 個平行跑 seed 0–3 的 20 個（實測 GPU 0.16 檔/秒、CPU 8 平行 0.64 檔/秒，兩邊約同時跑完）
( for arm in "${ARMS[@]}"; do echo "4|$arm"; done | DEV=gpu PYTHONUTF8=1 xargs -P 1 -I{} bash -c "$job" _ {} ) &
for s in 0 1 2 3; do
  for arm in "${ARMS[@]}"; do echo "$s|$arm"; done
done | DEV=cpu LSTM_DEVICE=cpu LSTM_THREADS=2 PYTHONUTF8=1 xargs -P 8 -I{} bash -c "$job" _ {}
wait
log "LSTM_ALL_DONE ($CM_TAG)"
PYTHONUTF8=1 python -u cmoney_full/summarize_cm.py > cmoney_full/summarize_cm.out 2>&1 && log "SUMMARY_DONE ($CM_TAG)" || log "SUMMARY_FAILED ($CM_TAG)"
