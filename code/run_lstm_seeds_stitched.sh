#!/bin/bash
# 20組種子 x (含/不含情緒) 的per-stock LSTM，資料為PTT全板斷句修復+強制三選一分類器(2020-2024)。
# 已完成的組合(有metrics.json)會自動跳過，可安全重複執行。
cd /d/專題claude
OUT=lstm_stitched_seeds
INPUT=final_dataset_267_stitched/all_stocks_final_dataset_267.csv
mkdir -p $OUT
SEEDS="42 1 7 123 6 10 55 99 256 777 2025 3141 8080 404 999 111 222 333 444 2024"
for s in $SEEDS; do
  echo "$s with"
  echo "$s without"
done | PYTHONUTF8=1 xargs -P 4 -L 1 bash -c '
  s=$0; m=$1
  d='$OUT'/seed${s}_${m}
  [ -f $d/metrics.json ] && exit 0
  flag=""; [ "$m" = "without" ] && flag="--no-sentiment"
  python -u train_lstm_per_stock.py --input '$INPUT' --out-dir $d --seed $s $flag > $d.log 2>&1
'
echo "ALL_DONE"
