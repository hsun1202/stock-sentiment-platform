#!/usr/bin/env bash
# 把專題資料夾的最新程式、結果紀錄、文件、最新版簡報同步到 GitHub 私人專案（只同步白名單，不含原始資料與 token）
#   來源：D:\專題claude    匯出：D:\專題claude_github（git 專案，remote＝GitHub 私人 repo）
#   用法：bash sync_github.sh            # 有變動才 commit + push；沒變動就什麼都不做
# 由工作排程器定時執行（見 PROJECT_STATUS.md），也可以手動跑。
set -euo pipefail
SRC="/d/專題claude"
DST="/d/專題claude_github"
GH="/c/Program Files/GitHub CLI/gh.exe"
LOG="$SRC/sync_github.log"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

mkdir -p "$DST"/{code,results,slides,docs}
# 1) 程式
cp -u "$SRC"/*.py "$SRC"/*.sh "$DST/code/" 2>/dev/null || true
# 2) 結果紀錄（.log/.out/*.json；排除 token、排除自己的同步紀錄）
cp -u "$SRC"/*.log "$SRC"/*.out "$DST/results/" 2>/dev/null || true
rm -f "$DST/results/sync_github.log"
for f in "$SRC"/*.json; do
  b=$(basename "$f")
  case "$b" in *token*) continue ;; esac
  cp -u "$f" "$DST/results/"
done
# 子資料夾：季頻高低波動、CMoney 完整資料（不傳抓下來的價格 csv）
mkdir -p "$DST/results/cmoney_full"
cp -u "$SRC"/cmoney_full/*.csv "$DST/results/cmoney_full/" 2>/dev/null || true
cp -u "$SRC"/cmoney_full/*.py "$SRC"/cmoney_full/*.sh "$DST/code/" 2>/dev/null || true
cp -u "$SRC"/cmoney_full/*.log "$SRC"/cmoney_full/*.md "$SRC"/cmoney_full/*.json "$DST/results/cmoney_full/" 2>/dev/null || true
mkdir -p "$DST/results/quarterly_vol"
cp -u "$SRC"/quarterly_vol/*.py "$DST/code/" 2>/dev/null || true
cp -u "$SRC"/quarterly_vol/*.log "$SRC"/quarterly_vol/*.png "$SRC"/quarterly_vol/*.md "$SRC"/quarterly_vol/quarterly_vol_returns.csv "$DST/results/quarterly_vol/" 2>/dev/null || true
# 3) 文件
cp -u "$SRC/PROJECT_STATUS.md" "$DST/docs/"
# 4) 最新版簡報（版本號最大的 build_deck_vN.js 及對應 pptx、講稿）
latest=$(ls "$SRC"/ppt_build/build_deck_v*.js | sed -E 's/.*_v([0-9]+)\.js/\1/' | sort -n | tail -1)
cp -u "$SRC/ppt_build/build_deck_v$latest.js" "$SRC/ppt_build/package.json" "$DST/slides/"
[ -f "$SRC/ppt_build/final_report_v$latest.pptx" ] && cp -u "$SRC/ppt_build/final_report_v$latest.pptx" "$DST/slides/"
[ -f "$SRC/ppt_build/講稿_v$latest.md" ] && cp -u "$SRC/ppt_build/講稿_v$latest.md" "$DST/slides/"

# 5) 程式索引（自動依每支程式的開頭說明產生）
PYTHONUTF8=1 python "$SRC/make_code_index.py" "$(cygpath -m "$DST")" >/dev/null 2>&1 || log "程式索引產生失敗（不影響同步）"

cd "$DST"
git add -A
if git diff --cached --quiet; then
  log "沒有變動"
  exit 0
fi
# 安全檢查：不可有 token 檔、不可有疑似金鑰、不可有大檔
if git diff --cached --name-only | grep -i -E 'token.*\.json|\.csv$|\.pkl$|\.jsonl$|\.zip$'; then
  log "中止：暫存區出現資料檔或 token 檔"; git reset -q; exit 1
fi
if git diff --cached -U0 | grep -E '^\+' | grep -q -E 'eyJ[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|Bearer [A-Za-z0-9._-]{20,}'; then
  log "中止：新增內容疑似含金鑰"; git reset -q; exit 1
fi
big=$(git diff --cached --name-only | while read -r f; do [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 50000000 ] && echo "$f"; done || true)
if [ -n "$big" ]; then log "中止：有超過 50MB 的檔案：$big"; git reset -q; exit 1; fi

n=$(git diff --cached --name-only | wc -l)
git -c user.name="hsun1202" commit -q -m "自動同步 $(date '+%F %H:%M')（$n 個檔案）"
if git push -q origin main 2>>"$LOG"; then
  log "已上傳 $n 個檔案"
else
  log "push 失敗（可能未登入或網路問題），下次再試"
  exit 1
fi
