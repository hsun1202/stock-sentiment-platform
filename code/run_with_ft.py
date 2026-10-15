# -*- coding: utf-8 -*-
"""
不改原程式，改用「微調分類器」版資料集重跑既有檢驗：
  讀檔時把 final_dataset_v3/ 換成 final_dataset_v3_ft/；寫出的 .log/.csv/.png/.pkl 檔名加上 _ft（不覆蓋原結果）。
python run_with_ft.py ptt_cc_deepdive.py
"""
import builtins
import runpy
import sys

import pandas as pd

TAG = "ft"
if len(sys.argv) > 2 and sys.argv[1] == "--tag":       # python run_with_ft.py --tag cm ptt_cc_deepdive.py → 讀 final_dataset_v3_cm/、輸出 *_cm.log
    TAG = sys.argv[2]
    del sys.argv[1:3]
SRC, DST = "final_dataset_v3/", f"final_dataset_v3_{TAG}/"
_read_csv, _open = pd.read_csv, builtins.open


def read_csv(path, *a, **k):
    if isinstance(path, str):
        path = path.replace(SRC, DST)
    return _read_csv(path, *a, **k)


def tag(path):
    for ext in (".log", ".csv", ".png", ".pkl", ".json"):
        if path.endswith(ext) and "/" not in path and "\\" not in path:
            return path[:-len(ext)] + "_" + TAG + ext
    return path


def open_(path, mode="r", *a, **k):
    if isinstance(path, str) and ("w" in mode or "a" in mode):
        path = tag(path)
    return _open(path, mode, *a, **k)


pd.read_csv = read_csv
builtins.open = open_
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
