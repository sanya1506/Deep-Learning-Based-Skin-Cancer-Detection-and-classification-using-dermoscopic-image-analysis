#!/bin/bash
# usage: watch.sh <interval_sec> slug1 [slug2 ...]
# exits when any slug reaches a terminal state, or after 3 failed status queries in a row
K=/c/Users/hp/AppData/Roaming/Python/Python314/Scripts/kaggle.exe
iv=$1; shift; fails=0
while true; do
  line="$(date +%H:%M)"; done_=0; unk=0
  for s in "$@"; do
    st=$("$K" kernels status "$s" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*' || echo UNKNOWN)
    line="$line $s=$st"
    case "$st" in *COMPLETE|*ERROR|*CANCEL*) done_=1;; UNKNOWN) unk=1;; esac
  done
  [ $unk = 1 ] && fails=$((fails+1)) || fails=0
  echo "$line"; [ $done_ = 1 ] && break; [ $fails -ge 3 ] && { echo "3 failed queries in a row"; break; }
  sleep "$iv"
done
