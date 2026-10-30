#!/usr/bin/env bash
# KITTI Object + Tracking 다운로드 (CC BY-NC-SA 3.0, 비상업 용도).
# 사용: KITTI_ROOT=~/data/kitti bash scripts/download_kitti.sh [object|tracking|all]
set -euo pipefail
ROOT="${KITTI_ROOT:-$HOME/data/kitti}"
URL="https://s3.eu-central-1.amazonaws.com/avg-kitti"
WHAT="${1:-all}"
mkdir -p "$ROOT/zips"

fetch() {  # 이어받기 지원
  curl -fSL -C - -o "$ROOT/zips/$1" "$URL/$1"
}

if [[ "$WHAT" == object || "$WHAT" == all ]]; then
  for f in data_object_calib.zip data_object_label_2.zip data_object_image_2.zip; do
    fetch "$f"; unzip -qo "$ROOT/zips/$f" -d "$ROOT/object"
  done
fi
if [[ "$WHAT" == tracking || "$WHAT" == all ]]; then
  for f in data_tracking_calib.zip data_tracking_label_2.zip data_tracking_image_2.zip; do
    fetch "$f"; unzip -qo "$ROOT/zips/$f" -d "$ROOT/tracking"
  done
fi
echo "KITTI ready at $ROOT"
