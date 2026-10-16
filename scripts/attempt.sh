#!/usr/bin/env bash
# Record one attempt. Add --execute to command the robot and scissor.
set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: bash scripts/attempt.sh MODEL NAME [--execute] [options]; MODEL: act_rgb/act_rgbd/dp_rgb/dp_rgbd/fm_rgb/fm_rgbd/act3d/dp3/fm3" >&2
  exit 2
fi
variant="$1"
label="$2"
shift 2
case "$variant" in
  act_rgb|act_rgbd|dp_rgb|dp_rgbd|fm_rgb|fm_rgbd|act3d|dp3|fm3) ;;
  *) echo "Unknown model: $variant" >&2; exit 2 ;;
esac
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec bash "$repo/scripts/robot/$variant.sh" --record-attempt "$label" \
  --chunks 100000 --show-camera "$@"
