#!/usr/bin/env bash
# run_arm.sh — launch one HTP-WM ablation arm and log start/end.
#
# Usage:
#   ./run_arm.sh <arm> <seed> [<task>] [<size>]
#
# Examples:
#   ./run_arm.sh baseline 0
#   ./run_arm.sh htp_full 1
#   ./run_arm.sh htp_separable 2 atari100k_pong
#   ./run_arm.sh htp_joint 0 atari100k_breakout size12m
#
# Arms (must match the tracker exactly):
#   baseline  htp_empty_projection  htp_full  htp_recon_only  htp_pdyn_only
#   htp_separable  htp_joint  htp_reverse_strides
#   htp_slim  htp_slim_vicreg  htp_slim_postHoc
#
# Writes stdout + timestamps to  runlogs/<run_name>.log
# Names the W&B run to match the tracker row via WANDB_RUN_NAME.

set -euo pipefail

ARM=${1:?arm required (baseline | htp_empty_projection | htp_full | htp_recon_only | htp_pdyn_only | htp_separable | htp_joint | htp_reverse_strides | htp_slim | htp_slim_vicreg | htp_slim_postHoc)}
SEED=${2:?seed required (e.g. 0, 1, 2)}
TASK=${3:-atari100k_breakout}
SIZE=${4:-size25m}

# ---------------------------------------------------- arm -> --configs chain
case "$ARM" in
  baseline)              CONFIGS="atari100k $SIZE wandb" ;;
  htp_empty_projection)  CONFIGS="htp_atari100k htp_empty_projection $SIZE wandb" ;;
  htp_full)              CONFIGS="htp_atari100k $SIZE wandb" ;;
  htp_recon_only)        CONFIGS="htp_atari100k htp_matryoshka_only $SIZE wandb" ;;
  htp_pdyn_only)         CONFIGS="htp_atari100k htp_pdyn_only $SIZE wandb" ;;
  htp_separable)         CONFIGS="htp_atari100k htp_separable $SIZE wandb" ;;
  htp_joint)             CONFIGS="htp_atari100k htp_joint $SIZE wandb" ;;
  htp_reverse_strides)   CONFIGS="htp_atari100k htp_reverse_strides $SIZE wandb" ;;
  htp_slim)              CONFIGS="htp_atari100k htp_slim $SIZE wandb" ;;
  htp_slim_vicreg)       CONFIGS="htp_atari100k htp_slim_vicreg $SIZE wandb" ;;
  htp_slim_postHoc)      CONFIGS="htp_atari100k htp_slim_postHoc $SIZE wandb" ;;
  *)
    echo "ERROR: unknown arm '$ARM'" >&2
    echo "Valid: baseline, htp_empty_projection, htp_full, htp_recon_only, htp_pdyn_only," >&2
    echo "       htp_separable, htp_joint, htp_reverse_strides," >&2
    echo "       htp_slim, htp_slim_vicreg, htp_slim_postHoc" >&2
    exit 2
    ;;
esac

# ---------------------------------------------------- W&B labels (match the tracker)
RUN_NAME="${ARM}_${TASK}_seed${SEED}"
export WANDB_PROJECT="${WANDB_PROJECT:-htp-wm}"
export WANDB_GROUP="${WANDB_GROUP:-${TASK}-ablation}"
export WANDB_JOB_TYPE="$ARM"
export WANDB_RUN_NAME="$RUN_NAME"
export WANDB_TAGS="ablation,${ARM},${TASK},seed${SEED},${SIZE}"
# WANDB_ENTITY / WANDB_MODE: set in your shell profile if you need them.

# ---------------------------------------------------- log setup
LOGDIR="runlogs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${RUN_NAME}.log"

banner() {
  local kind="$1" ts="$2" extra="${3:-}"
  {
    echo "======================================================================"
    printf "  %-8s %s\n" "$kind" "$ts"
    [ -n "$extra" ] && echo "$extra"
    echo "======================================================================"
  }
}

START=$(date '+%Y-%m-%d %H:%M:%S')
{
  banner "START" "$START" "$(printf '  %-8s %s\n' \
    'Run:' "$RUN_NAME" \
    'Arm:' "$ARM" \
    'Task:' "$TASK" \
    'Seed:' "$SEED" \
    'Size:' "$SIZE" \
    'Configs:' "$CONFIGS" \
    'Log:' "$LOGFILE")"
} | tee "$LOGFILE"

# ---------------------------------------------------- run
python -m dreamerv3.main_htp --configs $CONFIGS --task "$TASK" --seed "$SEED" 2>&1 | tee -a "$LOGFILE"
RC=${PIPESTATUS[0]}

END=$(date '+%Y-%m-%d %H:%M:%S')
{
  banner "END" "$END" "$(printf '  %-8s %s\n' 'Exit:' "$RC")"
} | tee -a "$LOGFILE"

# One-liner summary at the very end for grep-friendliness:
echo "TRACKER_ROW: run=$RUN_NAME  start=\"$START\"  end=\"$END\"  exit=$RC" | tee -a "$LOGFILE"

exit "$RC"