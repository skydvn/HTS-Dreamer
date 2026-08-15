#!/usr/bin/env bash
# run_warmup_arm.sh — two-phase training: naive DreamerV3 warmup, then any HTP arm.
#
# Phase 1: naive DreamerV3 for WARMUP_STEPS env steps.
# Phase 2: resume the checkpoint with the chosen HTP arm, train to TOTAL_STEPS.
#
# Both phases log to the SAME W&B run via WANDB_RUN_ID + WANDB_RESUME=allow,
# tagged phase1_naive / phase2_<arm> under Job Type so you can filter in the UI.
#
# Usage:
#   ./run_warmup_arm.sh <arm> <seed> [<task>] [<warmup_steps>] [<total_steps>] [<size>]
#
# Arms accepted for phase 2 (must be an HTP arm, not baseline):
#   htp_full  htp_empty_projection  htp_recon_only  htp_pdyn_only
#   htp_separable  htp_joint  htp_reverse_strides
#
# Examples:
#   ./run_warmup_arm.sh htp_full 0
#   ./run_warmup_arm.sh htp_separable 1 atari100k_pong
#   ./run_warmup_arm.sh htp_full 0 atari100k_breakout 60000 110000
#   ./run_warmup_arm.sh htp_recon_only 0 atari100k_breakout 40000 110000 size12m

set -euo pipefail

ARM=${1:?arm required (htp_full | htp_empty_projection | htp_recon_only | htp_pdyn_only | htp_separable | htp_joint | htp_reverse_strides)}
SEED=${2:?seed required (e.g. 0, 1, 2)}
TASK=${3:-atari100k_breakout}
WARMUP_STEPS=${4:-40000}
TOTAL_STEPS=${5:-110000}
SIZE=${6:-size25m}

# ---------------------------------------------------- arm -> phase-2 --configs
case "$ARM" in
  htp_full)              P2_CONFIGS="htp_atari100k $SIZE wandb" ;;
  htp_empty_projection)  P2_CONFIGS="htp_atari100k htp_empty_projection $SIZE wandb" ;;
  htp_recon_only)        P2_CONFIGS="htp_atari100k htp_matryoshka_only $SIZE wandb" ;;
  htp_pdyn_only)         P2_CONFIGS="htp_atari100k htp_pdyn_only $SIZE wandb" ;;
  htp_separable)         P2_CONFIGS="htp_atari100k htp_separable $SIZE wandb" ;;
  htp_joint)             P2_CONFIGS="htp_atari100k htp_joint $SIZE wandb" ;;
  htp_reverse_strides)   P2_CONFIGS="htp_atari100k htp_reverse_strides $SIZE wandb" ;;
  baseline)
    echo "ERROR: 'baseline' has no HTP to turn on. Use run_arm.sh for pure naive runs." >&2
    exit 2
    ;;
  *)
    echo "ERROR: unknown arm '$ARM'" >&2
    echo "Valid: htp_full, htp_empty_projection, htp_recon_only, htp_pdyn_only, htp_separable, htp_joint, htp_reverse_strides" >&2
    exit 2
    ;;
esac

# ---------------------------------------------------- shared identity
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="warmup_${ARM}_${TASK}_seed${SEED}_${STAMP}"
LOGDIR="/root/logdir/${RUN_ID}"

export WANDB_PROJECT="${WANDB_PROJECT:-HTS-Dreamer}"
export WANDB_GROUP="${WANDB_GROUP:-warmup-ablation}"
export WANDB_RUN_ID="$RUN_ID"           # ← both phases append to one run
export WANDB_RESUME=allow               # ← permits phase 2 to attach
export WANDB_RUN_NAME="$RUN_ID"
export WANDB_TAGS="warmup,${ARM},${TASK},seed${SEED},${SIZE}"

mkdir -p "$LOGDIR"
mkdir -p runlogs
MASTER_LOG="runlogs/${RUN_ID}.log"

echo "======================================================================" | tee "$MASTER_LOG"
echo "Run ID:       $RUN_ID"                                                    | tee -a "$MASTER_LOG"
echo "Logdir:       $LOGDIR"                                                    | tee -a "$MASTER_LOG"
echo "Arm:          $ARM"                                                       | tee -a "$MASTER_LOG"
echo "Task:         $TASK"                                                      | tee -a "$MASTER_LOG"
echo "Seed:         $SEED"                                                      | tee -a "$MASTER_LOG"
echo "Warmup:       $WARMUP_STEPS env steps (naive DreamerV3)"                  | tee -a "$MASTER_LOG"
echo "Then:         to $TOTAL_STEPS total steps with $ARM"                      | tee -a "$MASTER_LOG"
echo "Size:         $SIZE"                                                      | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"

# ---------------------------------------------------- Phase 1: naive
export WANDB_JOB_TYPE=phase1_naive
P1_START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$P1_START] PHASE 1 START — naive DreamerV3 to step $WARMUP_STEPS" | tee -a "$MASTER_LOG"

python -m dreamerv3.main_htp \
  --configs atari100k "$SIZE" wandb \
  --task "$TASK" \
  --seed "$SEED" \
  --logdir "$LOGDIR" \
  --run.steps "$WARMUP_STEPS" 2>&1 | tee -a "$MASTER_LOG"

P1_END=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$P1_END] PHASE 1 END" | tee -a "$MASTER_LOG"

# Snapshot the phase-1 checkpoint so nothing overwrites it.
if [ -f "$LOGDIR/checkpoint.ckpt" ]; then
    cp "$LOGDIR/checkpoint.ckpt" "$LOGDIR/checkpoint_phase1.ckpt"
    echo "Saved phase-1 snapshot: $LOGDIR/checkpoint_phase1.ckpt" | tee -a "$MASTER_LOG"
else
    echo "WARNING: no checkpoint.ckpt found after phase 1 — phase 2 will start fresh." | tee -a "$MASTER_LOG"
fi

# ---------------------------------------------------- Phase 2: HTP arm
export WANDB_JOB_TYPE="phase2_${ARM}"
P2_START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$P2_START] PHASE 2 START — $ARM, resuming from checkpoint, to step $TOTAL_STEPS" | tee -a "$MASTER_LOG"

python -m dreamerv3.main_htp \
  --configs $P2_CONFIGS \
  --task "$TASK" \
  --seed "$SEED" \
  --logdir "$LOGDIR" \
  --run.from_checkpoint "$LOGDIR/checkpoint_phase1.ckpt" \
  --run.steps "$TOTAL_STEPS" 2>&1 | tee -a "$MASTER_LOG"

P2_END=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$P2_END] PHASE 2 END" | tee -a "$MASTER_LOG"

echo "======================================================================" | tee -a "$MASTER_LOG"
echo "TRACKER_ROW: run=$RUN_ID arm=$ARM p1_start=\"$P1_START\" p1_end=\"$P1_END\" p2_start=\"$P2_START\" p2_end=\"$P2_END\"" | tee -a "$MASTER_LOG"