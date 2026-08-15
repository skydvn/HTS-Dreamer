#!/usr/bin/env bash
# run_warmup_slim.sh — two-phase training for Slim-HTP variants.
#
# Phase 1: PURE naive DreamerV3 for WARMUP_STEPS env steps.
# Phase 2: Slim-HTP (or Slim-HTP + VICReg), resumes phase-1 checkpoint,
#          trains to TOTAL_STEPS.
#
# This differs from run_warmup_arm.sh in a crucial way:
#
#   Original HTP arms (htp_full, htp_separable, htp_joint, ...) need phase 1
#   to be htp_empty_projection because the head input dimensionality changes
#   between naive Dreamer (h_t = feat_dim) and HTP-with-projection (z_t = 2048).
#   Phase 1 must build the projection to keep head shapes compatible.
#
#   Slim-HTP has NO projection. Head input dimensionality is feat_dim in both
#   naive Dreamer AND Slim-HTP. So phase 1 can be pure naive Dreamer —
#   checkpoint transfers cleanly, and the only "new" module in phase 2 is
#   htp_pdyn (freshly initialized via from_checkpoint_regex='.*').
#
# This is the honest "naive Dreamer warmup then HTP" recipe the user
# originally asked about — enabled only because Slim-HTP dropped the
# projection.
#
# Both phases log to the SAME W&B run via WANDB_RUN_ID + WANDB_RESUME=allow,
# tagged phase1_naive / phase2_<arm> under Job Type so you can filter in the UI.
#
# Usage:
#   ./run_warmup_slim.sh <arm> <seed> [<task>] [<warmup_steps>] [<total_steps>] [<size>]
#
# Arms accepted for phase 2:
#   htp_slim           - Slim-HTP (default)
#   htp_slim_vicreg    - Slim-HTP + VICReg anti-collapse regularizer
#
# NOT accepted (use run_warmup_arm.sh or run_arm.sh):
#   baseline           - use run_arm.sh; no HTP to warm up to
#   htp_slim_postHoc   - sanity check; warmup adds no info because grad_to_backbone=false
#   htp_full, htp_*    - use run_warmup_arm.sh; head-dim mismatch requires scaffold warmup
#
# Examples:
#   ./run_warmup_slim.sh htp_slim 0
#   ./run_warmup_slim.sh htp_slim_vicreg 1
#   ./run_warmup_slim.sh htp_slim 0 atari100k_pong
#   ./run_warmup_slim.sh htp_slim 0 atari100k_breakout 60000 110000
#   ./run_warmup_slim.sh htp_slim 0 atari100k_breakout 40000 110000 size12m

set -euo pipefail

ARM=${1:?arm required (htp_slim | htp_slim_vicreg)}
SEED=${2:?seed required (e.g. 0, 1, 2)}
TASK=${3:-atari100k_breakout}
WARMUP_STEPS=${4:-40000}
TOTAL_STEPS=${5:-110000}
SIZE=${6:-size25m}

# ---------------------------------------------------- arm -> phase-2 --configs
case "$ARM" in
  htp_slim)         P2_CONFIGS="htp_atari100k htp_slim $SIZE wandb" ;;
  htp_slim_vicreg)  P2_CONFIGS="htp_atari100k htp_slim_vicreg $SIZE wandb" ;;
  baseline)
    echo "ERROR: 'baseline' has no HTP to turn on. Use run_arm.sh for pure naive runs." >&2
    exit 2
    ;;
  htp_slim_postHoc)
    echo "ERROR: 'htp_slim_postHoc' is a sanity-check arm (grad_to_backbone=false, no HTP effect on control)." >&2
    echo "       Warmup adds no analytical value here. Use run_arm.sh for from-scratch." >&2
    exit 2
    ;;
  htp_full|htp_empty_projection|htp_recon_only|htp_pdyn_only|htp_separable|htp_joint|htp_reverse_strides)
    echo "ERROR: '$ARM' uses a projection (head input dim = 2048)." >&2
    echo "       Pure naive Dreamer warmup would cause a head-shape mismatch at phase 2." >&2
    echo "       Use run_warmup_arm.sh instead — it warms up with htp_empty_projection scaffold." >&2
    exit 2
    ;;
  *)
    echo "ERROR: unknown arm '$ARM'" >&2
    echo "Valid: htp_slim, htp_slim_vicreg" >&2
    exit 2
    ;;
esac

# Phase 1 is always PURE naive DreamerV3 for Slim-HTP warmup.
# This works because Slim-HTP doesn't add a projection, so head input dims
# (feat_dim) match between phases.
P1_CONFIGS="atari100k $SIZE wandb"

# ---------------------------------------------------- shared identity
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="warmup_slim_${ARM}_${TASK}_seed${SEED}_${STAMP}"
LOGDIR="/root/logdir/${RUN_ID}"

export WANDB_PROJECT="${WANDB_PROJECT:-HTS-Dreamer}"
export WANDB_GROUP="${WANDB_GROUP:-slim-warmup-ablation}"
export WANDB_RUN_ID="$RUN_ID"        # both phases append to one run
export WANDB_RESUME=allow            # permits phase 2 to attach
export WANDB_RUN_NAME="$RUN_ID"
export WANDB_TAGS="warmup-slim,${ARM},${TASK},seed${SEED},${SIZE}"

mkdir -p "$LOGDIR"
mkdir -p runlogs
MASTER_LOG="runlogs/${RUN_ID}.log"

echo "======================================================================" | tee "$MASTER_LOG"
echo "Run ID:       $RUN_ID"                                                    | tee -a "$MASTER_LOG"
echo "Logdir:       $LOGDIR"                                                    | tee -a "$MASTER_LOG"
echo "Arm:          $ARM"                                                       | tee -a "$MASTER_LOG"
echo "Task:         $TASK"                                                      | tee -a "$MASTER_LOG"
echo "Seed:         $SEED"                                                      | tee -a "$MASTER_LOG"
echo "Warmup:       $WARMUP_STEPS env steps (PURE naive DreamerV3)"             | tee -a "$MASTER_LOG"
echo "Then:         to $TOTAL_STEPS total steps with $ARM"                      | tee -a "$MASTER_LOG"
echo "Size:         $SIZE"                                                      | tee -a "$MASTER_LOG"
echo "P1 configs:   $P1_CONFIGS"                                                | tee -a "$MASTER_LOG"
echo "P2 configs:   $P2_CONFIGS"                                                | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"

# ---------------------------------------------------- Phase 1: pure naive Dreamer
export WANDB_JOB_TYPE=phase1_naive
P1_START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$P1_START] PHASE 1 START — pure naive DreamerV3 to step $WARMUP_STEPS" | tee -a "$MASTER_LOG"

python -m dreamerv3.main_htp \
  --configs $P1_CONFIGS \
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

# ---------------------------------------------------- Phase 2: Slim-HTP
export WANDB_JOB_TYPE="phase2_${ARM}"
P2_START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$P2_START] PHASE 2 START — $ARM, resuming from checkpoint, to step $TOTAL_STEPS" | tee -a "$MASTER_LOG"

python -m dreamerv3.main_htp \
  --configs $P2_CONFIGS \
  --task "$TASK" \
  --seed "$SEED" \
  --logdir "$LOGDIR" \
  --run.from_checkpoint "$LOGDIR/checkpoint_phase1.ckpt" \
  --run.from_checkpoint_regex '.*' \
  --run.steps "$TOTAL_STEPS" 2>&1 | tee -a "$MASTER_LOG"

P2_END=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$P2_END] PHASE 2 END" | tee -a "$MASTER_LOG"

echo "======================================================================" | tee -a "$MASTER_LOG"
echo "TRACKER_ROW: run=$RUN_ID arm=$ARM p1_start=\"$P1_START\" p1_end=\"$P1_END\" p2_start=\"$P2_START\" p2_end=\"$P2_END\"" | tee -a "$MASTER_LOG"