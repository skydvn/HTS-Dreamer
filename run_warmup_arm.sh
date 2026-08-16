#!/usr/bin/env bash
# run_warmup_arm.sh — two-phase training: HTP scaffold warmup, then full HTP arm.
#
# Phase 1: HTP with both auxiliary loss scales set to 0 (htp_empty_projection).
#          Same module tree and head input dims as phase 2, so the checkpoint
#          transfers cleanly. During phase 1 the projection S_ψ is trained
#          purely by gradients from the actor/critic/reward/continuation heads.
# Phase 2: Same architecture, HTP auxiliary losses turned ON (the chosen arm).
#
# NOTE: phase 1 is NOT pure naive DreamerV3 — that would fail because naive
# Dreamer heads consume h_t (dim = feat_dim) whereas HTP heads consume
# z_t = htp_proj(h_t) (dim = proj.dims[-1]). To warm up from pure Dreamer,
# you'd need a code change to rebuild the heads at phase 2 (not currently
# supported). This scaffold-warmup approach matches the paper's App. A.3
# "temporal refinement" regime: the architecture is fixed from the start;
# only the auxiliary supervision is switched on later.
#
# Both phases log to the SAME W&B run via WANDB_RUN_ID + WANDB_RESUME=allow,
# tagged phase1_scaffold / phase2_<arm> under Job Type so you can filter.
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
echo "Warmup:       $WARMUP_STEPS env steps (HTP scaffold; aux losses OFF)"      | tee -a "$MASTER_LOG"
echo "Then:         to $TOTAL_STEPS total steps with $ARM (aux losses ON)"      | tee -a "$MASTER_LOG"
echo "Size:         $SIZE"                                                      | tee -a "$MASTER_LOG"
echo "======================================================================" | tee -a "$MASTER_LOG"

# ---------------------------------------------------- Phase 1: HTP scaffold, aux losses OFF
# We use htp_empty_projection (htp.enabled=true, both loss scales=0) rather than pure
# naive Dreamer because the head input dimensions and module tree must match phase 2:
#   * Naive Dreamer heads consume raw h_t (dim 10240).
#   * HTP heads consume z_t = htp_proj(h_t) (dim 2048 with default dims).
# If phase 1 were naive, phase 2 would crash on the first forward pass (shape mismatch)
# and the checkpoint would be missing the htp_* module keys entirely.
export WANDB_JOB_TYPE=phase1_scaffold
P1_START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$P1_START] PHASE 1 START — HTP scaffold (aux losses OFF) to step $WARMUP_STEPS" | tee -a "$MASTER_LOG"

python -m dreamerv3.main_htp \
  --configs htp_atari100k htp_empty_projection "$SIZE" wandb \
  --task "$TASK" \
  --seed "$SEED" \
  --logdir "$LOGDIR" \
  --run.steps "$WARMUP_STEPS" \
  --run.save_every 60 2>&1 | tee -a "$MASTER_LOG"

P1_END=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$P1_END] PHASE 1 END" | tee -a "$MASTER_LOG"

# Snapshot the phase-1 checkpoint DIRECTORY so phase 2's own saves don't
# overwrite it. Embodied writes checkpoints as `<logdir>/ckpt/<timestamp>/`
# with a `latest` pointer file at the top level. --run.from_checkpoint expects
# the path to the ckpt/-style directory (embodied reads `latest` inside).
if [ -d "$LOGDIR/ckpt" ] && [ -f "$LOGDIR/ckpt/latest" ]; then
    cp -r "$LOGDIR/ckpt" "$LOGDIR/ckpt_phase1"
    echo "Saved phase-1 snapshot: $LOGDIR/ckpt_phase1" | tee -a "$MASTER_LOG"
    echo "  Latest checkpoint marker points to:" | tee -a "$MASTER_LOG"
    echo "    $(cat "$LOGDIR/ckpt_phase1/latest" 2>/dev/null || echo '<unreadable>')" | tee -a "$MASTER_LOG"
elif [ -d "$LOGDIR/ckpt" ]; then
    echo "WARNING: $LOGDIR/ckpt/ exists but has no 'latest' pointer." | tee -a "$MASTER_LOG"
    echo "         Phase 1 may have crashed mid-save. Copying anyway." | tee -a "$MASTER_LOG"
    cp -r "$LOGDIR/ckpt" "$LOGDIR/ckpt_phase1"
elif [ -f "$LOGDIR/checkpoint.ckpt" ]; then
    cp "$LOGDIR/checkpoint.ckpt" "$LOGDIR/checkpoint_phase1.ckpt"
    echo "Saved phase-1 snapshot from checkpoint.ckpt (legacy format)" | tee -a "$MASTER_LOG"
else
    echo "ERROR: no phase-1 checkpoint found in $LOGDIR after phase 1." | tee -a "$MASTER_LOG"
    echo "       Directory contents:" | tee -a "$MASTER_LOG"
    ls -la "$LOGDIR" 2>&1 | tee -a "$MASTER_LOG"
    echo "       Phase 2 aborted." | tee -a "$MASTER_LOG"
    exit 3
fi

if [ -d "$LOGDIR/ckpt_phase1" ]; then
    PHASE1_CKPT="$LOGDIR/ckpt_phase1"
elif [ -f "$LOGDIR/checkpoint_phase1.ckpt" ]; then
    PHASE1_CKPT="$LOGDIR/checkpoint_phase1.ckpt"
else
    echo "ERROR: neither ckpt_phase1/ nor checkpoint_phase1.ckpt exists — aborting." | tee -a "$MASTER_LOG"
    exit 3
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
  --run.from_checkpoint "$PHASE1_CKPT" \
  --run.from_checkpoint_regex '.*' \
  --run.steps "$TOTAL_STEPS" 2>&1 | tee -a "$MASTER_LOG"

P2_END=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$P2_END] PHASE 2 END" | tee -a "$MASTER_LOG"

echo "======================================================================" | tee -a "$MASTER_LOG"
echo "TRACKER_ROW: run=$RUN_ID arm=$ARM p1_start=\"$P1_START\" p1_end=\"$P1_END\" p2_start=\"$P2_START\" p2_end=\"$P2_END\"" | tee -a "$MASTER_LOG"