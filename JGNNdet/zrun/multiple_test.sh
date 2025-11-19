#!/usr/bin/env bash

RUNS=(
  "sourcescripts/storage/archive/p_284"
  "sourcescripts/storage/archive/p_664"
  "sourcescripts/storage/archive/p_691"
  "sourcescripts/storage/archive/p_693"
  "sourcescripts/storage/archive/p_703"
  "sourcescripts/storage/archive/p_707"
)

GTYPE="pdg+raw"
SPLITS="default"

for ROOT in "${RUNS[@]}"; do
  CKPT="$ROOT/checkpoints/model-checkpoint.ckpt"     
  OUT="$ROOT/outputs"

  python3 -B sourcescripts/test.py \
    --root "$ROOT" \
    --checkpoint "$CKPT" \
    --out "$OUT" \
    --gtype "$GTYPE" \
    --splits "$SPLITS"
done
