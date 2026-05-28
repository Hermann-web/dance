#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-}"
shift || true
RECORDS=("$@")

if [[ -z "${ROOT_DIR}" ]]; then
  echo "Usage: $0 <ludb_root> <record1> [record2 ...]" >&2
  exit 1
fi
if [[ ${#RECORDS[@]} -eq 0 ]]; then
  echo "Provide at least one LUDB record stem." >&2
  exit 1
fi

command -v dance >/dev/null 2>&1 || { echo "dance CLI not found in PATH."; exit 1; }

for r in "${RECORDS[@]}"; do
  [[ -f "${ROOT_DIR}/${r}.hea" ]] || { echo "Missing ${ROOT_DIR}/${r}.hea"; exit 1; }
  [[ -f "${ROOT_DIR}/${r}.dat" ]] || { echo "Missing ${ROOT_DIR}/${r}.dat"; exit 1; }
  [[ -f "${ROOT_DIR}/${r}.atr" ]] || { echo "Missing ${ROOT_DIR}/${r}.atr"; exit 1; }
done

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/ecg/ludb/${STAMP}"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/train.log"

echo "Running LUDB training. Logs: ${LOG}"
dance ecg-ludb-train \
  --root "${ROOT_DIR}" \
  --records "${RECORDS[@]}" \
  --lead 0 \
  --epochs 5 \
  --batch-size 8 \
  --lr 1e-3 \
  --duration 4.0 \
  --stride 2.0 \
  --n-queries 64 \
  --device cpu | tee "${LOG}"

echo "Done. Results in ${OUT_DIR}"
