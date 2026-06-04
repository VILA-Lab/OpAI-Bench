#!/usr/bin/env bash
# Smoke-test every detector on a small slice of one CSV.
# Writes per-detector logs to logs/smoke/<detector>.log, then prints
# a one-line summary per detector (PASS / FAIL).
#
# Configurable via environment variables:
#   LOCAL_CSV    -- local CSV to evaluate on (matching the OpAI-Bench
#                   schema). If unset, the script falls back to the
#                   OpAI-Bench HF dataset.
#   N            -- number of documents to sample (default: 20).
#   DEVICE       -- CUDA device for inference (default: cuda:0).
#   ONLY         -- comma-separated subset of detector slugs to run.
#   HF_HOME      -- HuggingFace cache directory (default: <repo>/cache/hf_cache).

set -u

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
N=${N:-20}
DEVICE=${DEVICE:-cuda:0}
LOCAL_CSV=${LOCAL_CSV:-}

export UV_LINK_MODE=${UV_LINK_MODE:-copy}
export HF_HOME=${HF_HOME:-$REPO/cache/hf_cache}
export TRANSFORMERS_CACHE=$HF_HOME

# Pick up an HF token from the cache file (huggingface-cli login output).
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

mkdir -p "$REPO/logs/smoke"

# Detectors expected to work without extra credentials.
# `mgtd` is GATED on HuggingFace (needs user-level access grant on the
# model repo); add it back via `ONLY=mgtd` once you have approval.
DETECTORS=(
  e5-small
  desklib
  radar
  roberta-openai
  roft-boundary
  detectllm
  gigacheck
  damasha
  binoculars
  dna-detectllm
  fast-detectgpt
  seqxgpt
  adaloc
  sendetex
  genai-sentence
  gl-clic
  seqxgpt-finetuned
)

ONLY=${ONLY:-}
if [[ -n "$ONLY" ]]; then
  IFS=',' read -ra DETECTORS <<< "$ONLY"
fi

cd "$REPO"

# Build the eval CLI override list. Pass local_csv only when the caller
# provided one; otherwise eval.py pulls from HF.
extra_args=()
if [[ -n "$LOCAL_CSV" ]]; then
  extra_args+=("local_csv=$LOCAL_CSV")
fi

for d in "${DETECTORS[@]}"; do
  log="$REPO/logs/smoke/${d}.log"
  echo "=== [$(date +%H:%M:%S)] $d ==="
  PYTHONPATH="$REPO:$REPO/src" uv run python "$REPO/eval.py" \
      detector="$d" max_samples=$N device=$DEVICE \
      "${extra_args[@]}" >"$log" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    # Pull a one-line summary out of the log.
    summary=$(grep -E '"accuracy"|"auroc"|"n_errors"' "$log" | head -3 | tr '\n' ' ')
    echo "  PASS  $d  $summary"
  else
    last_err=$(grep -E '^(Error|Traceback|.*Error:|.*Exception:)' "$log" | tail -1)
    echo "  FAIL  $d  rc=$rc  $last_err  (see $log)"
  fi
done
