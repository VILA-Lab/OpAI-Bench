"""Adapter: OpAI-Bench HF dataset → flat CSV consumed by eval.py.

eval.py loads detection inputs as a single CSV with these columns:

    text_clean / tokens / tok_labels / sentences / sent_labels /
    split / ai_model / version / operation / AI_token_ratio /
    AI_sent_ratio / AI_char_ratio / ai_spans_char / ai_spans_token /
    domain / essay_id

This module pulls the OpAI-Bench dataset from the HuggingFace Hub and
writes a CSV that matches that schema. List-valued columns are
serialized via `repr(list(v))` so that downstream readers can recover
them with `ast.literal_eval`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset


# OpAI-Bench column → flat-CSV column. Pure renames, no value transforms.
RENAME_MAP = {
    "text": "text_clean",
    "token_labels": "tok_labels",
    "sentence_labels": "sent_labels",
    "ai_token_ratio": "AI_token_ratio",
    "ai_sentence_ratio": "AI_sent_ratio",
    "ai_char_ratio": "AI_char_ratio",
    "edit_operation": "operation",
    "generator": "ai_model",
}

# Columns that arrive as Python lists (or strings of lists) and must be
# round-trippable through ast.literal_eval after to_csv.
LIST_COLS = ["tokens", "tok_labels", "sentences", "sent_labels",
             "ai_spans_char", "ai_spans_token"]


def load_opai_bench_as_csv(
    out_path: str | Path,
    hf_repo: str,
    config: str = "default",
    split: str = "test",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """Load OpAI-Bench from HuggingFace and write a detector-friendly CSV.

    Args:
        out_path: where to write the CSV.
        hf_repo: HuggingFace Hub repo ID for the dataset, e.g.
            "<org>/<dataset>". Passed verbatim to `datasets.load_dataset`.
        config: HF dataset config; one of {"default", "ablations"}.
        split: HF split; one of {"train", "dev", "test"}.
        max_samples: if set, truncate to first N rows after loading.
        cache_dir: HuggingFace datasets cache directory. If None, uses
            $HF_HOME or the default location.

    Returns:
        Absolute path of the CSV that was written.
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_dir is not None:
        os.environ.setdefault("HF_HOME", str(cache_dir))

    print(f"[opai_bench] loading {hf_repo} ({config}/{split}) "
          f"from HuggingFace ...", flush=True)
    ds = load_dataset(hf_repo, config, split=split, cache_dir=cache_dir)
    if max_samples is not None and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    df: pd.DataFrame = ds.to_pandas()
    print(f"[opai_bench] {len(df)} rows loaded; columns={list(df.columns)}",
          flush=True)

    # Rename columns that have direct counterparts.
    df = df.rename(columns=RENAME_MAP)

    # Some downstream code prefers `model_used` over `ai_model`; emit both.
    if "ai_model" in df.columns:
        df["model_used"] = df["ai_model"]

    # `essay_id` is the stable per-document ID expected by token/
    # sentence-level evaluators. OpAI-Bench exposes it as `record_id`.
    if "record_id" in df.columns and "essay_id" not in df.columns:
        df["essay_id"] = df["record_id"].astype(str)

    # Serialize list columns as Python literals so ast.literal_eval can
    # reconstruct them after pd.read_csv.
    for col in LIST_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: repr(list(v)) if v is not None
                                    else "[]")

    df.to_csv(out_path, index=False)
    print(f"[opai_bench] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)",
          flush=True)
    return out_path


