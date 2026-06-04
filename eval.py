#!/usr/bin/env python3
"""OpAI-Bench detection — Hydra entry point.

Loads the OpAI-Bench dataset from HuggingFace, runs the requested
detector via the shared pipeline interface, and writes per-document
predictions plus aggregate metrics.

Usage:
    uv run python eval.py detector=e5-small
    uv run python eval.py detector=fast-detectgpt max_samples=100
    uv run python eval.py detector=adaloc dataset.split=dev \\
        detector.overrides.checkpoint_path=/path/to/epoch-best.pkl
    uv run python eval.py -m detector=e5-small,desklib max_samples=50
"""

from __future__ import annotations

import ast
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


def _load_cfg(cli_overrides: list[str]) -> DictConfig:
    """Load Hydra config via the compose API.

    Avoids @hydra.main, which on Python 3.14 trips an argparse-strictness
    bug in hydra 1.3.2's shell-completion help string. compose() doesn't
    touch argparse so it works on 3.14.
    """
    conf_dir = str((REPO_ROOT / "conf").resolve())
    with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
        cfg = compose(config_name="config", overrides=cli_overrides)
    return cfg


def _resolve(p: str) -> Path:
    """Resolve a config path: absolute -> as-is, relative -> against REPO_ROOT."""
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp).resolve()


def _sanitize(v):
    """Make a value JSON-serializable."""
    import numpy as np
    if v is None:
        return None
    if isinstance(v, (str, int, bool)):
        return v
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return _sanitize(float(v))
    if isinstance(v, (np.ndarray,)):
        return [_sanitize(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {k: _sanitize(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize(x) for x in v]
    return str(v)


def _doc_label_from_tok_labels(tok_labels_str) -> int:
    """Binary doc label: 1 if any AI token present, 0 otherwise."""
    try:
        tl = ast.literal_eval(str(tok_labels_str))
        return 1 if (tl and sum(tl) > 0) else 0
    except (ValueError, SyntaxError):
        return 0


def main(cfg: DictConfig):
    print("[eval] config:")
    print(OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------------
    # Step 1: pick the source CSV.
    #   (a) cfg.local_csv set  -> use it directly (bypass HuggingFace).
    #   (b) cfg.local_csv null -> download OpAI-Bench from HF Hub.
    # ------------------------------------------------------------------
    csv_dir = _resolve(cfg.prepared_csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    hf_cache = _resolve(cfg.hf_cache)

    if cfg.get("local_csv"):
        local = cfg.local_csv
        if isinstance(local, str):
            csv_path = _resolve(local)
            assert csv_path.exists(), f"local_csv not found: {csv_path}"
            print(f"[eval] using local CSV (bypass HF): {csv_path}")
        else:
            paths = [_resolve(p) for p in local]
            for p in paths:
                assert p.exists(), f"local_csv not found: {p}"
            import pandas as pd
            print(f"[eval] concatenating {len(paths)} local CSVs (bypass HF)")
            df_all = pd.concat([pd.read_csv(p, low_memory=False)
                                for p in paths], ignore_index=True)
            csv_path = csv_dir / ("local_concat_"
                                  + "_".join(p.stem for p in paths) + ".csv")
            df_all.to_csv(csv_path, index=False)
            print(f"[eval] wrote concatenated CSV: {csv_path}")
        ds_config = cfg.dataset.get("config", "local")
    else:
        from opai_bench.data_loader import load_opai_bench_as_csv
        hf_repo = cfg.dataset.get("hf_repo")
        if not hf_repo or str(hf_repo).strip() == "???":
            raise ValueError(
                "dataset.hf_repo is not set. Either edit "
                "conf/dataset/opai_bench.yaml and replace `???` with the "
                "HuggingFace dataset repo ID, or pass it on the CLI: "
                "`dataset.hf_repo=<org>/<repo>`. Alternatively, switch to "
                "a local CSV via `dataset=local_csv` and the LOCAL_CSV_DIR "
                "environment variable."
            )
        ds_config = cfg.dataset.get("config", "default")
        csv_name = f"{ds_config}_{cfg.dataset.split}"
        if cfg.max_samples:
            csv_name += f"_n{cfg.max_samples}"
        csv_name += ".csv"
        csv_path = csv_dir / csv_name
        force_rebuild = bool(cfg.get("force_rebuild", False))
        if force_rebuild or not csv_path.exists():
            load_opai_bench_as_csv(
                out_path=csv_path,
                hf_repo=hf_repo,
                config=ds_config,
                split=cfg.dataset.split,
                max_samples=cfg.max_samples,
                cache_dir=str(hf_cache),
            )
        else:
            print(f"[eval] reusing cached CSV: {csv_path}")

    # ------------------------------------------------------------------
    # Step 2: load + compute ground-truth doc labels.
    # ------------------------------------------------------------------
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"] == cfg.dataset.split].reset_index(drop=True)
    if cfg.max_samples:
        df = df.head(cfg.max_samples).reset_index(drop=True)
    df["_doc_label"] = df["tok_labels"].apply(_doc_label_from_tok_labels)
    print(f"[eval] loaded {len(df)} rows; pos={int(df['_doc_label'].sum())} "
          f"neg={int((1 - df['_doc_label']).sum())}")

    # ------------------------------------------------------------------
    # Step 3: build the detector.
    # ------------------------------------------------------------------
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))

    from opai_bench_detectors.core import pipeline

    overrides = OmegaConf.to_container(cfg.detector.overrides, resolve=True) or {}

    # Detect unfilled `???` placeholders (used in YAML to mark required
    # local checkpoints) and fail fast with a helpful message.
    for k, v in list(overrides.items()):
        if isinstance(v, str) and v.strip() == "???":
            raise ValueError(
                f"Detector '{cfg.detector.name}' requires you to set "
                f"`detector.overrides.{k}`. Edit "
                f"conf/detector/{cfg.detector.name}.yaml or pass it on the "
                f"CLI: detector.overrides.{k}=/path/to/checkpoint"
            )

    print(f"[eval] loading detector '{cfg.detector.name}' overrides={overrides}")
    t_load = time.time()
    pipe = pipeline("ai-text-detection",
                    model=cfg.detector.name,
                    device=cfg.device,
                    **overrides)
    load_seconds = time.time() - t_load
    print(f"[eval] detector loaded in {load_seconds:.1f}s")

    # ------------------------------------------------------------------
    # Step 4: run detection.
    # ------------------------------------------------------------------
    predictions = []
    n_errors = 0
    t_score = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_score
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{cfg.detector.name}] {i+1}/{len(df)} ({rate:.1f} docs/s)")

        out = {
            "essay_id": _sanitize(row.get("essay_id")),
            "split": _sanitize(row.get("split")),
            "domain": _sanitize(row.get("domain")),
            "ai_model": _sanitize(row.get("ai_model")),
            "version": _sanitize(row.get("version")),
            "operation": _sanitize(row.get("operation")),
            "AI_token_ratio": _sanitize(row.get("AI_token_ratio")),
            "detection_gt_label": int(row["_doc_label"]),
        }
        try:
            result = pipe(row["text_clean"])
            out["detection_label"] = int(result["label"])
            out["detection_score"] = _sanitize(result.get("score"))
            out["detection_metadata"] = _sanitize(result.get("metadata", {}))
            out["detection_error"] = None
        except Exception as e:
            out["detection_label"] = 0
            out["detection_score"] = 0.0
            out["detection_metadata"] = {}
            out["detection_error"] = f"{type(e).__name__}: {e}"
            n_errors += 1

        predictions.append(out)

    score_seconds = time.time() - t_score
    print(f"[eval] {cfg.detector.name} scored {len(df)} docs in "
          f"{score_seconds:.1f}s ({n_errors} errors)")

    # ------------------------------------------------------------------
    # Step 5: metrics.
    # ------------------------------------------------------------------
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    valid = [p for p in predictions if p["detection_error"] is None]
    y_true = [p["detection_gt_label"] for p in valid]
    y_pred = [p["detection_label"] for p in valid]
    y_score = [p["detection_score"] if p["detection_score"] is not None else 0.0
               for p in valid]

    summary = {
        "detector": cfg.detector.name,
        "dataset": ds_config,
        "split": cfg.dataset.split,
        "n_total": len(predictions),
        "n_valid": len(valid),
        "n_errors": n_errors,
        "load_seconds": round(load_seconds, 2),
        "score_seconds": round(score_seconds, 2),
    }
    if y_true:
        summary["accuracy"] = float(accuracy_score(y_true, y_pred))
        summary["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        summary["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        summary["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
            summary["confusion_matrix"] = {"tn": cm[0][0], "fp": cm[0][1],
                                           "fn": cm[1][0], "tp": cm[1][1]}
        except Exception:
            pass
        if len(set(y_true)) > 1:
            try:
                summary["auroc"] = float(roc_auc_score(y_true, y_score))
                summary["auprc"] = float(average_precision_score(y_true, y_score))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Step 6: write outputs.
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = cfg.detector.name.replace("/", "-")
    output_dir = _resolve(cfg.output_dir)
    run_dir = output_dir / f"{safe_name}_{ds_config}_{cfg.dataset.split}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "predictions.jsonl", "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    print(f"\n[eval] DONE → {run_dir}")
    print(f"[eval] summary: {json.dumps({k: v for k, v in summary.items() if k != 'confusion_matrix'}, indent=2)}")

    try:
        pipe.cleanup()
    except Exception:
        pass
    gc.collect()


if __name__ == "__main__":
    # Hydra-style CLI overrides: every CLI arg is a dot-path override.
    cli = sys.argv[1:]
    cfg = _load_cfg(cli)
    main(cfg)
