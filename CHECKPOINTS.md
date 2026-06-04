# Detector checkpoints

How to obtain weights for each of the 17 detectors evaluated by
`eval.py`.

Twelve detectors pull their weights from the HuggingFace Hub on first
use and require no manual setup. **Five detectors load weights from
local files** and ship with `checkpoint_path: ???` placeholders in
`conf/detector/<slug>.yaml` — fill them in via one of the routes below
before evaluating.

---

## HuggingFace Hub (no manual download)

These twelve detectors fetch their weights automatically from the
HuggingFace Hub the first time `eval.py` runs them. The cache is
controlled by the `hf_cache` config field (default: `./cache/hf_cache`).

| Detector slug    | HF repo                                              |
|------------------|------------------------------------------------------|
| `e5-small`       | `MayZhou/e5-small-lora-ai-generated-detector`        |
| `desklib`        | `desklib/ai-text-detector-v1.01`                     |
| `radar`          | `TrustSafeAI/RADAR-Vicuna-7B`                        |
| `roberta-openai` | `openai-community/roberta-base-openai-detector`      |
| `roft-boundary`  | `gpt2`                                               |
| `detectllm`      | `gpt2-xl` (or any HF causal LM)                      |
| `gigacheck`      | `iitolstykh/GigaCheck-Detector-Multi`                |
| `damasha`        | `saiteja33/DAMASHA-RMC`                              |
| `binoculars`     | `tiiuae/falcon-7b` + `tiiuae/falcon-7b-instruct`     |
| `dna-detectllm`  | `tiiuae/falcon-7b` + `tiiuae/falcon-7b-instruct`     |
| `fast-detectgpt` | `tiiuae/falcon-7b` + `tiiuae/falcon-7b-instruct`     |
| `seqxgpt`        | `zcahjl3/seqxgpt-detector` (+ four feature LMs)      |
| `mgtd`           | `1-800-SHARED-TASKS/MGTD-Checkpoints` *(gated repo)* |

`mgtd` requires an access grant on the HuggingFace model page before
the cache call will succeed. All other entries are public.

---

## Local checkpoints

Five detectors load weights from a local file. For each, you have two
options: (a) download a checkpoint shipped by the upstream authors if
one is publicly available, or (b) train from scratch using the script
in `training/`.

After obtaining the weights, point the corresponding config at the
file — either edit `conf/detector/<slug>.yaml` directly or pass the
path as a Hydra override at run time:

```bash
uv run python eval.py detector=<slug> \
    detector.overrides.checkpoint_path=/path/to/checkpoint
```

### `adaloc`

- **Architecture.** Sliding-window sentence head over
  `roberta-large-openai-detector` with LoRA on `q/v` (paper-faithful;
  see `baseline/mgt-localization/AdaLoc/roberta_adaloc.py`).
- **Expected file.** `epoch-best.pkl` saved via `torch.save(model, ...)`.
- **Upstream weights.** The original authors released a Google Drive
  checkpoint — follow the link in
  `baseline/mgt-localization/README.md`.
- **Train from scratch.** First convert the OpAI-Bench training split
  into the AdaLoc JSON format expected by
  `baseline/mgt-localization/dataloaders/dataloader.py`, then run

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
      training/train_adaloc.py \
      --train-json ./data/adaloc_json/train_all.json \
      --dev-json ./data/adaloc_json/dev_all.json \
      --out-dir   ./checkpoints/adaloc
  ```

### `sendetex`

- **Architecture.** SenDetEX with a frozen Llama-7B proxy and a
  trainable scoring head; see `baseline/sendetex/SenDetEX.py`.
- **Expected file.** `best.pt` (state-dict for the head + style
  features).
- **Upstream weights.** Not released by the authors at the time of
  writing — you must train from scratch.
- **Train from scratch.**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
      training/train_sendetex.py \
      --data-dir ./data/seqxgpt_bench \
      --out-dir  ./checkpoints/sendetex/seqxgpt_bench
  ```

### `genai-sentence`

- **Architecture.** DeBERTa-v3-base + BiGRU + CRF token tagger with
  LoRA; matches `baseline/genai-detect-sentence/models.py`.
- **Expected file.** `best_model.pt` containing
  `{"model_state_dict", "config", "metrics"}`. The detector adapter
  strips the PEFT/LoRA prefix at load time.
- **Upstream weights.** Not released — train from scratch.
- **Train from scratch.**

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
      training/train_genai_sentence.py \
      --csvs   ./data/csv/{essay,abstract,news,report}.csv \
      --out-dir ./checkpoints/genai-sentence-v2
  ```

### `gl-clic`

- **Architecture.** GL-CLiC sentence-level tagger over
  `microsoft/deberta-v3-base`; see `baseline/gl-clic/scripts/model.py`.
- **Expected file.** A PyTorch Lightning `.ckpt` produced by
  `baseline/gl-clic/train.py`, **or** a simplified state-dict
  `.pt` (this repo's adapter accepts either via `strict=False`).
- **Upstream weights.** Not released — train from scratch.
- **Train from scratch.** The upstream training pipeline is vendored
  under `baseline/gl-clic/`. From inside that directory:

  ```bash
  cd baseline/gl-clic
  python train.py +experiment=gl_clic
  ```

  See `baseline/gl-clic/README.md` for full setup, including its own
  `pyproject.toml` and `uv.lock`.

### `seqxgpt-finetuned`

- **Architecture.** Same SeqXGPT classifier head as the public
  `seqxgpt` slug, but retrained on the OpAI-Bench training split.
- **Expected file.** `seqxgpt_transformer_final.pt` (the small
  classifier head; the four backbone LMs are still pulled from the
  Hub).
- **Upstream weights.** N/A — this slug exists specifically for
  retrained checkpoints.
- **Train from scratch.** SeqXGPT is a two-stage pipeline:

  ```bash
  # 1. Extract per-token log-probability features from the four backbone LMs
  python training/preprocess/extract_seqxgpt_features.py \
      --csv-dir   ./data/csv \
      --out-dir   ./data/seqxgpt_features

  # 2. Train the small classifier on top of the features
  python training/train_seqxgpt_classifier.py \
      --feat-root  ./data/seqxgpt_features \
      --output-dir ./checkpoints/seqxgpt-finetuned
  ```

---

## Optional: re-fine-tune the HuggingFace baselines

`damasha` and `gigacheck` work out of the box from the HuggingFace
Hub, but you can retrain a LoRA adapter on top of them using:

```bash
python training/train_damasha_lora.py --output-dir ./checkpoints/damasha-lora

# `gigacheck` requires the data to be converted to its training JSONL
# format first. The preprocess helper does that conversion.
python training/preprocess/build_gigacheck_jsonl.py \
    --csv-dir ./data/csv --out-dir ./data/gigacheck_jsonl
python training/train_gigacheck_lora.py --output-dir ./checkpoints/gigacheck-lora
```

The resulting checkpoints can be plugged in via the same
`detector.overrides.checkpoint_path` override.
