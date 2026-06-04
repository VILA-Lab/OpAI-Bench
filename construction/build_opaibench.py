#!/usr/bin/env python3
"""
Build OpAI-Bench-style progressive human-to-AI revision trajectories.

This script constructs v0-v8 cumulative revision trajectories with:
- deterministic sentence selection,
- five edit operations,
- document/sentence/token/span provenance,
- checkpointed CSV output.

Required input CSV columns by default:
- id
- text

Example:
    export LLM_PROVIDER=gemini
    export GEMINI_API_KEY="your_key_here"
    export GEMINI_MODEL="gemini-2.5-flash"

    python construction/build_opaibench.py \
        --input_csv OpAI-Bench1/OpAI-Bench \
        --output_csv outputs/OpAI-Bench.csv \
        --id_column id \
        --text_column text \
        --max_docs 2000 \
        --num_threads 8
"""


import os
import re
import math
import json
import hashlib
import random
import nltk
import requests
import argparse
import traceback
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from multiprocessing.pool import ThreadPool
from tenacity import RetryError, retry, wait_exponential, stop_after_attempt
from google.genai import types
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ============================================================
# Provider selection
# ============================================================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

HF_MODEL = os.getenv("HF_MODEL", "deepseek-ai/DeepSeek-R1")
HF_ROUTER_BASE_URL = os.getenv("HF_ROUTER_BASE_URL", "https://router.huggingface.co/v1")
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "128"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.2"))



SCHEDULE = [
    ("v0", "none",       0.00),
    ("v1", "polish",     0.15),
    ("v2", "paraphrase", 0.25),
    ("v3", "style",      0.40),
    ("v4", "compress",   0.50),
    ("v5", "expand",     0.60),
    ("v6", "style",      0.75),
    ("v7", "paraphrase", 0.90),
    ("v8", "polish",     1.00),
]


# ============================================================
# Utilities
# ============================================================
def version_num(v: str) -> int:
    return int(v[1:])

def stable_seed(*parts) -> int:
    s = "||".join(map(str, parts))
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def deterministic_shuffle(lst, seed):
    r = random.Random(seed)
    out = list(lst)
    r.shuffle(out)
    return out

def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t.strip())
    return t

def split_paragraphs(t: str):
    paras = re.split(r"\n\s*\n+", t)
    return [p.strip() for p in paras if p.strip()]


def split_sentences(paragraph: str):
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    sents = sent_tokenize(paragraph)
    return [s.strip() for s in sents if s.strip()]

def force_one_sentence(text: str) -> str:
    sents = split_sentences(text)
    return sents[0].strip() if sents else text.strip()

def strip_tags(t: str) -> str:
    return t.replace("<AI_Start>", "").replace("</AI_End>", "")

def overlap_len(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def whitespace_tokens_with_offsets(text: str):
    tokens, offsets = [], []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))
    return tokens, offsets

def labels_from_char_spans(offsets, ai_spans):
    return [1 if any(overlap_len(off, sp) > 0 for sp in ai_spans) else 0
            for off in offsets]

def spans_from_labels(labels):
    spans, i, n = [], 0, len(labels)
    while i < n:
        if labels[i] == 1:
            j = i + 1
            while j < n and labels[j] == 1:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans

def whitespace_word_spans(text: str):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

def get_boundary_pattern(tok_labels):
    """Convert token labels to boundary pattern e.g. H, M, HMH, HMHM."""
    if not tok_labels:
        return "H"
    pattern, prev = [], None
    for label in tok_labels:
        symbol = "M" if label == 1 else "H"
        if symbol != prev:
            pattern.append(symbol)
            prev = symbol
    return "".join(pattern)

# ============================================================
# Split and model label helpers
# ============================================================
def assign_split(doc_id, train=0.70, dev=0.15):
    """Deterministic — all versions of same essay stay in same split."""
    h = int(hashlib.sha256(str(doc_id).encode()).hexdigest()[:8], 16)
    r = (h % 1000) / 1000.0
    if r < train:
        return "train"
    elif r < train + dev:
        return "dev"
    else:
        return "test"

def get_model_label() -> str:
    if LLM_PROVIDER == "openai":
        return f"openai/{OPENAI_MODEL}"
    elif LLM_PROVIDER == "deepseek":
        return f"deepseek/{DEEPSEEK_MODEL}"
    elif LLM_PROVIDER == "gemini":
        return f"gemini/{GEMINI_MODEL}"
    elif LLM_PROVIDER in ("hf_router", "hf_inference"):
        return f"{LLM_PROVIDER}/{HF_MODEL}"
    return LLM_PROVIDER

# ============================================================
# Tags -> clean + spans
# ============================================================
def tags_to_clean_and_spans(text_tagged: str):
    i, clean, spans, cur = 0, [], [], None
    while i < len(text_tagged):
        if text_tagged.startswith("<AI_Start>", i):
            i += len("<AI_Start>")
            if cur is None:
                cur = len(clean)
            continue
        if text_tagged.startswith("</AI_End>", i):
            i += len("</AI_End>")
            if cur is not None:
                spans.append((cur, len(clean)))
                cur = None
            continue
        clean.append(text_tagged[i])
        i += 1
    if cur is not None:
        spans.append((cur, len(clean)))
    spans = sorted(spans)
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return "".join(clean), merged

# ============================================================
# Cumulative selection
# ============================================================
def _paras_for_sents(sids, para_to_sents):
    """Return set of paragraph ids containing any of the given sentence ids."""
    sid_set = set(sids)
    return {pid for pid, sids_in_para in para_to_sents.items()
            if any(s in sid_set for s in sids_in_para)}


def select_touched_blocks(doc_id, para_ids, para_to_sents, C_sent, touched_prev=None):
    touched_prev = touched_prev or set()
    all_sids = [sid for pid in para_ids for sid in para_to_sents.get(pid, [])]
    N = len(all_sids)
    k_sent = int(math.ceil(C_sent * N))

    # One fixed order per essay (independent of version/C_sent)
    order = deterministic_shuffle(all_sids, stable_seed(doc_id, "sent_order"))

    # Preserve previously touched, projected onto the fixed order
    prev_in_order = [sid for sid in order if sid in touched_prev]

    if len(prev_in_order) >= k_sent:
        chosen = prev_in_order[:k_sent]
        return set(chosen), _paras_for_sents(chosen, para_to_sents)

    # Add new sentences from the same fixed order
    remaining_in_order = [sid for sid in order if sid not in touched_prev]
    need = k_sent - len(prev_in_order)
    chosen = prev_in_order + remaining_in_order[:need]

    return set(chosen), _paras_for_sents(chosen, para_to_sents)
# ============================================================
# Prompt builder
# ============================================================
def _wc(x: str) -> int:
    return len(x.strip().split())

def _len_bounds(base_words: int, lo_mult: float, hi_mult: float):
    lo = max(1, int(round(base_words * lo_mult)))
    hi = max(lo, int(round(base_words * hi_mult)))
    return lo, hi

def op_to_guidance(op, style_target=None):
    tgt = style_target or "formal, natural, human-written"

    if op == "polish":
        return (
            "Make light edits to improve grammar, punctuation, fluency, and clarity while preserving meaning. "
            "Keep the sentence structure mostly unchanged. "
            "The output must not be identical to the input. "
            "Keep it as exactly one sentence. Do not add new facts."
        )

    if op == "paraphrase":
        return (
            "Rewrite the sentence with clearly different wording while preserving meaning. "
            "You may restructure the phrasing, but keep the same core content. "
            "The output must not be identical to the input. "
            "Keep it as exactly one sentence. Do not add new facts."
        )

    if op == "style":
        return (
            f"Rewrite the sentence in a {tgt} style by changing tone, register, and phrasing while preserving meaning. "
            "Do not change the underlying facts or add new content. "
            "The output must not be identical to the input. "
            "Keep it as exactly one sentence."
        )

    if op == "compress":
        return (
            "Rewrite the sentence to be shorter and more concise while preserving all essential meaning. "
            "Remove redundancy and non-essential phrasing, but do not omit important information. "
            "The output must not be identical to the input. "
            "Keep it as exactly one sentence. Do not add new facts."
        )

    if op == "expand":
        return (
            "Rewrite the sentence to be slightly more detailed using only information already stated or directly implied. "
            "You may add brief clarifying or descriptive phrasing, but do not introduce new facts, examples, names, dates, numbers, or claims. "
            "The output must not be identical to the input. "
            "Keep it as exactly one sentence."
        )

    return (
        "Rewrite the sentence while preserving meaning. "
        "The output must not be identical to the input. "
        "Keep it as exactly one sentence. Do not add new facts."
    )

def build_prompt(text, op,paragraph_context,
                 style_target="formal, natural, human-written"):
    base_words = _wc(text)

    if op in ["paraphrase", "style", "polish"]:
        lo, hi = _len_bounds(base_words, 0.85, 1.15)
    elif op == "compress":
        lo, hi = _len_bounds(base_words, 0.60, 0.80)
    elif op == "expand":
        lo, hi = _len_bounds(base_words, 1.20, 1.50)
    else:
        lo, hi = max(1, base_words), max(1, base_words)

    # guidance = intensity_to_guidance(op, intensity, style_target=style_target)
    guidance = op_to_guidance(op, style_target=style_target)

    return (
            f"Operation: {op}\n"
            f"Style target (if applicable): {style_target}\n"
            "Constraints:\n"
            "- Rewrite ONLY the target sentence.\n"
            "- Do NOT add new facts.\n"
            "- Preserve names, numbers, and entities.\n"
            "- Keep the same language as the target.\n"
            "- Return ONLY the rewritten text. No explanations, no quotes, no prefixes.\n"
            f"- Length constraint: {lo} to {hi} words.\n"
            "- No line breaks.\n"
            "- Output EXACTLY ONE sentence.\n\n"
            f"Guidance: {guidance}\n\n"
            "Paragraph context (for coherence only; do not rewrite it):\n"
            f"{paragraph_context}\n\n"
            "Target sentence:\n"
            f"{text}\n"
        )


def build_paragraph_prompt(paragraph_tagged, op,
                            style_target="formal, natural, human-written"):
    # guidance = intensity_to_guidance(op, intensity, style_target=style_target)
    guidance = op_to_guidance(op, style_target=style_target)

    return (
        f"Operation: {op}\n\n"
        "You are given a paragraph where some sentences are marked with "
        "<AI_Start>...</AI_End> tags.\n\n"
        "YOUR TASK:\n"
        "- Rewrite ONLY the sentences inside <AI_Start>...</AI_End> tags.\n"
        "- Leave ALL other sentences completely unchanged.\n"
        "- Return the full paragraph with the same tag structure.\n"
        "- Keep tags in the output around your rewritten sentences.\n\n"
        "CONSTRAINTS:\n"
        "- Do NOT rewrite untagged sentences.\n"
        "- Do NOT add new facts.\n"
        "- Preserve names, numbers, entities.\n"
        "- Each tagged sentence must remain exactly ONE sentence.\n"
        "- Keep the same language.\n\n"
        f"Guidance for tagged sentences: {guidance}\n\n"
        "Paragraph:\n"
        f"{paragraph_tagged}\n"
    )

def validate_paragraph_output(original_tagged, llm_output, sids_in_para, update_set):
    """
    Validate LLM paragraph output:
    1. Output contains <AI_Start>...</AI_End> tags
    2. Number of tagged spans matches number of sentences in update_set
    3. Output is not empty
    Returns llm_output if valid, None if fallback needed.
    """
    if not llm_output or not llm_output.strip():
        return None

    n_open  = llm_output.count("<AI_Start>")
    n_close = llm_output.count("</AI_End>")
    n_expected = sum(1 for sid in sids_in_para if sid in update_set)

    # must have matching open and close tags
    if n_open != n_close:
        return None

    # must have at least the expected number of tags
    # (allow extra tags — parse_paragraph_back handles them gracefully)
    if n_open < n_expected:
        return None

    return llm_output



def parse_paragraph_back(paragraph_output, sids_in_para, para_to_sents,
                          current_tagged, update_set, version):
    """
    Parse LLM paragraph output back into per-sentence current_tagged entries.
    
    Tagged sentences → extract rewritten content → wrap with tags → update current_tagged
    Untagged sentences → leave current_tagged unchanged
    """
    # Extract all tagged spans from output
    tagged_pattern = re.compile(r"<AI_Start>(.*?)</AI_End>", re.DOTALL)
    tagged_contents = [m.group(1).strip() for m in tagged_pattern.finditer(paragraph_output)]

    # Map back to sentence ids in order
    tag_idx = 0
    for sid in sids_in_para:
        if sid not in update_set:
            continue  # untagged — leave current_tagged[sid] unchanged

        if tag_idx < len(tagged_contents):
            rewritten = force_one_sentence(tagged_contents[tag_idx])
            if rewritten:
                current_tagged[sid] = f"<AI_Start>{rewritten}</AI_End>"
            else:
                # empty output for this sentence — keep old tagged if was AI
                pass
            tag_idx += 1

    return current_tagged


# ============================================================
# Provider clients (lazy init)
# ============================================================
_openai_client = None
_deepseek_client = None
_hf_router_client = None
_gemini_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("Missing dependency: pip install openai") from e
        _openai_client = OpenAI()
    return _openai_client

def get_deepseek_client():
    global _deepseek_client
    if _deepseek_client is None:
        from openai import OpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required.")
        _deepseek_client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    return _deepseek_client

def get_hf_router_client():
    global _hf_router_client
    if _hf_router_client is None:
        from openai import OpenAI
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required.")
        _hf_router_client = OpenAI(api_key=token, base_url=HF_ROUTER_BASE_URL)
    return _hf_router_client

def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        try:
            from google import genai
        except Exception as e:
            raise RuntimeError("Missing dependency: pip install google-genai") from e
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        _gemini_client = genai.Client(api_key=key) if key else genai.Client()
    return _gemini_client

# ============================================================
# Output cleanup
# ============================================================
def _clean_model_output(out: str) -> str:
    out = (out or "").strip()
    if not out:
        return out
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return ""
    first = lines[0].lower()
    bad_starts = (
        "sure", "here's", "here is", "of course", "certainly",
        "rewritten text", "revised version", "rewrite:", "revised:",
        "result:", "output:", "response:", "answer:",
        "here's the rewritten", "here is the rewritten",
        "i have rewritten", "i've rewritten",
        "below is", "the following",
    )
    if any(first.startswith(x) for x in bad_starts) and len(lines) >= 2:
        lines = lines[1:]
    elif any(first.startswith(x) for x in bad_starts) and len(lines) == 1:
        if ":" in lines[0]:
            lines[0] = lines[0].split(":", 1)[1].strip()
    out = " ".join(lines).strip()
    if (out.startswith('"') and out.endswith('"')) or \
       (out.startswith("'") and out.endswith("'")):
        out = out[1:-1].strip()
    out = out.replace("<AI_Start>", "").replace("</AI_End>", "").strip()
    return out

# ============================================================
# System message
# ============================================================
def get_system_msg(op: str) -> dict:
    base = (
         "You are a precise text rewriting assistant. You must output ONLY the rewritten target sentence.\n"
        "Return ONLY the rewritten text — nothing else.\n"
        "STRICT OUTPUT RULES:\n"
        "- Do NOT start with phrases like 'Here is', 'Here's', 'Sure', "
        "'Certainly', 'Of course', 'Rewritten:', 'Revised:', 'Result:'.\n"
        "- Do NOT explain what you did.\n"
        "- Do NOT add quotes around your output.\n"
        "- Do NOT add any prefix or suffix.\n"
        "- Your entire response = the rewritten text only.\n\n"
        "- Preserve meaning. Do NOT add new facts.\n"
        "- Preserve all names, numbers, dates, locations, and other entities exactly.\n"
        "- Do NOT introduce any new named entities, numbers, or specific claims.\n"
    )

    op_specific = {
        "paraphrase": "Your task is paraphrasing — same meaning, different wording.",
        "style":      "Your task is style transfer — same meaning, different tone/voice.",
        "compress":   "Your task is compression — shorter text, same meaning, no facts lost.",
        "expand":     "Your task is expansion — extend current text.",
        "polish":     "Your task is proofreading — fix errors and improve fluency only.",
    }
    specific = op_specific.get(op, "Follow the instruction precisely.")
    return {"role": "system", "content": f"{base}\n{specific}"}

@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))
def rewrite_paragraph(paragraph_tagged, op,
                      doc_id="?", version="?", pid="?",
                      style_target="formal, natural, human-written"):
    """
    Send whole paragraph to LLM. Tagged sentences get rewritten.
    Untagged sentences must stay unchanged.
    """
    prompt = build_paragraph_prompt(paragraph_tagged, op,
                                    style_target=style_target)
    base = {
        "role": "system",
        "content": (
            "You are a precise text rewriting assistant.\n"
            "You will receive a paragraph where some sentences are wrapped in "
            "<AI_Start>...</AI_End> tags.\n"
            "Rewrite ONLY the tagged sentences according to the operation.\n"
            "Leave all untagged sentences EXACTLY as they are — character for character.\n"
            "Keep the <AI_Start> and </AI_End> tags in your output around each rewritten sentence.\n"
            "Do NOT add any explanation, prefix, or suffix.\n"
            "Output ONLY the paragraph with rewritten tagged sentences."
        )
    }
    op_specific = {
        "paraphrase": "Your task is paraphrasing ",
        "style":      "Your task is style transfer — same meaning, different tone/voice.",
        "compress":   "Your task is compression — shorter text, same meaning, no facts lost.",
        "expand":     "Your task is expansion — extend current text.",
        "polish":     "Your task is proofreading — fix errors and improve fluency only.",
    }
    specific = op_specific.get(op, "Follow the instruction precisely.")
    user_msg = {"role": "user", "content": prompt}
    system_msg = {
    "role": "system",
    "content": base["content"] + f"\n{specific}"
}
    out = ""

    if LLM_PROVIDER == "openai":
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            # messages=[f"{base}\n{specific}", user_msg],
            messages=[system_msg, user_msg],
            # temperature=0.2,
        )
        out = resp.choices[0].message.content or ""

    elif LLM_PROVIDER == "deepseek":
        client = get_deepseek_client()
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[system_msg, user_msg],
            temperature=0.2,
        )
        out = resp.choices[0].message.content or ""


    elif LLM_PROVIDER == "gemini":
        client = get_gemini_client()
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_msg["content"],
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                # temperature=0.2,
            ),
        )
        out = resp.text or ""    

    else:
        raise ValueError(f"Unsupported provider for paragraph rewrite: {LLM_PROVIDER}")

    return out.strip()
# ============================================================
# LLM rewrite
# ============================================================
@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))
def rewrite_one_sentence(text, op, paragraph_context,
                         style_target="formal, natural, human-written"):
    if op == "none" or text.strip() == "":
        return force_one_sentence(text)

    prompt = build_prompt(text, op, paragraph_context,
                          style_target=style_target)
    system_msg = get_system_msg(op)
    user_msg = {"role": "user", "content": prompt}
    out = ""

    if LLM_PROVIDER == "openai":
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[system_msg, user_msg],
            # temperature=0.2,
        )
        out = resp.choices[0].message.content or ""

    elif LLM_PROVIDER == "deepseek":
        client = get_deepseek_client()
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[system_msg, user_msg],
            temperature=0.2,
        )
        out = resp.choices[0].message.content or ""

    elif LLM_PROVIDER == "hf_router":
        client = get_hf_router_client()
        resp = client.chat.completions.create(
            model=HF_MODEL,
            messages=[system_msg, user_msg],
            temperature=0.2,
        )
        out = resp.choices[0].message.content or ""

    elif LLM_PROVIDER == "hf_inference":
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required.")
        api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": HF_MAX_NEW_TOKENS,
                           "temperature": HF_TEMPERATURE},
            "options": {"wait_for_model": True},
        }
        r = requests.post(api_url, headers=headers, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HF error: {data.get('error')}")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            out = data[0].get("generated_text", "") or data[0].get("text", "") or ""
        elif isinstance(data, dict):
            out = data.get("generated_text", "") or data.get("text", "") or ""
        else:
            out = str(data)

    elif LLM_PROVIDER == "gemini":
        client = get_gemini_client()
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_msg["content"],
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                # temperature=0.2,
            ),
        )
        out = resp.text or ""
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

    out = _clean_model_output(out)
    out = force_one_sentence(out)
    return out

# ============================================================
# Apply op to one sentence
# ============================================================


def apply_op(
    sentence_clean, op, paragraph_context,
    doc_id="?", version="?", sid="?",
    force_tag=False,
    was_ai=False,
    old_tagged=None,
):
    sentence_clean = sentence_clean.strip()
    if not sentence_clean:
        return sentence_clean

    out = rewrite_one_sentence(
        sentence_clean, op,
        paragraph_context=paragraph_context,
        style_target="formal, natural, human-written",
    ).strip()
    out = force_one_sentence(out)
    return f"<AI_Start>{out}</AI_End>"

# ============================================================
# Reconstruction helpers
# ============================================================
def reconstruct_doc(current_tagged, para_ids, para_to_sents):
    paras_tagged = []
    for pid in para_ids:
        sids = para_to_sents.get(pid, [])
        sent_txt = [current_tagged[sid].strip() for sid in sids
                    if current_tagged[sid].strip()]
        paras_tagged.append(" ".join(sent_txt).strip())
    text_tagged = "\n\n".join([p for p in paras_tagged if p.strip()])
    text_clean, ai_spans = tags_to_clean_and_spans(text_tagged)
    return text_tagged, text_clean, ai_spans

def _doc_clean_structure(current_tagged, para_ids, para_to_sents):
    out = []
    for pid in para_ids:
        items = []
        for sid in para_to_sents.get(pid, []):
            s = strip_tags(current_tagged[sid]).strip()
            if s:
                items.append((sid, s))
        out.append((pid, items))
    return out

def compute_sentence_spans_in_clean(current_tagged, para_ids, para_to_sents):
    spans, pos = {}, 0
    struct = _doc_clean_structure(current_tagged, para_ids, para_to_sents)
    for pi, (pid, items) in enumerate(struct):
        for si, (sid, s) in enumerate(items):
            spans[sid] = (pos, pos + len(s))
            pos += len(s)
            if si != len(items) - 1:
                pos += 1
        if pi != len(struct) - 1:
            pos += 2
    return spans

def compute_paragraph_spans_in_clean(current_tagged, para_ids, para_to_sents):
    spans, pos = {}, 0
    struct = _doc_clean_structure(current_tagged, para_ids, para_to_sents)
    for pi, (pid, items) in enumerate(struct):
        para_clean = " ".join(s for _, s in items).strip()
        spans[pid] = (pos, pos + len(para_clean))
        pos += len(para_clean)
        if pi != len(struct) - 1:
            pos += 2
    return spans

# ============================================================
# Main builder for one essay
# ============================================================

def build_versions_for_essay(doc_id, full_text):
   
    full_text = normalize_text(str(full_text))
    paras = split_paragraphs(full_text)

    para_ids = []
    para_to_sents = {}
    sent_text_v0 = {}

    sid_counter = 0
    for p_idx, p in enumerate(paras, start=1):
        pid = f"p{p_idx:03d}"
        para_ids.append(pid)
        sents = split_sentences(p)
        para_to_sents[pid] = []
        for s in sents:
            sid_counter += 1
            sid = f"s{sid_counter:04d}"
            para_to_sents[pid].append(sid)
            sent_text_v0[sid] = s.strip()

    all_sids = [sid for pid in para_ids for sid in para_to_sents.get(pid, [])]
    if not all_sids:
        return []

    current_tagged = {sid: sent_text_v0[sid] for sid in all_sids}
    touched_prev = set()
    rows = []

    # for (version, op, intensity, C_sent) in SCHEDULE:
    for (version, op, C_sent) in SCHEDULE:

        op_str  = op        if op        else "none"
        # int_str = intensity if intensity else "-"



        if version == "v0" or op == "none":
            current_tagged = {sid: sent_text_v0[sid] for sid in all_sids}
            touched_prev   = set()
            update_set = set()    

        # ── v1–v8: rewrite touched sentences ──
        else:
            touched_sents, touched_paras = select_touched_blocks(
                doc_id, para_ids, para_to_sents,
                C_sent,
                touched_prev=touched_prev,
            )
            update_set = touched_sents

            # ── ONE loop over touched paragraphs ──
            for pid in touched_paras:
                sids_in_para = para_to_sents.get(pid, [])

                # build tagged paragraph: mark sentences to rewrite with tags
                para_parts = []
                for sid in sids_in_para:
                    clean = strip_tags(current_tagged[sid]).strip()
                    if not clean:
                        continue
                    if sid in update_set:
                        para_parts.append(f"<AI_Start>{clean}</AI_End>")
                    else:
                        para_parts.append(clean)

                paragraph_tagged_input = " ".join(para_parts)

                if os.getenv("VERBOSE", "0") == "1":
                    print(f"    [para {pid}] → {paragraph_tagged_input[:120]}...")

                # ── primary path: one LLM call for whole paragraph ──
                llm_output = rewrite_paragraph(
                    paragraph_tagged_input, op,
                    doc_id=doc_id, version=version, pid=pid
                )

                validated = validate_paragraph_output(
                    paragraph_tagged_input, llm_output, sids_in_para, update_set
                )

                if validated is not None:
                    # parse rewritten sentences back into current_tagged
                    current_tagged = parse_paragraph_back(
                        validated, sids_in_para, para_to_sents,
                        current_tagged, update_set, version
                    )

                else:
                    
                    with open("fallback.log", "a", encoding="utf-8") as f:
                        f.write(f"[FALLBACK] doc={doc_id} version={version} pid={pid}\n")

                    paragraph_context = " ".join(
                        strip_tags(current_tagged[x]).strip()
                        for x in sids_in_para
                        if strip_tags(current_tagged[x]).strip()
                    ).strip()

                    for sid in sids_in_para:
                        if sid not in update_set:
                            continue
                        old      = current_tagged[sid]
                        was_ai   = "<AI_Start>" in old
                        base_clean = strip_tags(old)
                        current_tagged[sid] = apply_op(
                            base_clean, op, paragraph_context,
                            doc_id=doc_id, version=version, sid=sid,
                            force_tag=(version == "v8"),
                            was_ai=was_ai,
                            old_tagged=old,
                        )

            touched_prev = touched_sents


        # ══════════════════════════════════════════════
        # Reconstruct + compute metrics (runs for ALL versions including v0)
        # ══════════════════════════════════════════════

        # ── reconstruct full document ──
        text_tagged, text_clean, ai_spans = reconstruct_doc(
            current_tagged, para_ids, para_to_sents)

        # ── token labels ──
        tokens, tok_offsets = whitespace_tokens_with_offsets(text_clean)
        tok_labels      = labels_from_char_spans(tok_offsets, ai_spans)
        AI_token_ratio  = sum(tok_labels) / max(1, len(tok_labels))
        ai_spans_tok    = spans_from_labels(tok_labels)
        num_ai_spans_tok    = len(ai_spans_tok)
        avg_ai_span_len_tok = (
            sum(e - s for s, e in ai_spans_tok) / num_ai_spans_tok
            if num_ai_spans_tok else 0.0
        )
        boundary_pattern = get_boundary_pattern(tok_labels)

        # ── char ratio ──
        ai_chars       = sum(e - s for s, e in ai_spans)
        AI_char_ratio  = ai_chars / max(1, len(text_clean))

        # ── sentence-level ratios ──
        sent_spans = compute_sentence_spans_in_clean(current_tagged, para_ids, para_to_sents)

        sentence_ids = list(sent_spans.keys())
        sentences = []
        sent_ai_fracs = []
        sent_labels = []
        sent_touched = 0

        for sid in sentence_ids:
            ss = sent_spans[sid]

            sent_text = strip_tags(current_tagged[sid]).strip()
            sentences.append(sent_text)

            s_ai = sum(overlap_len(ss, sp) for sp in ai_spans)
            frac = s_ai / max(1, ss[1] - ss[0])
            sent_ai_fracs.append(frac)

            label = 1 if frac > 0 else 0
            sent_labels.append(label)

            if label == 1:
                sent_touched += 1

        AI_sent_ratio = sent_touched / max(1, len(sent_spans))
        Avg_sent_ai_frac = sum(sent_ai_fracs) / max(1, len(sent_ai_fracs))
        touched_fracs = [f for f in sent_ai_fracs if f > 0]
        Avg_sent_ai_frac_touched = (
            sum(touched_fracs) / len(touched_fracs) if touched_fracs else 0.0
        )

        # ── paragraph-level ratios ──
        para_spans = compute_paragraph_spans_in_clean(current_tagged, para_ids, para_to_sents)
        para_ai_fracs, para_touched = [], 0
        for pid, ps in para_spans.items():
            p_ai = sum(overlap_len(ps, sp) for sp in ai_spans)
            frac = p_ai / max(1, ps[1] - ps[0])
            para_ai_fracs.append(frac)
            if frac > 0:
                para_touched += 1
        C_para_measured  = para_touched / max(1, len(para_spans))
        Avg_para_ai_frac = sum(para_ai_fracs) / max(1, len(para_ai_fracs))

        rows.append({
            # identity
            "doc_id":   str(doc_id),
            "version":    version,
            "split":      assign_split(doc_id),
            "model_used": get_model_label(),

            # essay metadata
            "num_paragraphs":   len(para_ids),
            "num_sentences":    len(all_sids),
            "essay_length":     len(full_text.split()),
            "C_para_meaningful": len(para_ids) > 1,

            # edit controls
            "operation":    op,
            "C_sent_target": C_sent,

            # AI ratio metrics — sentence primary, token/char secondary
            "AI_sent_ratio":             AI_sent_ratio,
            "Avg_sent_ai_frac":          Avg_sent_ai_frac,
            "Avg_sent_ai_frac_touched":  Avg_sent_ai_frac_touched,
            "AI_token_ratio":            AI_token_ratio,
            "AI_char_ratio":             AI_char_ratio,
            "C_para_measured":           C_para_measured,
            "Avg_para_ai_frac":          Avg_para_ai_frac,

            # text
            "text_clean":  text_clean,
            "text_tagged": text_tagged,

            # span annotations
            "ai_spans_char":        json.dumps(ai_spans, ensure_ascii=False),
            "ai_spans_tok":         json.dumps(ai_spans_tok),
            "num_ai_spans_tok":     num_ai_spans_tok,
            "avg_ai_span_len_tok":  avg_ai_span_len_tok,

            # token labels
            "tokens":          json.dumps(tokens, ensure_ascii=False),
            "tok_labels":      json.dumps(tok_labels),
            "boundary_pattern": boundary_pattern,

            "num_sentences_total": len(sentence_ids),
            "num_sentences_edited": len(update_set),
            "num_sentences_ai_total": sent_touched,

            "sentence_ids": json.dumps(sentence_ids, ensure_ascii=False),
            "sentences": json.dumps(sentences, ensure_ascii=False),
            "sent_labels": json.dumps(sent_labels),
            "sent_ai_fracs": json.dumps(sent_ai_fracs),
        })


    return rows

# ============================================================
# Build all essays with checkpointing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--id_column", default="id")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--start_idx", type=int, default=0)
    return parser.parse_args()


def process_essay(row, id_column="id", text_column="document_clean"):
    idx, record = row
    doc_id = record[id_column]
    full_text = record[text_column]
    thread_name = f"Thread-{idx % 8 + 1}"

    try:
        return build_versions_for_essay(doc_id, full_text)
    except RetryError as e:
        root = e.last_attempt.exception()
        msg = f"[{thread_name}] FAILED essay {doc_id}: {type(root).__name__}: {root}"
        print(msg, flush=True)
        with open("failed_essays.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        return []
    except Exception as e:
        msg = f"[{thread_name}] FAILED essay {doc_id}: {type(e).__name__}: {e}"
        print(msg, flush=True)
        with open("failed_essays.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        return []


def build_all(
    input_csv,
    output_csv,
    id_column="id",
    text_column="text",
    max_docs=None,
    num_threads=8,
    start_idx=0,
):
    df = pd.read_csv(input_csv)

    if id_column not in df.columns or text_column not in df.columns:
        raise ValueError(
            f"Input CSV must contain columns: {id_column!r} and {text_column!r}. "
            f"Found columns: {list(df.columns)}"
        )

    if max_docs is not None:
        df = df.iloc[start_idx:start_idx + max_docs]
    else:
        df = df.iloc[start_idx:]

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = output_csv.replace(".csv", "_checkpoint.csv")

    already_done = set()
    if os.path.exists(checkpoint_path):
        done_df = pd.read_csv(checkpoint_path)
        already_done = set(done_df["doc_id"].astype(str).unique())
        print(f"Resuming: {len(already_done)} documents already completed.")

    remaining = df[~df[id_column].astype(str).isin(already_done)].copy()

    print(
        f"Processing {len(remaining)} documents x {len(SCHEDULE)} versions = "
        f"{len(remaining) * len(SCHEDULE)} rows using {num_threads} threads."
    )

    if len(remaining) == 0:
        final_df = pd.read_csv(checkpoint_path) if os.path.exists(checkpoint_path) else pd.DataFrame()
        final_df.to_csv(output_csv, index=False)
        print(f"Saved: {output_csv} rows={len(final_df)}")
        return

    records_iter = list(remaining.iterrows())

    def _process(row):
        return process_essay(row, id_column=id_column, text_column=text_column)

    with ThreadPool(processes=num_threads) as pool:
        for essay_rows in tqdm(pool.imap(_process, records_iter), total=len(records_iter), desc="doc"):
            if not essay_rows:
                continue
            essay_df = pd.DataFrame(essay_rows)
            write_header = not os.path.exists(checkpoint_path)
            essay_df.to_csv(checkpoint_path, mode="a", header=write_header, index=False)

    final_df = pd.read_csv(checkpoint_path)
    final_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv} rows={len(final_df)}")


if __name__ == "__main__":
    args = parse_args()
    build_all(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        id_column=args.id_column,
        text_column=args.text_column,
        max_docs=args.max_docs,
        num_threads=args.num_threads,
        start_idx=args.start_idx,
    )
