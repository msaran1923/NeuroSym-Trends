#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
thesis_topic_trends_pipeline.py

End-to-end neuro-symbolic topic modeling pipeline for trend analysis on a thesis corpus.

Features:
  - Robust preprocessing & hash-based deduplication.
  - CSO (Computer Science Ontology) concept extraction.
  - BERTopic modeling with advanced outlier management.
  - LLM-based gating/refinement for difficult outliers.
  - Neuro-symbolic augmentation (merging symbolic CSO concepts with dense clusters).
  - Trend analysis (Mann-Kendall, Kleinberg Bursts) & Semantic Shift quantification.
"""

from __future__ import annotations

import datetime
import argparse
import ast
import gzip
import hashlib
import json
import math
import re
import sys
import unicodedata
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
try:
    import torch
except Exception:
    torch = None

# -----------------------------
# Optional Dependencies
# -----------------------------
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

# -----------------------------
# Utilities
# -----------------------------
_TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def robust_read_csv(path: Path) -> pd.DataFrame:
    """Reads CSV with fallback encodings."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            if "doc_id" in df.columns:
                df["doc_id"] = df["doc_id"].astype(str)
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV {path}. Last error: {last_err}")


def looks_non_english(text: str, max_tr_chars: int = 0) -> bool:
    if not text:
        return False
    c = sum(1 for ch in str(text) if ch in _TURKISH_CHARS)
    return c > int(max_tr_chars)

def fix_hyphenated_linebreaks(text: str) -> str:
    """Fixes words broken by hyphens at line endings (e.g. 'algo-\nrithm')."""
    if not text:
        return ""
    t = str(text).replace("\r", "\n")
    # Join hyphenated words across newlines
    t = re.sub(r"(?P<a>\w)-\s*\n\s*(?P<b>\w)", r"\g<a>\g<b>", t)
    # Replace newlines with spaces
    t = re.sub(r"\s*\n\s*", " ", t)
    # Collapse multiple spaces
    t = re.sub(r"[ \t]+", " ", t).strip()
    return t

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = fix_hyphenated_linebreaks(text)
    try:
        import ftfy  # type: ignore
        t = ftfy.fix_text(str(t))
    except ImportError:
        pass
    except Exception:
        pass

    t = unicodedata.normalize("NFKC", t)
    t = (t
         .replace("\u2010", "-")
         .replace("\u2011", "-")
         .replace("\u2012", "-")
         .replace("\u2013", "-")
         .replace("\u2014", "-")
         .replace("\u2212", "-")
         .replace("\t", " ")
         )
    t = re.sub(r"[ \u00A0]+", " ", t).strip()
    return t

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalizes matrix rows."""
    X = np.asarray(X, dtype=np.float32)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norm, eps)

def save_embeddings_with_doc_ids(embeddings_path: Path, emb: np.ndarray, doc_ids: List[str]) -> Path:
    ensure_dir(embeddings_path.parent)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(embeddings_path, emb)
    sidecar = embeddings_path.with_name(embeddings_path.stem + "_doc_ids.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump([str(d) for d in doc_ids], f, ensure_ascii=False)
    return sidecar

def load_embeddings_with_doc_ids(embeddings_path: Path) -> Tuple[np.ndarray, List[str]]:
    sidecar = embeddings_path.with_name(embeddings_path.stem + "_doc_ids.json")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not sidecar.exists():
        # Fallback: if no sidecar, assume consistency is user's responsibility (legacy mode)
        # But generally we raise error to prevent misalignment.
        raise FileNotFoundError(f"Embeddings doc_id sidecar missing: {sidecar}")

    emb = np.load(embeddings_path)
    with open(sidecar, "r", encoding="utf-8") as f:
        doc_ids = json.load(f)
    doc_ids = [str(d) for d in doc_ids]

    if int(emb.shape[0]) != len(doc_ids):
        raise ValueError(f"Embedding rows ({emb.shape[0]}) != doc_ids ({len(doc_ids)}).")
    return np.asarray(emb, dtype=np.float32), doc_ids


def make_doc_id_to_emb_idx(doc_ids: List[str]) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for i, d in enumerate(doc_ids):
        sd = str(d)
        if sd not in m:
            m[sd] = int(i)
    return m

def make_doc_ids_unique(df: pd.DataFrame, id_col: str = "doc_id") -> pd.DataFrame:
    if id_col not in df.columns:
        return df
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    # Check for duplicates
    if df[id_col].duplicated().any():
        cc = df.groupby(id_col).cumcount()
        dup_mask = cc > 0
        df.loc[dup_mask, id_col] = df.loc[dup_mask, id_col] + "__dup" + cc.loc[dup_mask].astype(str)
    return df

def require_doc_ids_present(df: pd.DataFrame) -> None:
    if "doc_id" not in df.columns:
        raise ValueError("Expected column 'doc_id' not found.")
    if df["doc_id"].isna().any():
        raise ValueError("doc_id contains NaN values.")

# -----------------------------
# LLM utilities (Transformers / Ollama)
# -----------------------------

def load_llm_transformers(model_name: str):
    """Load a local Transformers chat/instruct model for Stage-5 gating."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    kwargs = dict(device_map=device_map, trust_remote_code=True)
    try:
        kwargs["dtype"] = dtype
        mdl = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        mdl = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if device_map is None:
        mdl = mdl.to("cpu")

    if getattr(mdl.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        mdl.config.pad_token_id = tok.pad_token_id

    try:
        mdl.config.use_cache = False
    except Exception:
        pass
    try:
        if hasattr(mdl, "generation_config") and mdl.generation_config is not None:
            mdl.generation_config.use_cache = False
    except Exception:
        pass

    mdl.eval()
    return tok, mdl


def llm_choose_topic_transformers_batch(tok, mdl, prompts: List[str], max_new_tokens: int = 256) -> List[str]:
    """Batch-generate short answers with a local Transformers model."""
    outs: List[str] = []
    for p in tqdm(prompts, desc="Local LLM", unit="doc"):
        inputs = tok([p], return_tensors="pt", padding=True, truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            mdl = mdl.to("cuda")
        with torch.no_grad():
            gen = mdl.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        txt = tok.decode(gen[0], skip_special_tokens=True)
        if txt.startswith(p):
            txt = txt[len(p):].strip()
        outs.append(txt.strip())
    return outs


def llm_choose_topic_ollama_batch(
    prompts: List[str],
    llm_model: str,
    host: str = "http://localhost:11434",
    system_prompt: Optional[str] = None,
    max_workers: int = 4,
    timeout: int = 600,
    num_predict: int = 256,
) -> List[str]:
    """Batch-call Ollama. Prefers /api/chat (supports system messages), falls back to /api/generate."""
    import urllib.request

    host = str(host or "http://localhost:11434").rstrip("/")
    url_chat = host + "/api/chat"
    url_gen = host + "/api/generate"

    def _post_json(url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except Exception:
            return {"_raw": raw}

    def _call_one(prompt: str) -> str:
        # 1) try chat endpoint (system role supported)
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": str(system_prompt)})
        msgs.append({"role": "user", "content": str(prompt)})

        payload_chat = {
            "model": llm_model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": int(num_predict)},
        }
        try:
            j = _post_json(url_chat, payload_chat)
            content = ""
            if isinstance(j, dict):
                if isinstance(j.get("message"), dict):
                    content = str(j["message"].get("content", "")).strip()
                if not content and "response" in j:
                    content = str(j.get("response", "")).strip()
            if content:
                return content
        except Exception:
            pass

        # 2) fallback generate endpoint (inject system prompt into prompt text)
        prompt2 = str(prompt)
        if system_prompt:
            prompt2 = f"{system_prompt}\n\n{prompt2}"
        payload_gen = {
            "model": llm_model,
            "prompt": prompt2,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": int(num_predict)},
        }
        try:
            j = _post_json(url_gen, payload_gen)
            if isinstance(j, dict):
                return str(j.get("response", "")).strip()
            return ""
        except Exception:
            return '{"topic_id": -1}'

    from concurrent.futures import ThreadPoolExecutor, as_completed

    outs = [""] * len(prompts)
    with ThreadPoolExecutor(max_workers=int(max_workers or 1)) as ex:
        futs = {ex.submit(_call_one, p): i for i, p in enumerate(prompts)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Ollama LLM", unit="doc"):
            i = futs[fut]
            try:
                outs[i] = str(fut.result() or "").strip()
            except Exception:
                outs[i] = '{"topic_id": -1}'
    return outs


def _parse_list_maybe(x: Any) -> List[str]:
    """Best-effort parser for columns that may contain a list serialized as a string.

    Accepts:
      - list/tuple/set objects
      - JSON list strings, e.g., '["a","b"]'
      - Python literal list strings, e.g., "['a', 'b']"
      - delimiter-joined strings, e.g., 'a; b; c'

    Returns an order-preserving, de-duplicated list of non-empty strings.
    """
    if x is None:
        return []
    try:
        # NaN check (float NaN only)
        if isinstance(x, float) and np.isnan(x):
            return []
    except Exception:
        pass

    if isinstance(x, list):
        out = [str(v).strip() for v in x if str(v).strip()]
        return list(dict.fromkeys(out))
    if isinstance(x, (tuple, set)):
        out = [str(v).strip() for v in list(x) if str(v).strip()]
        return list(dict.fromkeys(out))

    if isinstance(x, str):
        s = x.strip()
        if (not s) or (s.lower() in ("nan", "none", "null")):
            return []

        # Try JSON or Python literal forms
        if (s[0] in "[{(") and (s[-1] in "]})"):
            obj = None
            try:
                obj = json.loads(s)
            except Exception:
                try:
                    obj = ast.literal_eval(s)
                except Exception:
                    obj = None

            if isinstance(obj, list):
                out = [str(v).strip() for v in obj if str(v).strip()]
                return list(dict.fromkeys(out))
            if isinstance(obj, (tuple, set)):
                out = [str(v).strip() for v in list(obj) if str(v).strip()]
                return list(dict.fromkeys(out))
            if isinstance(obj, dict):
                out = [str(k).strip() for k in obj.keys() if str(k).strip()]
                return list(dict.fromkeys(out))

        # Fallback: split on common delimiters
        for sep in (";", "|", ","):
            if sep in s:
                parts = [p.strip() for p in s.split(sep) if p.strip()]
                return list(dict.fromkeys(parts))

        # Single token
        return [s]

    # Other scalar types
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass

    sx = str(x).strip()
    return [sx] if sx else []


# -----------------------------
# Statistics & Trends
# -----------------------------
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg False Discovery Rate correction."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    q = np.full(m, np.nan, dtype=float)
    if m == 0:
        return q

    order = np.argsort(p)
    ranked = p[order]

    valid = np.isfinite(ranked)
    if not valid.any():
        return q

    ranked_valid = ranked[valid]
    idx_valid = np.where(valid)[0]
    m_valid = len(ranked_valid)

    # q = p * m / rank
    q_valid = ranked_valid * m_valid / (np.arange(1, m_valid + 1))
    # enforce monotonicity
    q_valid = np.minimum.accumulate(q_valid[::-1])[::-1]
    q_valid = np.clip(q_valid, 0.0, 1.0)

    q_temp = np.full(m, np.nan, dtype=float)
    q_temp[idx_valid] = q_valid
    q[order] = q_temp
    return q


def mann_kendall_test(
    series: List[float],
    alpha: float = 0.05,
    method: str = "classic",
    lag_max: Optional[int] = None
) -> Tuple[float, float, str]:
    """Non-parametric Mann–Kendall trend test with optional autocorrelation handling.

    Parameters
    ----------
    series:
        Observations ordered in time.
    alpha:
        Significance level used only to label the returned `trend` string.
    method:
        - "classic": standard MK (assumes independent observations)
        - "tfpw": trend-free prewhitening (TFPW) using a Sen/Theil slope estimate + lag-1 AR correction
        - "hamed_rao": Hamed–Rao variance correction using autocorrelation of ranks
        (Hyphens/spaces are ignored; e.g., "hamed-rao" is accepted.)
    lag_max:
        Maximum lag used for the Hamed–Rao correction. If None, uses min(10, n-1).

    Returns
    -------
    (p_value, kendall_tau, trend_label)
    """
    x0 = np.asarray(series, dtype=float)
    # Remove NaNs/Infs conservatively
    x = x0[np.isfinite(x0)]
    n = int(len(x))
    if n < 3:
        return 1.0, 0.0, "no_trend"

    mth = (method or "classic").strip().lower().replace("-", "_").replace(" ", "_")
    if mth in ("hamedrao",):
        mth = "hamed_rao"

    def _mk_stat(xx: np.ndarray) -> Tuple[int, float, float]:
        """Compute S, var(S) (with tie correction), and Kendall tau."""
        nn = int(len(xx))
        s = 0
        for i in range(nn - 1):
            s += int(np.sum(np.sign(xx[i + 1:] - xx[i])))

        # Variance with tie correction
        xr = np.round(xx, 10)
        _, counts = np.unique(xr, return_counts=True)
        tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
        var_s = (nn * (nn - 1) * (2 * nn + 5) - tie_term) / 18.0

        denom = nn * (nn - 1) / 2.0
        tau = float(s / denom) if denom else 0.0
        return int(s), float(var_s), tau

    # Optional preprocessing for autocorrelation
    if mth == "tfpw":
        # Trend-free prewhitening:
        # 1) estimate slope, 2) detrend, 3) estimate lag-1 autocorr on detrended,
        # 4) prewhiten, 5) add trend back, 6) run classic MK.
        t = np.arange(n, dtype=float)
        slope = float(theil_sen_slope(x.tolist()))
        x_detr = x - slope * t
        if n >= 3:
            try:
                r1 = float(np.corrcoef(x_detr[1:], x_detr[:-1])[0, 1])
            except Exception:
                r1 = 0.0
            if not np.isfinite(r1):
                r1 = 0.0
            r1 = float(np.clip(r1, -0.99, 0.99))
        else:
            r1 = 0.0
        x_pw = x_detr.copy()
        x_pw[1:] = x_detr[1:] - r1 * x_detr[:-1]
        x = x_pw + slope * t
        # n unchanged

    # Compute MK S and var(S)
    s, var_s, tau = _mk_stat(x)
    if var_s <= 0:
        return 1.0, 0.0, "no_trend"

    # Hamed–Rao: variance inflation/deflation using autocorrelation of ranks
    if mth == "hamed_rao":
        # Use ranks to reduce sensitivity to non-normality and ties
        r = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        kmax = int(lag_max) if lag_max is not None else min(10, n - 1)
        kmax = max(1, min(kmax, n - 1))
        denom = 1.0
        acc = 0.0
        for k in range(1, kmax + 1):
            a = r[k:]
            b = r[:-k]
            if len(a) < 3:
                break
            try:
                rk = float(np.corrcoef(a, b)[0, 1])
            except Exception:
                rk = 0.0
            if not np.isfinite(rk):
                rk = 0.0
            acc += (1.0 - (k / n)) * rk
        denom = 1.0 + 2.0 * acc
        # guard against pathological negative denominators
        denom = float(max(0.1, denom))
        var_s = float(var_s * denom)

    # Z-score and p-value
    z = (s - 1.0) / math.sqrt(var_s) if s > 0 else ((s + 1.0) / math.sqrt(var_s) if s < 0 else 0.0)

    try:
        from math import erf
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / math.sqrt(2.0))))
    except Exception:
        p = 1.0

    trend = "increasing" if (p <= alpha and tau > 0) else ("decreasing" if (p <= alpha and tau < 0) else "no_trend")
    return float(p), float(tau), trend

def kendall_tau(series: List[float]) -> Tuple[float, float]:
    try:
        from scipy.stats import kendalltau  # type: ignore
        tau, p = kendalltau(np.arange(len(series)), series)
        return float(tau if np.isfinite(tau) else 0.0), float(p if np.isfinite(p) else 1.0)
    except ImportError:
        return 0.0, 1.0


def theil_sen_slope(y: List[float], x: Optional[List[float]] = None) -> float:
    yy = np.asarray(y, dtype=float)
    n = len(yy)
    xx = np.arange(n, dtype=float) if x is None else np.asarray(x, dtype=float)
    if n < 2:
        return 0.0

    slopes: List[float] = []
    for i in range(n - 1):
        dx = xx[i + 1:] - xx[i]
        dy = yy[i + 1:] - yy[i]
        # Avoid division by zero
        m = dy / np.where(dx == 0, np.nan, dx)
        slopes.extend(m[np.isfinite(m)].tolist())

    if not slopes:
        return 0.0
    return float(np.median(slopes))


def kleinberg_bursts(counts: List[int], s: float = 2.0, gamma: float = 1.0) -> Tuple[List[int], int, List[int]]:
    """Kleinberg's burst detection algorithm for discrete time series."""
    y = np.asarray(counts, dtype=int)
    n = len(y)
    if n == 0:
        return [], 0, []
    if y.max() == 0:
        return [], 0, [0] * n

    # Base rate estimate
    base = float(max(y.mean(), 1e-6))
    max_obs = float(y.max())
    # Calculate number of levels
    L = int(max(0, math.ceil(math.log(max_obs / base, s)))) + 1
    rates = np.array([base * (s ** i) for i in range(L + 1)], dtype=float)

    def nll(k: int, lam: float) -> float:
        # Poisson Negative Log Likelihood: -ln(e^-lam * lam^k / k!) ~ lam - k*ln(lam) (ignoring constants)
        return lam - k * math.log(max(lam, 1e-12))

    # Viterbi
    dp = np.full((n, L + 1), np.inf, dtype=float)
    back = np.zeros((n, L + 1), dtype=int)

    # Init
    for j in range(L + 1):
        dp[0, j] = nll(int(y[0]), float(rates[j])) + gamma * j

    # Forward
    for t in range(1, n):
        for j in range(L + 1):
            # Transition cost from previous state k to current state j
            # Cost = prev_cost + transition_cost (gamma * |k-j|)
            costs = dp[t - 1, :] + gamma * np.abs(np.arange(L + 1) - j)
            k_best = int(np.argmin(costs))
            dp[t, j] = costs[k_best] + nll(int(y[t]), float(rates[j]))
            back[t, j] = k_best

    # Backtrack
    last = int(np.argmin(dp[n - 1, :]))
    states = [last]
    for t in range(n - 1, 0, -1):
        last = int(back[t, last])
        states.append(last)
    states = states[::-1]

    # Identify burst starts (where state increases)
    starts = []
    for t in range(1, n):
        if states[t] > states[t - 1]:
            starts.append(t)

    return starts, int(max(states)), states


# -----------------------------
# Stage 1: Preprocess
# -----------------------------
def stage01_preprocess(df: pd.DataFrame, out_csv: Path, max_tr_chars: int = 0) -> pd.DataFrame:
    required = ["thesis_type", "english_title", "year", "abstract_en"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Create stable doc_id if missing
    if "doc_id" not in df.columns:
        key = (df["thesis_type"].astype(str).fillna("") + "|" +
               df["english_title"].astype(str).fillna("") + "|" +
               df["year"].astype(str).fillna("") + "|" +
               df["abstract_en"].astype(str).fillna(""))
        df.insert(0, "doc_id", key.map(lambda s: hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()))

    df["doc_id"] = df["doc_id"].astype(str)
    df = make_doc_ids_unique(df, id_col="doc_id")

    # Text cleaning
    df["english_title_clean"] = df["english_title"].map(normalize_text)
    df["abstract_en_clean"] = df["abstract_en"].map(normalize_text)

    # Language filtering heuristics
    df["flag_non_english_title"] = df["english_title_clean"].map(lambda x: looks_non_english(x, max_tr_chars=max_tr_chars))

    # Combined text for embedding
    df["text"] = (df["english_title_clean"].fillna("") + ". " + df["abstract_en_clean"].fillna("")).map(normalize_text)

    # Filter invalid years
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df = df[df["year"] > 1000].copy()  # Basic sanity check

    require_doc_ids_present(df)
    df.to_csv(out_csv, index=False)
    return df


# -----------------------------
# Stage 2: CSO Extraction
# -----------------------------
def _require_deps_cso():
    try:
        import rdflib
        import faiss
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Missing deps for CSO (rdflib, faiss, sentence-transformers). Install them via pip."
        ) from e


def _cso_unify_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _cso_norm_for_collision(s: str) -> str:
    # Collision key for label texts (aggressive normalize)
    t = _cso_unify_spaces(s).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = _cso_unify_spaces(t)
    return t


def _cso_slug(uri: str) -> str:
    return str(uri).split("/")[-1].replace("_", " ")


def _cso_is_english_literal(o) -> bool:
    # rdflib Literal may have language tags; keep English or unlabeled literals.
    try:
        lang = getattr(o, "language", None)
        if not lang:
            return True
        lang = str(lang).lower()
        return lang == "en" or lang.startswith("en-") or lang == "eng"
    except Exception:
        return True


def prepare_cso_index(
    ttl_path: Path,
    model,
    cache_dir: Path,
    model_name: str,
    label_batch_size: int = 128,
    drop_ambiguous_short: bool = True,
):
    """
    Builds (or loads) a FAISS index over CSO label texts.
    Updated to handle CSO-specific equivalence properties which point to URIs.
    """
    _require_deps_cso()
    import pickle
    import rdflib
    import faiss
    ensure_dir(cache_dir)
    ttl_path = Path(ttl_path)
    ttl_stat = ttl_path.stat()

    # Bump version to force cache rebuild since logic changed
    schema_version = 5
    cache_index = cache_dir / "cso_faiss_v5.index"
    cache_meta = cache_dir / "cso_meta_v5.pkl"

    # --- Cache Loading Block (Same as before) ---
    if cache_index.exists() and cache_meta.exists():
        try:
            with open(cache_meta, "rb") as f:
                meta = pickle.load(f)
            ok = True
            ok = ok and meta.get("schema_version") == schema_version
            ok = ok and meta.get("ttl_size") == int(ttl_stat.st_size)
            ok = ok and meta.get("ttl_mtime_ns") == int(getattr(ttl_stat, "st_mtime_ns", int(ttl_stat.st_mtime * 1e9)))
            ok = ok and str(meta.get("model_name", "")) == str(model_name)
            ok = ok and meta.get("uris") and meta.get("texts") and meta.get("ambiguous_mask") is not None
            if ok:
                uris = list(meta["uris"])
                texts = list(meta["texts"])
                ambiguous_mask = list(meta["ambiguous_mask"])
                if len(uris) == len(texts) == len(ambiguous_mask) and len(texts) > 0:
                    print(f"   (CSO) Loaded {len(texts)} label entries from cache.")
                    return faiss.read_index(str(cache_index)), uris, texts, ambiguous_mask
        except Exception:
            pass

    print(f"   (CSO) Parsing ontology: {ttl_path} ...")
    g = rdflib.Graph()
    try:
        g.parse(str(ttl_path), format="turtle")
    except Exception as e:
        raise RuntimeError(f"Could not parse Turtle file {ttl_path}: {e}")

    # --- UPDATED PREDICATE LOGIC ---
    # Define standard label and CSO-specific equivalence properties
    pred_label = rdflib.URIRef("http://www.w3.org/2000/01/rdf-schema#label")
    pred_pref_equiv = rdflib.URIRef("http://cso.kmi.open.ac.uk/schema/cso#preferentialEquivalent")
    pred_rel_equiv = rdflib.URIRef("http://cso.kmi.open.ac.uk/schema/cso#relatedEquivalent")

    preds = [pred_label, pred_pref_equiv, pred_rel_equiv]

    # Set of predicates where the object is a URI (not a literal string)
    uri_target_preds = {str(pred_pref_equiv), str(pred_rel_equiv)}

    uri_to_texts: Dict[str, Set[str]] = {}
    norm_to_uris: Dict[str, Set[str]] = {}

    for p in preds:
        p_str = str(p)
        is_uri_target = p_str in uri_target_preds

        # Friendly name for progress bar
        p_name = p_str.split('#')[-1]

        for s, o in tqdm(g.subject_objects(p), desc=f"(CSO) read {p_name}", unit="pair"):
            s_str = str(s)
            if "/topics/" not in s_str:
                continue

            # --- Logic Split: Literal vs URI ---
            if is_uri_target:
                # Object is a URI (synonym topic). Extract text from slug.
                # e.g. <.../topics/non-rigid_registration> -> "non rigid registration"
                txt = _cso_slug(str(o))
            else:
                # Object is a Literal (rdfs:label). Check lang and unify spaces.
                if not _cso_is_english_literal(o):
                    continue
                txt = _cso_unify_spaces(str(o))

            if not txt or len(txt) < 3:
                continue

            uri_to_texts.setdefault(s_str, set()).add(txt)

            n = _cso_norm_for_collision(txt)
            if n:
                norm_to_uris.setdefault(n, set()).add(s_str)

    # --- Flatten, Encode, and Save (Same as before) ---
    texts: List[str] = []
    uris: List[str] = []
    ambiguous_mask: List[bool] = []
    dropped_ambig = 0

    for uri, lbls in uri_to_texts.items():
        for txt in sorted(lbls):
            n = _cso_norm_for_collision(txt)
            # Ambiguous if the normalized string maps to > 1 unique URI
            is_ambig = bool(n and len(norm_to_uris.get(n, set())) > 1)

            if is_ambig and drop_ambiguous_short:
                if len(n.replace(" ", "")) <= 4 or (len(n.split()) == 1 and len(n) <= 6):
                    dropped_ambig += 1
                    continue

            uris.append(uri)
            texts.append(txt)
            ambiguous_mask.append(is_ambig)

    if not texts:
        raise RuntimeError(f"CSO parse failed (0 topics extracted). Check file format of {ttl_path}.")

    ambig_n = int(sum(1 for x in ambiguous_mask if x))
    print(
        f"   (CSO) Prepared {len(texts)} label entries "
        f"(ambiguous kept: {ambig_n}, dropped ambiguous-short: {dropped_ambig}). Encoding..."
    )

    try:
        emb = model.encode(
            texts,
            batch_size=int(label_batch_size),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        emb = np.asarray(emb, dtype=np.float32)
    except TypeError:
        emb = model.encode(
            texts,
            batch_size=int(label_batch_size),
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        emb = np.asarray(emb, dtype=np.float32)
        faiss.normalize_L2(emb)

    index = faiss.IndexFlatIP(int(emb.shape[1]))
    index.add(emb)
    faiss.write_index(index, str(cache_index))

    with open(cache_meta, "wb") as f:
        pickle.dump(
            {
                "schema_version": schema_version,
                "ttl_size": int(ttl_stat.st_size),
                "ttl_mtime_ns": int(getattr(ttl_stat, "st_mtime_ns", int(ttl_stat.st_mtime * 1e9))),
                "model_name": str(model_name),
                "uris": uris,
                "texts": texts,
                "ambiguous_mask": ambiguous_mask,
            },
            f,
        )

    return index, uris, texts, ambiguous_mask


def stage02_cso(
    df: pd.DataFrame,
    out_jsonl_gz: Path,
    cso_ttl: Path,
    cso_model: str,
    doc_batch_size: int,
    topk: int,
    sim_threshold: float,
):
    _require_deps_cso()
    from sentence_transformers import SentenceTransformer
    import faiss

    st = SentenceTransformer(cso_model, device=("cuda" if torch.cuda.is_available() else "cpu"))
    index, target_uris, _, ambiguous_mask = prepare_cso_index(
        cso_ttl,
        st,
        out_jsonl_gz.parent / "cso_cache",
        model_name=str(cso_model),
    )

    doc_ids = df["doc_id"].astype(str).tolist()
    docs_text = df["text"].fillna("").astype(str).tolist()

    user_k = int(topk)
    k = max(1, user_k)

    # Stream in chunks to cap memory
    encode_bs = max(1, int(doc_batch_size))
    chunk_size = min(len(docs_text), max(1024, encode_bs * 16))  # larger chunks reduce Python overhead

    print("   (CSO) Encoding + searching documents...")
    with gzip.open(out_jsonl_gz, "wt", encoding="utf-8") as f:
        for start in tqdm(range(0, len(docs_text), chunk_size), desc="Stage 2: CSO batches", unit="batch"):
            end = min(len(docs_text), start + chunk_size)
            batch_texts = docs_text[start:end]

            # Encode + normalize
            try:
                doc_embs = st.encode(
                    batch_texts,
                    batch_size=encode_bs,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                doc_embs = np.asarray(doc_embs, dtype=np.float32)
            except TypeError:
                doc_embs = st.encode(
                    batch_texts,
                    batch_size=encode_bs,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                doc_embs = np.asarray(doc_embs, dtype=np.float32)
                faiss.normalize_L2(doc_embs)

            D, I = index.search(doc_embs, k)

            for bi, doc_id in enumerate(doc_ids[start:end]):
                found: Dict[str, float] = {}
                support: Dict[str, int] = {}
                ambig_hits: List[Tuple[str, float]] = []

                # Collect hits
                for rank in range(k if user_k > 0 else 0):
                    sim = float(D[bi][rank])
                    if sim < float(sim_threshold):
                        continue
                    idx = int(I[bi][rank])
                    if idx < 0 or idx >= len(target_uris):
                        continue
                    uri = str(target_uris[idx])

                    if bool(ambiguous_mask[idx]):
                        ambig_hits.append((uri, sim))
                    else:
                        found[uri] = max(found.get(uri, -1.0), sim)
                        support[uri] = support.get(uri, 0) + 1

                # Collision-resolving heuristic:
                # only accept ambiguous labels if that URI is also supported by at least one non-ambiguous hit.
                if ambig_hits:
                    for uri, sim in ambig_hits:
                        if uri in found or support.get(uri, 0) > 0:
                            found[uri] = max(found.get(uri, -1.0), sim)

                # Write record (keep backward compatible fields)
                semantic = sorted([_cso_slug(u) for u in found.keys()])
                rec = {
                    "doc_id": str(doc_id),
                    "union": semantic,
                    "uris": sorted(list(found.keys())),
                    "weight_uri": {u: float(s) for u, s in found.items()},
                    "weight_union": {_cso_slug(u): float(s) for u, s in found.items()},
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_cso_jsonl_gz(
    path: Path,
    entropy_topm: int = 15,
    entropy_margin: float = 0.05,
    softmax_temp: float = 0.07,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                # Skip malformed lines instead of crashing the whole run.
                continue

            # Prefer new field but keep backward compatibility
            w = data.get("weight_union", {}) or {}
            if not isinstance(w, dict):
                w = {}

            items = []
            try:
                items = sorted([(k, float(v)) for k, v in w.items()], key=lambda x: x[1], reverse=True)[: int(entropy_topm)]
            except Exception:
                items = []

            entropy = 0.0
            if items:
                # Filter by margin from top1
                thr = items[0][1] - float(entropy_margin)
                filtered_items = [x for x in items if x[1] >= thr]

                scores = np.array([x[1] for x in filtered_items], dtype=float)
                if float(softmax_temp) > 0:
                    z = scores / float(softmax_temp)
                    probs = np.exp(z - np.max(z))  # stable softmax
                    denom = float(probs.sum())
                    if denom > 0:
                        probs /= denom
                else:
                    denom = float(scores.sum())
                    probs = scores / (denom if denom > 0 else 1e-12)

                entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

            data["entropy"] = entropy
            data["cso_top1_sim"] = float(items[0][1]) if items else 0.0
            data["cso_top1_label"] = str(items[0][0]) if items else ""
            rows.append(data)
    return pd.DataFrame(rows)

# -----------------------------
# Stage 3: BERTopic
# -----------------------------
@dataclass
class TopicRunResult:
    doc_topics_path: Path
    topics_path: Path
    model_dir: Path
    embeddings_path: Path
    embeddings: Optional[np.ndarray]
    embeddings_doc_ids: Optional[List[str]]


def _compute_sentence_embeddings(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_name, device=("cuda" if torch.cuda.is_available() else "cpu"))

    # Deduplicate while preserving order (faster than sorted(set(...)) on large corpora)
    txt = [str(t) for t in texts]
    codes, uniques = pd.factorize(txt, sort=False)
    u_txt = uniques.tolist()

    u_emb = st.encode(
        u_txt,
        batch_size=int(batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Map back to original order
    emb = u_emb[codes]
    return np.asarray(emb, dtype=np.float32)


def stage03_bertopic(
    df: pd.DataFrame,
    out_doc_topics_csv: Path,
    out_topics_csv: Path,
    model_dir: Path,
    embeddings_path: Path,
    embed_model: str,
    umap_n_neighbors: int,
    umap_n_components: int,
    umap_min_dist: float,
    hdb_min_cluster_size: int,
    hdb_min_samples: int,
    seed: int,
    batch_size: int,
    extra_stopwords: str,
    reduce_topics: str,
    vectorizer_min_df: int = 5,
) -> TopicRunResult:
    """
    Stage 3: BERTopic clustering + topic representations.

    Design goals:
      - Reuse cached embeddings if present; preserve doc_id alignment.
      - Use UMAP(metric='cosine') over embeddings; cluster in UMAP space with HDBSCAN(metric='euclidean') (standard).
      - If topic reduction is enabled, refresh topics/probabilities robustly (prefer model attributes; fall back to transform).
      - Write coherence-friendly 'keywords' derived from BERTopic 'Representation' when available.
    """
    from bertopic import BERTopic
    import umap
    import hdbscan
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

    # ---- inputs
    texts = df["text"].astype(str).tolist()
    doc_ids = df["doc_id"].astype(str).tolist()
    ensure_dir(embeddings_path.parent)

    # ---- embeddings: load if possible (and reorder to match df); else compute+save
    emb: np.ndarray
    if embeddings_path.exists():
        emb_loaded, emb_doc_ids = load_embeddings_with_doc_ids(embeddings_path)
        emb_doc_ids = [str(x) for x in emb_doc_ids]
        # Reorder if needed (robust to prior runs / different df order)
        if emb_loaded is not None and len(emb_doc_ids) == len(doc_ids) and set(emb_doc_ids) == set(doc_ids):
            if emb_doc_ids != doc_ids:
                idx = {d: i for i, d in enumerate(emb_doc_ids)}
                emb = np.vstack([emb_loaded[idx[d]] for d in doc_ids])
            else:
                emb = emb_loaded
        else:
            # Fallback: recompute if cache is missing/misaligned
            print("   (WARN) Cached embeddings exist but do not align with current doc_id list; recomputing.")
            emb = _compute_sentence_embeddings(texts, embed_model, batch_size)
            save_embeddings_with_doc_ids(embeddings_path, emb, doc_ids)
    else:
        emb = _compute_sentence_embeddings(texts, embed_model, batch_size)
        save_embeddings_with_doc_ids(embeddings_path, emb, doc_ids)

    # ---- vectorizer / stopwords
    extras = [w.strip() for w in str(extra_stopwords or "").split(",") if w.strip()]
    stop_list = list(ENGLISH_STOP_WORDS.union(set(extras)))
    vectorizer_model = CountVectorizer(
        stop_words=stop_list,
        min_df=int(vectorizer_min_df),
        ngram_range=(1, 2),
    )

    # ---- UMAP + HDBSCAN
    umap_model = umap.UMAP(
        n_neighbors=int(umap_n_neighbors),
        n_components=int(umap_n_components),
        min_dist=float(umap_min_dist),
        metric="cosine",
        random_state=int(seed),
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=int(hdb_min_cluster_size),
        min_samples=int(hdb_min_samples),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(texts, emb)

    # ---- optional topic reduction
    rt = str(reduce_topics).strip().lower()
    if rt not in {"none", "0", "false", "no"}:
        nr = "auto" if rt == "auto" else int(rt)
        try:
            topic_model.reduce_topics(texts, nr_topics=nr)
            topics = getattr(topic_model, "topics_", topics)

            # probabilities may not always be updated; prefer model attr if valid, else transform
            probs2 = getattr(topic_model, "probabilities_", None)
            if probs2 is not None:
                try:
                    import numpy as _np
                    probs2 = _np.asarray(probs2)
                    if probs2.ndim == 2 and probs2.shape[0] == len(texts):
                        probs = probs2
                except Exception:
                    pass

            if probs is None or (hasattr(probs, "shape") and len(getattr(probs, "shape", ())) == 0):
                try:
                    _, probs_t = topic_model.transform(texts, emb)
                    probs = probs_t
                except Exception:
                    pass
        except Exception as e:
            print(f"   (WARN) Topic reduction failed; continuing without reduction: {e}")

    # ---- doc-topic outputs
    doc_topics = df[["doc_id"]].copy()
    if "year" in df.columns:
        doc_topics["year"] = df["year"].values
    if "thesis_type" in df.columns:
        doc_topics["thesis_type"] = df["thesis_type"].values

    doc_topics["topic"] = topics

    # Probability: max over topic probs if available; else 0
    prob = 0.0
    try:
        import numpy as _np
        if probs is None:
            prob = _np.zeros(len(texts), dtype=float)
        else:
            probs_arr = _np.asarray(probs)
            if probs_arr.ndim == 2:
                prob = probs_arr.max(axis=1)
            else:
                prob = probs_arr.astype(float)
            prob = _np.clip(prob, 0.0, 1.0)
    except Exception:
        prob = np.zeros(len(texts), dtype=float)

    doc_topics["probability"] = prob
    doc_topics["assignment_source"] = np.where(doc_topics["topic"] == -1, "outlier", "bertopic")
    doc_topics["assignment_confidence"] = np.where(doc_topics["topic"] == -1, 0.0, doc_topics["probability"])
    # keep columns used later in the pipeline (harmless defaults here)
    if "similarity_score" not in doc_topics.columns:
        doc_topics["similarity_score"] = doc_topics["assignment_confidence"]
    if "membership_score" not in doc_topics.columns:
        doc_topics["membership_score"] = doc_topics["assignment_confidence"]

    doc_topics.to_csv(out_doc_topics_csv, index=False)

    # ---- topic info + coherence-friendly keywords
    ti = topic_model.get_topic_info()
    # Ensure a stable keywords column for evaluators (from Representation if available)
    if "Representation" in ti.columns and "keywords" not in ti.columns:
        def _repr_to_keywords(x):
            if isinstance(x, list):
                return ", ".join(map(str, x))
            return str(x)
        ti["keywords"] = ti["Representation"].apply(_repr_to_keywords)

    ti = ti.rename(columns={"Topic": "topic", "Count": "size", "Name": "label"})
    ti.to_csv(out_topics_csv, index=False)

    # ---- save model
    ensure_dir(model_dir)
    try:
        topic_model.save(str(model_dir), serialization="safetensors", save_ctfidf=True)
    except Exception as e:
        print(f"   (WARN) Could not save BERTopic model: {e}")

    return TopicRunResult(out_doc_topics_csv, out_topics_csv, model_dir, embeddings_path, emb, doc_ids)





# -----------------------------
# Stage 4: Outlier Reassignment
# -----------------------------
def stage04_outlier_reassign(
    doc_topics: pd.DataFrame,
    embeddings: np.ndarray,
    doc_id_to_emb_idx: Dict[str, int],
    cso_df: Optional[pd.DataFrame],
    out_csv: Path,
    max_outliers: int = 2000,
    sim_threshold: float = 0.65,
    entropy_threshold: float = 1.0,
    margin_threshold: float = 0.0,
    margin_auto: bool = True,
    margin_quantile: float = 0.85,
    entropy_gate: bool = False,
    entropy_auto: bool = False,
    entropy_quantile: float = 0.85,
    cso_sim_threshold_gate: float = 0.85,
) -> pd.DataFrame:
    dt = doc_topics.copy()
    dt["doc_id"] = dt["doc_id"].astype(str)

    # Normalize columns
    if "membership_score" not in dt.columns and "probability" in dt.columns:
        dt["membership_score"] = pd.to_numeric(dt["probability"], errors="coerce").fillna(0.0)

    # Calculate Centroids
    assigned = dt[dt["topic"] != -1].copy()
    if assigned.empty:
        dt.to_csv(out_csv, index=False)
        return dt

    assigned["emb_idx"] = assigned["doc_id"].map(doc_id_to_emb_idx)
    assigned = assigned.dropna(subset=["emb_idx"]).copy()
    assigned["emb_idx"] = assigned["emb_idx"].astype(int)

    centroids: Dict[int, np.ndarray] = {}
    for tid, grp in assigned.groupby("topic"):
        idxs = grp["emb_idx"].values
        if len(idxs) < 2:
            continue
        vec = embeddings[idxs].mean(axis=0)
        centroids[int(tid)] = vec

    if not centroids:
        dt.to_csv(out_csv, index=False)
        return dt

    topic_ids = np.array(sorted(centroids.keys()), dtype=int)
    topic_vecs = np.vstack([centroids[t] for t in topic_ids]).astype(np.float32)
    topic_vecs = l2_normalize(topic_vecs)

    # Identify Outliers
    outliers = dt[dt["topic"] == -1].copy()
    if outliers.empty:
        dt.to_csv(out_csv, index=False)
        return dt

    outliers["emb_idx"] = outliers["doc_id"].map(doc_id_to_emb_idx)
    outliers = outliers.dropna(subset=["emb_idx"]).copy()
    outliers["emb_idx"] = outliers["emb_idx"].astype(int)

    if int(max_outliers or 0) > 0 and len(outliers) > int(max_outliers):
        outliers = outliers.iloc[: int(max_outliers)].copy()

    # Calculate Similarity: (N_out, K)
    out_vecs = l2_normalize(embeddings[outliers["emb_idx"].values].astype(np.float32))
    sims = np.dot(out_vecs, topic_vecs.T)

    best_idx = np.argmax(sims, axis=1)
    best_sim = sims[np.arange(sims.shape[0]), best_idx]

    # Calculate Margin
    if sims.shape[1] >= 2:
        part = np.partition(sims, kth=-2, axis=1)
        top2 = part[:, -2:]
        second_best = np.min(top2, axis=1)
        margin = best_sim - second_best
    else:
        margin = np.ones_like(best_sim)

    # Dynamic thresholding
    eff_margin_thr = float(margin_threshold)
    if bool(margin_auto):
        cand = margin[best_sim >= float(sim_threshold)]
        if cand.size > 0:
            q = min(max(float(margin_quantile), 0.0), 1.0)
            eff_margin_thr = float(np.quantile(cand, q))
        eff_margin_thr = max(eff_margin_thr, 0.0)

    # Filter
    mask = (best_sim >= float(sim_threshold)) & (margin >= float(eff_margin_thr))

    if bool(entropy_gate) and (cso_df is not None) and (not cso_df.empty):
        try:
            c = cso_df.copy()
            c["doc_id"] = c["doc_id"].astype(str)
            if "cso_top1_sim" in c.columns and "entropy" in c.columns:
                c = c.set_index("doc_id", drop=False)
                out_ids = outliers["doc_id"].astype(str).tolist()
                cso_sim = np.array([float(c.loc[d, "cso_top1_sim"]) if d in c.index else np.nan for d in out_ids], dtype=float)
                ent = np.array([float(c.loc[d, "entropy"]) if d in c.index else np.nan for d in out_ids], dtype=float)

                eff_ent_thr = float(entropy_threshold)
                if bool(entropy_auto):
                    vals = ent[(cso_sim >= float(cso_sim_threshold_gate)) & np.isfinite(ent)]
                    if vals.size > 0:
                        q = min(max(float(entropy_quantile), 0.0), 1.0)
                        eff_ent_thr = float(np.quantile(vals, q))

                protect = (cso_sim >= float(cso_sim_threshold_gate)) & (ent <= float(eff_ent_thr)) & np.isfinite(ent)
                if protect.any():
                    mask = mask & (~protect)
                    print(
                        f"   (Stage 4) CSO gate protected {int(protect.sum())} outliers "
                        f"(top1_sim >= {float(cso_sim_threshold_gate):.3f}, entropy <= {float(eff_ent_thr):.3f})."
                    )
        except Exception:
            pass


    if mask.any():
        chosen_topics = topic_ids[best_idx[mask]]
        chosen_sims = best_sim[mask]
        chosen_margins = margin[mask]
        out_doc_ids = outliers.loc[mask, "doc_id"].astype(str).tolist()

        idxs = dt.index[dt["doc_id"].isin(out_doc_ids)]
        m_topic = dict(zip(out_doc_ids, chosen_topics.astype(int).tolist()))
        m_sim = dict(zip(out_doc_ids, chosen_sims.astype(float).tolist()))
        m_mar = dict(zip(out_doc_ids, chosen_margins.astype(float).tolist()))

        dt.loc[idxs, "topic"] = dt.loc[idxs, "doc_id"].map(m_topic).astype(int)
        dt.loc[idxs, "similarity_score"] = dt.loc[idxs, "doc_id"].map(m_sim).astype(float)
        dt.loc[idxs, "similarity_margin"] = dt.loc[idxs, "doc_id"].map(m_mar).astype(float)
        dt.loc[idxs, "assignment_source"] = "reassign"

        # Penalize confidence by margin
        if eff_margin_thr > 0:
            scale = np.minimum(1.0, dt.loc[idxs, "similarity_margin"].astype(float) / (eff_margin_thr + 1e-12))
        else:
            scale = 1.0
        conf = np.clip(dt.loc[idxs, "similarity_score"].astype(float) * scale, 0.0, 1.0)
        dt.loc[idxs, "assignment_confidence"] = conf
        if "membership_score" in dt.columns:
            dt.loc[idxs, "membership_score"] = conf
        if "probability" in dt.columns:
            dt.loc[idxs, "probability"] = conf

        print(
            f"   (Stage 4) Reassigned {int(mask.sum())} docs "
            f"(sim >= {float(sim_threshold):.3f}, margin >= {float(eff_margin_thr):.3f})"
        )
    else:
        print("   (Stage 4) No outliers reassigned.")

    dt.to_csv(out_csv, index=False)
    return dt


def stage05_llm_gate(
    doc_topics: pd.DataFrame,
    topics_df: pd.DataFrame,
    df_text: pd.DataFrame,
    embeddings: np.ndarray,
    doc_id_to_emb_idx: Dict[str, int],
    out_csv: Path,
    llm_mode: str,
    llm_model: str,
    max_outliers: int,
    top_k_candidates: int,
    sim_floor: float,
    cso_df: Optional[pd.DataFrame] = None,
    entropy_threshold: float = 1.0,
    low_sim_threshold: float = 0.45,
    reassign_margin_threshold: float = 0.02,
    entropy_auto: bool = False,
    entropy_quantile: float = 0.85,
    cso_sim_threshold: float = 0.65,
    cso_entropy_margin: float = 0.02,
    entropy_gate: bool = False,
    ollama_url: str = "http://localhost:11434",
    ollama_workers: int = 4,
    ollama_timeout: int = 600,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dt = doc_topics.copy()
    dt["doc_id"] = dt["doc_id"].astype(str)

    if str(llm_mode).lower() == "none":
        dt.to_csv(out_csv, index=False)
        return dt, dt[dt["topic"] == -1].copy()

    dt_valid = dt.copy()
    dt_valid["emb_idx"] = dt_valid["doc_id"].map(doc_id_to_emb_idx)
    dt_valid = dt_valid.dropna(subset=["emb_idx"]).copy()
    dt_valid["emb_idx"] = dt_valid["emb_idx"].astype(int)

    outliers_valid = dt_valid[dt_valid["topic"] == -1].copy()

    if bool(entropy_gate) and (cso_df is not None) and (not cso_df.empty):
        try:
            c = cso_df.copy()
            c["doc_id"] = c["doc_id"].astype(str)
            if "cso_top1_sim" in c.columns and "entropy" in c.columns:
                c = c.set_index("doc_id", drop=False)
                out_ids = outliers_valid["doc_id"].astype(str).tolist()
                cso_sim = np.array([float(c.loc[d, "cso_top1_sim"]) if d in c.index else np.nan for d in out_ids], dtype=float)
                ent = np.array([float(c.loc[d, "entropy"]) if d in c.index else np.nan for d in out_ids], dtype=float)

                eff_ent_thr = float(entropy_threshold)
                sim_thr_eff = float(cso_sim_threshold) + float(cso_entropy_margin)
                if bool(entropy_auto):
                    vals = ent[(cso_sim >= sim_thr_eff) & np.isfinite(ent)]
                    if vals.size > 0:
                        q = min(max(float(entropy_quantile), 0.0), 1.0)
                        eff_ent_thr = float(np.quantile(vals, q))

                protect = (cso_sim >= sim_thr_eff) & (ent <= float(eff_ent_thr)) & np.isfinite(ent)
                if protect.any():
                    protected_ids = outliers_valid.loc[protect, "doc_id"].astype(str).tolist()
                    outliers_valid = outliers_valid.loc[~protect].copy()
                    try:
                        dt.loc[dt["doc_id"].isin(protected_ids), "llm_skip_reason"] = "cso_entropy_gate"
                        dt.loc[dt["doc_id"].isin(protected_ids), "outlier_protected"] = True
                    except Exception:
                        pass
                    print(
                        f"   (Stage 5) CSO gate skipped {int(len(protected_ids))} outliers for LLM "
                        f"(top1_sim >= {float(sim_thr_eff):.3f}, entropy <= {float(eff_ent_thr):.3f}); preserved for augmentation."
                    )
        except Exception:
            pass

    assigned_valid = dt_valid[dt_valid["topic"] != -1].copy()

    if outliers_valid.empty or assigned_valid.empty:
        dt.to_csv(out_csv, index=False)
        return dt, outliers_valid

    if int(max_outliers or 0) > 0 and len(outliers_valid) > int(max_outliers):
        outliers_valid = outliers_valid.iloc[: int(max_outliers)].copy()

    cents: Dict[int, np.ndarray] = {}
    for t, grp in assigned_valid.groupby("topic"):
        idxs = grp["emb_idx"].values
        cents[int(t)] = embeddings[idxs].mean(axis=0)

    tids = sorted(cents.keys())
    C = l2_normalize(np.stack([cents[t] for t in tids]).astype(np.float32))

    outliers_valid = outliers_valid.reset_index().rename(columns={"index": "dt_index"})
    X = l2_normalize(embeddings[outliers_valid["emb_idx"].values].astype(np.float32))

    sims = X @ C.T

    prompts: List[str] = []
    metas: List[Tuple[int, List[Tuple[int, float]]]] = []  # (dt_index, candidates)

    df_text_idx = df_text.copy()
    df_text_idx["doc_id"] = df_text_idx["doc_id"].astype(str)
    df_text_idx = df_text_idx.set_index("doc_id", drop=False)

    topic_labels_map = {}
    for _, r in topics_df.iterrows():
        if int(r.get("topic", -1)) != -1:
            topic_labels_map[int(r["topic"])] = str(r.get("label", f"Topic {r['topic']}"))
    # ---- Helper: build richer candidate "topic cards" (label + keywords) ----
    topics_df_idx = topics_df.copy()
    try:
        if "topic" in topics_df_idx.columns:
            topics_df_idx["topic"] = topics_df_idx["topic"].astype(int)
    except Exception:
        pass
    if "topic" in topics_df_idx.columns:
        topics_df_idx = topics_df_idx.set_index("topic", drop=False)

    def _parse_terms(val: Any) -> List[str]:
        """Parse BERTopic topic-info cells that may contain lists or list-like strings."""
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            return [str(x).strip() for x in val if str(x).strip()]
        s = str(val).strip()
        if not s:
            return []
        # Try literal eval for strings like "['a', 'b']"
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
        # Fallback: split on commas
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]

    def _topic_keywords(tid: int, limit: int = 10) -> List[str]:
        """Return up to `limit` keywords for a topic, if available."""
        try:
            if "topic" in topics_df_idx.columns and tid in topics_df_idx.index:
                row = topics_df_idx.loc[tid]
                for col in ("Representation", "representation", "Top_n_words", "top_n_words", "keywords", "KeyBERT", "keybert"):
                    if col in topics_df_idx.columns:
                        terms = _parse_terms(row.get(col))
                        if terms:
                            return terms[: int(limit)]
        except Exception:
            pass
        return []

    for i in tqdm(range(len(outliers_valid)), total=len(outliers_valid), desc="Stage 5: Build Prompts", unit="doc"):
        row = outliers_valid.iloc[i]
        row_sims = sims[i]

        best_sim = float(np.max(row_sims))
        if best_sim < float(sim_floor):
            continue

        top_k_idx = row_sims.argsort()[::-1][: int(top_k_candidates)]
        cands = [(int(tids[k]), float(row_sims[k])) for k in top_k_idx if float(row_sims[k]) > float(sim_floor)]
        if not cands:
            continue

        doc_id = str(row["doc_id"])
        if doc_id not in df_text_idx.index:
            continue

        txt = str(df_text_idx.loc[doc_id, "text"])
        cand_cards = []
        for tid, _s in cands:
            lbl = topic_labels_map.get(tid, "Topic " + str(tid))
            kws = _topic_keywords(tid, limit=10)
            kw_txt = ", ".join(kws) if kws else ""
            if kw_txt:
                cand_cards.append(f"[{tid}] {lbl}\n  Keywords: {kw_txt}")
            else:
                cand_cards.append(f"[{tid}] {lbl}")
        cands_text = "\n".join(cand_cards)
        cand_ids_list = [c[0] for c in cands]

        p = f"""Thesis abstract:
{txt[:1200]}

Candidate topics:
{cands_text}

Choose the best matching candidate. If none match well, choose -1.

Return ONLY JSON: {{\"topic_id\": <one of {cand_ids_list} or -1>}}

(If you absolutely cannot return JSON, return ONLY the exact candidate label text or 'Outlier'.)
"""
        prompts.append(p)
        metas.append((int(row["dt_index"]), cands))

    called_dt_indices = [m[0] for m in metas]

    if called_dt_indices:
        dt.loc[called_dt_indices, "llm_called"] = True

    if not prompts:
        dt.to_csv(out_csv, index=False)
        return dt, dt[dt["topic"] == -1].copy()

    llm_system_prompt = (
        "You are a strict topic assignment assistant in Computer Science domain. "
        "Follow the user's formatting requirements exactly. "
        "Output must be either a single JSON object with key \"topic_id\", "
        "or, if none of the candidates match, a JSON object {\"topic_id\": -1}. "
        "Do not include any other text."
    )
    if str(llm_mode).lower() == "ollama":
        resps = llm_choose_topic_ollama_batch(
            prompts,
            llm_model,
            host=ollama_url,
            system_prompt=llm_system_prompt,
            max_workers=ollama_workers,
            timeout=ollama_timeout,
        )
    else:
        tok, mdl = load_llm_transformers(llm_model)
        resps = llm_choose_topic_transformers_batch(tok, mdl, prompts)

    def _clean_label_fallback(lbl: str) -> str:
        """Clean label-like outputs for robust candidate matching."""
        if lbl is None:
            return ""
        s = str(lbl).strip()
        if not s:
            return ""
        s = s.replace("```json", "").replace("```", "").strip()
        s = re.sub(r"^\s*\d+\s*[\.:\)]\s*", "", s)  # remove leading numbering like "1. "
        s = re.sub(r"^\s*(Topic|Label)\s*[:\-]\s*", "", s, flags=re.IGNORECASE)
        s = s.strip().strip(" \'\"`").strip()
        s = re.sub(r"\s+", " ", s).strip()
        return s.lower()


    def _extract_choice(text: str, candidates: Set[int], label_to_id: Dict[str, int]) -> int:
        s = str(text or "").strip()
        s = re.sub(r"```json", "", s, flags=re.IGNORECASE)
        s = re.sub(r"```", "", s)
        s = s.strip()

        low_raw = s.lower()
        # explicit abstentions
        if any(x in low_raw for x in ["outlier", "no match", "unknown", "n/a", "uncertain", "none"]):
            return -1

        # --- 1) JSON-first (strict, candidate-checked)
        def _json_pick(obj) -> Optional[int]:
            if not isinstance(obj, dict):
                return None
            val = obj.get("topic_id")
            if val is None:
                val = obj.get("topic") or obj.get("id")
            if val is None:
                return None
            try:
                ival = int(val)
            except Exception:
                return None
            if ival == -1:
                return -1
            return ival if ival in candidates else None

        try:
            obj = json.loads(s)
            pick = _json_pick(obj)
            if pick is not None:
                return pick
        except Exception:
            pass

        # 1b) embedded JSON object inside text
        m = re.search(r"\{.*?\}", s, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                pick = _json_pick(obj)
                if pick is not None:
                    return pick
            except Exception:
                pass

        # --- 2) Numeric fallback (GUARDED)
        nums: List[int] = []
        for n_str in re.findall(r"(-?\d+)", s):
            try:
                nums.append(int(n_str))
            except Exception:
                continue

        if -1 in nums:
            return -1

        candidate_hits = [n for n in nums if n in candidates]
        candidate_hits = list(dict.fromkeys(candidate_hits))  # unique

        if len(candidate_hits) == 1:
            return int(candidate_hits[0])
        if len(candidate_hits) > 1:
            return -1

        # --- 3) Label fallback (conservative, candidate-restricted)
        low = _clean_label_fallback(s)
        if not low:
            return -1
        if low in {"outlier", "none", "no match", "unknown", "n/a", "uncertain"}:
            return -1

        # exact label match
        if low in label_to_id:
            tid = int(label_to_id[low])
            return tid if tid in candidates else -1

        # unique substring match
        subs: List[int] = []
        for k, tid in label_to_id.items():
            if not k:
                continue
            if k in low or low in k:
                subs.append(int(tid))
        subs = list(dict.fromkeys(subs))
        if len(subs) == 1 and subs[0] in candidates:
            return int(subs[0])
        if len(subs) > 1:
            return -1

        # fuzzy match among candidate labels only (high cutoff)
        try:
            import difflib
            keys = [k for k, tid in label_to_id.items() if int(tid) in candidates and k]
            if keys:
                matches = difflib.get_close_matches(low, keys, n=2, cutoff=0.85)
                if matches:
                    tid1 = int(label_to_id[matches[0]])
                    if tid1 in candidates:
                        return tid1
        except Exception:
            pass

        return -1
    parsed_count = 0
    for (dt_index, cands), raw_resp in tqdm(zip(metas, resps), total=len(metas), desc="Stage 5: Parse LLM", unit="doc"):
        valid_ids = {c[0] for c in cands}
        label_to_id: Dict[str, int] = {}
        for _tid, _s in cands:
            _lbl = topic_labels_map.get(_tid, "Topic " + str(_tid))
            _cl = _clean_label_fallback(_lbl)
            if _cl:
                label_to_id[_cl] = int(_tid)

        choice = _extract_choice(str(raw_resp), valid_ids, label_to_id)

        if choice != -1 and choice in valid_ids:
            parsed_count += 1
            sim_val = next((c[1] for c in cands if c[0] == choice), 0.0)

            dt.loc[dt_index, "topic"] = choice
            dt.loc[dt_index, "assignment_source"] = "llm_gate"
            dt.loc[dt_index, "similarity_score"] = sim_val
            dt.loc[dt_index, "assignment_confidence"] = sim_val

            if "membership_score" in dt.columns:
                 dt.loc[dt_index, "membership_score"] = sim_val
            if "probability" in dt.columns:
                 dt.loc[dt_index, "probability"] = sim_val

    print(f"   (Stage 5) LLM reassigned {parsed_count}/{len(prompts)} outlier documents.")
    dt.to_csv(out_csv, index=False)
    return dt, dt[dt["topic"] == -1].copy()


# -----------------------------
# Stage 6: Augment residuals
# -----------------------------
_DEFAULT_GENERIC_CSO_BLOCKLIST = {
    "computer science", "computer sciences", "computer system", "computer systems",
    "software", "software engineering", "information system", "information systems",
    "engineering", "data", "dataset", "datasets", "algorithm", "algorithms",
    "machine learning", "artificial intelligence"
}

@dataclass
class SymbolicDiscoveryResult:
    mapping: Dict[str, Tuple[int, float]]
    topics: pd.DataFrame
    next_topic_id: int

def discover_symbolic_topics_from_cso(
    residual_f: pd.DataFrame,
    df_idx: pd.DataFrame,
    cso_path: Path,
    min_size: int,
    top1_sim_min: float,
    max_topics: int,
    next_topic_id: int,
    embeddings: Optional[np.ndarray],
    doc_id_to_emb_idx: Optional[Dict[str, int]],
    existing_centroids: Optional[Dict[int, np.ndarray]],
    zombie_sim_threshold: float = 0.90,
    generic_blocklist: Optional[Set[str]] = None,
) -> SymbolicDiscoveryResult:
    mapping: Dict[str, Tuple[int, float]] = {}
    topics = pd.DataFrame(columns=["topic", "label", "keywords", "size", "source"])

    if min_size <= 0 or not cso_path.exists() or residual_f.empty:
        return SymbolicDiscoveryResult(mapping, topics, next_topic_id)

    try:
        cso_stats = pd.read_csv(cso_path)
    except Exception:
        return SymbolicDiscoveryResult(mapping, topics, next_topic_id)

    cso_stats["doc_id"] = cso_stats["doc_id"].astype(str)
    need_ids = set(residual_f["doc_id"].astype(str).tolist())
    cso_stats = cso_stats[cso_stats["doc_id"].isin(need_ids)].copy()

    if cso_stats.empty:
        return SymbolicDiscoveryResult(mapping, topics, next_topic_id)

    def _get_label_sim(row):
        lbl = str(row.get("cso_top1_label") or "").strip()
        sim = row.get("cso_top1_sim", 0.0)
        return lbl, float(sim) if not np.isnan(sim) else 0.0

    cso_stats["_lbl"], cso_stats["_sim"] = zip(*cso_stats.apply(_get_label_sim, axis=1))

    block = set(x.lower() for x in _DEFAULT_GENERIC_CSO_BLOCKLIST)
    if generic_blocklist:
        block.update(x.lower() for x in generic_blocklist)

    mask = (cso_stats["_lbl"].str.len() > 0) & \
           (~cso_stats["_lbl"].str.lower().isin(block)) & \
           (cso_stats["_sim"] >= top1_sim_min)

    valid_cso = cso_stats[mask].copy()
    if valid_cso.empty:
        return SymbolicDiscoveryResult(mapping, topics, next_topic_id)

    counts = valid_cso["_lbl"].value_counts()
    promotable = counts[counts >= min_size].sort_values(ascending=False).head(max_topics)

    if promotable.empty:
        return SymbolicDiscoveryResult(mapping, topics, next_topic_id)

    old_mat = None
    old_ids = []
    if embeddings is not None and existing_centroids:
        old_ids = list(existing_centroids.keys())
        if old_ids:
            old_mat = np.stack([existing_centroids[tid] for tid in old_ids])
            old_mat = l2_normalize(old_mat)

    new_topic_list = []
    new_centroids_list = []
    label_to_tid = {}

    current_tid = next_topic_id
    zombie_count = 0

    for lbl in promotable.index:
        doc_ids_in_concept = valid_cso.loc[valid_cso["_lbl"] == lbl, "doc_id"].tolist()

        cand_vec = None
        if embeddings is not None and doc_id_to_emb_idx:
            e_idxs = [doc_id_to_emb_idx[d] for d in doc_ids_in_concept if d in doc_id_to_emb_idx]
            if e_idxs:
                cand_vec = embeddings[e_idxs].mean(axis=0).reshape(1, -1)
                cand_vec = l2_normalize(cand_vec)

        target_tid = -1
        is_zombie = False

        if cand_vec is not None:
            if old_mat is not None:
                sims = np.dot(cand_vec, old_mat.T)[0]
                best_idx = np.argmax(sims)
                if sims[best_idx] >= zombie_sim_threshold:
                    target_tid = old_ids[best_idx]
                    is_zombie = True

            if not is_zombie and new_centroids_list:
                new_mat = np.vstack(new_centroids_list)
                sims = np.dot(cand_vec, new_mat.T)[0]
                best_idx = np.argmax(sims)
                if sims[best_idx] >= zombie_sim_threshold:
                     target_tid = new_topic_list[best_idx]["topic"]
                     is_zombie = True

        if is_zombie:
            label_to_tid[lbl] = target_tid
            zombie_count += 1
        else:
            label_to_tid[lbl] = current_tid
            new_topic_list.append({
                "topic": int(current_tid),
                "label": lbl,
                "keywords": lbl,
                "size": int(promotable[lbl]),
                "source": "cso_symbolic"
            })
            if cand_vec is not None:
                new_centroids_list.append(cand_vec)
            current_tid += 1

    if zombie_count > 0:
        print(f"   (Symbolic) Merged {zombie_count} zombie concepts into existing topics.")

    valid_cso["assigned_topic"] = valid_cso["_lbl"].map(label_to_tid)
    valid_cso = valid_cso.dropna(subset=["assigned_topic"])

    for _, row in valid_cso.iterrows():
        mapping[row["doc_id"]] = (int(row["assigned_topic"]), float(row["_sim"]))

    return SymbolicDiscoveryResult(mapping, pd.DataFrame(new_topic_list), current_tid)


def ctfidf_keywords(
    texts: List[str],
    labels: np.ndarray,
    topn: int = 12,
    weights: Optional[np.ndarray] = None,
    extra_stopwords: str = "",
) -> Dict[int, List[Tuple[str, float]]]:
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

    extras = [w.strip() for w in str(extra_stopwords or "").split(",") if w.strip()]
    stop = list(ENGLISH_STOP_WORDS.union(set(extras)))

    vec = CountVectorizer(ngram_range=(1, 2), stop_words=stop, min_df=2)
    try:
        X = vec.fit_transform([str(t) for t in texts])
    except ValueError:
        return {}

    terms = np.array(vec.get_feature_names_out())

    if weights is not None:
        w = np.asarray(weights).reshape(-1)
        if X.shape[0] == w.shape[0]:
            from scipy.sparse import spdiags
            D = spdiags(w, 0, X.shape[0], X.shape[0])
            X = D * X

    labels_i = np.asarray(labels, dtype=int)
    clusters = sorted(list(set(labels_i) - {-1}))

    out = {}
    if not clusters:
        return out

    rows = []
    for c in clusters:
        idx = np.where(labels_i == c)[0]
        if len(idx) == 0:
            rows.append(np.zeros(len(terms)))
        else:
            rows.append(np.asarray(X[idx].sum(axis=0)).flatten())

    M = np.vstack(rows)

    tf = M / np.maximum(M.sum(axis=1, keepdims=True), 1e-12)
    df = np.sum(M > 0, axis=0)
    idf = np.log((1 + len(clusters)) / (1 + df)) + 1

    ctfidf = tf * idf

    for i, c in enumerate(clusters):
        row = ctfidf[i]
        top_idx = np.argsort(row)[::-1][:topn]
        out[c] = [(terms[j], float(row[j])) for j in top_idx if row[j] > 0]

    return out



def stage06_augment_residuals(
    doc_topics: pd.DataFrame,
    residual: pd.DataFrame,
    df_text: pd.DataFrame,
    embeddings: np.ndarray,
    doc_id_to_emb_idx: Dict[str, int],
    out_doc_topics_csv: Path,
    out_new_topics_csv: Path,
    min_cluster_size: int,
    min_samples: int,
    quality_min_topword_weight: float,
    run_dir: Path,
    augment_cso_purity_min: float = 0.0,
    symbolic_min_cluster_size: int = 0,
    symbolic_top1_sim_min: float = 0.55,
    symbolic_max_topics: int = 200,
    extra_stopwords: str = "",
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    umap_min_dist: float = 0.0,
    cluster_selection_method: str = "leaf",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    import hdbscan

    df_idx = df_text.set_index("doc_id", drop=False)
    residual_f = residual[residual["doc_id"].isin(df_idx.index)].copy()
    residual_f["emb_idx"] = residual_f["doc_id"].map(doc_id_to_emb_idx).fillna(-1).astype(int)
    residual_f = residual_f[residual_f["emb_idx"] != -1].copy()

    # 1. Symbolic Discovery
    symbolic_topics = pd.DataFrame()

    max_tid = -1
    if (doc_topics["topic"] != -1).any():
        max_tid = doc_topics["topic"].max()
    next_tid = max_tid + 1

    if symbolic_min_cluster_size > 0:
        cso_path = run_dir / "03_cso_entropy_stats.csv"

        existing_centroids = {}
        assigned = doc_topics[doc_topics["topic"] != -1]
        for t, grp in assigned.groupby("topic"):
            eidxs = [doc_id_to_emb_idx[d] for d in grp["doc_id"] if d in doc_id_to_emb_idx]
            if len(eidxs) > 2:
                existing_centroids[int(t)] = embeddings[eidxs].mean(axis=0)

        sym_res = discover_symbolic_topics_from_cso(
            residual_f=residual_f,
            df_idx=df_idx,
            cso_path=cso_path,
            min_size=symbolic_min_cluster_size,
            top1_sim_min=symbolic_top1_sim_min,
            max_topics=symbolic_max_topics,
            next_topic_id=next_tid,
            embeddings=embeddings,
            doc_id_to_emb_idx=doc_id_to_emb_idx,
            existing_centroids=existing_centroids,
            zombie_sim_threshold=0.90
        )

        symbolic_topics = sym_res.topics
        next_tid = sym_res.next_topic_id

        if sym_res.mapping:
            dt_map = pd.DataFrame.from_dict(sym_res.mapping, orient='index', columns=['topic', 'conf'])
            mask = doc_topics["doc_id"].isin(dt_map.index) & (doc_topics["topic"] == -1)

            for doc_id, (tid, conf) in sym_res.mapping.items():
                idx = doc_topics.index[doc_topics["doc_id"] == doc_id]
                if not idx.empty and doc_topics.loc[idx[0], "topic"] == -1:
                    doc_topics.loc[idx, "topic"] = tid
                    doc_topics.loc[idx, "assignment_source"] = "cso_symbolic"
                    doc_topics.loc[idx, "similarity_score"] = conf
                    doc_topics.loc[idx, "assignment_confidence"] = conf

            residual_f = residual_f[~residual_f["doc_id"].isin(dt_map.index)]

    # 2. Dense Clustering on Remaining Residuals
    if len(residual_f) > max(10, min_cluster_size):
        X = embeddings[residual_f["emb_idx"].values]

        try:
            import umap
            reducer = umap.UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, min_dist=umap_min_dist, metric='cosine', random_state=seed)
            X_red = reducer.fit_transform(X)
        except Exception:
            from sklearn.decomposition import PCA
            X_red = PCA(n_components=min(50, X.shape[1])).fit_transform(X)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method=cluster_selection_method
        )
        labels = clusterer.fit_predict(X_red)

        texts = [str(df_idx.loc[d, "text"]) for d in residual_f["doc_id"]]
        kw = ctfidf_keywords(texts, labels, extra_stopwords=extra_stopwords)

        keep_clusters = []
        cluster_qc = {}

        def _cluster_cso_purity(doc_ids: List[str]) -> Tuple[float, float, str]:
            if "cso_top1_label" not in df_idx.columns:
                return (float("nan"), 0.0, "")
            vals: List[str] = []
            for d in doc_ids:
                try:
                    v = df_idx.loc[d, "cso_top1_label"]
                except Exception:
                    continue
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                s = str(v).strip()
                if s:
                    vals.append(s)
            if not vals:
                return (0.0, 0.0, "")
            from collections import Counter
            cnt = Counter(vals)
            top_lbl, top_n = cnt.most_common(1)[0]
            cov = float(len(vals)) / float(max(1, len(doc_ids)))
            purity = float(top_n) / float(len(vals))
            eff = purity * min(1.0, cov / 0.50)
            return (float(eff), float(cov), str(top_lbl))

        for c, words in kw.items():
            if c == -1:
                continue
            if (not words) or (words[0][1] is None) or (float(words[0][1]) < quality_min_topword_weight):
                continue

            idxs = np.where(labels == c)[0]
            doc_ids_c = residual_f.iloc[idxs]["doc_id"].astype(str).tolist()
            eff_purity, cov, top_lbl = _cluster_cso_purity(doc_ids_c)
            if augment_cso_purity_min > 0.0:
                if (not np.isfinite(eff_purity)) or (float(eff_purity) < float(augment_cso_purity_min)):
                    continue

            keep_clusters.append(c)
            cluster_qc[int(c)] = {
                "topword_weight": float(words[0][1]) if (words and words[0][1] is not None) else float("nan"),
                "cso_purity_eff": float(eff_purity) if np.isfinite(eff_purity) else float("nan"),
                "cso_label_coverage": float(cov),
                "cso_top1_label": str(top_lbl) if top_lbl else "",
            }

        if keep_clusters:
            _GENERIC_LABEL_WORDS = {
                "analysis","approach","based","model","models","method","methods","methodology",
                "system","systems","framework","architecture","architectures","design",
                "application","applications","implementation","performance","evaluation","results",
                "optimization","control","management","detection","classification","prediction",
                "algorithm","algorithms","data","learning","network","networks","study","research"
            }

            def _label_from_words(words, max_terms: int = 3, min_rel_w: float = 0.35) -> str:
                if not words:
                    return ""
                top_w = float(words[0][1]) if (words and words[0][1] is not None) else 0.0
                picked = []
                for w, wt in words:
                    if not w:
                        continue
                    wl = str(w).strip().lower()
                    if not wl or len(wl) < 3:
                        continue
                    if wl in _GENERIC_LABEL_WORDS:
                        continue
                    if picked and wt is not None and top_w > 0 and float(wt) < (min_rel_w * top_w):
                        continue
                    picked.append(str(w).strip())
                    if len(picked) >= max_terms:
                        break

                if not picked:
                    picked = [str(w).strip() for w, _ in (words[:max_terms] if words else []) if w]

                return " / ".join(picked[:max_terms])

            new_cluster_rows = []
            cid_to_tid = {}
            for cid in keep_clusters:
                cid_to_tid[cid] = next_tid
                words = kw.get(cid, [])
                label = _label_from_words(words) or f"Cluster {cid}"
                keywords = "; ".join([w for w, _ in words[:10]])

                new_cluster_rows.append({
                    "topic": next_tid,
                    "label": label,
                    "keywords": keywords,
                    "size": np.sum(labels == cid),
                    "source": "augmentation",
                    "cso_purity_eff": cluster_qc.get(int(cid), {}).get("cso_purity_eff", float("nan")),
                    "cso_label_coverage": cluster_qc.get(int(cid), {}).get("cso_label_coverage", float("nan")),
                    "cso_top1_label": cluster_qc.get(int(cid), {}).get("cso_top1_label", ""),
                })
                next_tid += 1

            for cid, tid in cid_to_tid.items():
                idxs = np.where(labels == cid)[0]
                if idxs.size == 0:
                    continue
                sub = residual_f.iloc[idxs].copy()
                doc_ids = sub["doc_id"].astype(str).tolist()
                emb_idxs = sub["emb_idx"].values.astype(int)

                cent = embeddings[emb_idxs].mean(axis=0)
                cent_norm = float(np.linalg.norm(cent) + 1e-12)
                vecs = embeddings[emb_idxs]
                vnorm = np.linalg.norm(vecs, axis=1) + 1e-12
                sims = (vecs @ cent) / (vnorm * cent_norm)
                sims = np.clip(sims, -1.0, 1.0)

                sim_map = {d: float(s) for d, s in zip(doc_ids, sims)}

                mask = doc_topics["doc_id"].isin(doc_ids) & (doc_topics["topic"] == -1)
                idx_dt = doc_topics.index[mask]
                doc_topics.loc[idx_dt, "topic"] = tid
                doc_topics.loc[idx_dt, "assignment_source"] = "augmentation"
                doc_topics.loc[idx_dt, "similarity_score"] = doc_topics.loc[idx_dt, "doc_id"].map(sim_map).astype(float)
                doc_topics.loc[idx_dt, "assignment_confidence"] = doc_topics.loc[idx_dt, "similarity_score"]

            aug_topics = pd.DataFrame(new_cluster_rows)
            if not symbolic_topics.empty:
                symbolic_topics = pd.concat([symbolic_topics, aug_topics], ignore_index=True)
            else:
                symbolic_topics = aug_topics

    doc_topics.to_csv(out_doc_topics_csv, index=False)
    symbolic_topics.to_csv(out_new_topics_csv, index=False)
    return doc_topics, symbolic_topics




def stage07_split_topics(
    doc_topics: pd.DataFrame,
    df_text: pd.DataFrame,
    embeddings: np.ndarray,
    doc_id_to_emb_idx: Dict[str, int],
    out_doc_topics_csv: Path,
    out_split_topics_csv: Path,
    enable: bool = True,
    min_topic_size: int = 200,
    trigger_mean_sim: float = 0.60,
    trigger_std_sim: float = 0.12,
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    umap_min_dist: float = 0.0,
    hdb_min_cluster_size: int = 6,
    hdb_min_samples: int = 2,
    cluster_selection_method: str = "leaf",
    quality_min_topword_weight: float = 0.0035,
    duplicate_sim_threshold: float = 0.90,
    cso_purity_min: float = 0.0,
    extra_stopwords: str = "",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import hdbscan

    dt = doc_topics.copy()
    df_idx = df_text.set_index("doc_id", drop=False)

    for col in ["assignment_source", "similarity_score", "assignment_confidence"]:
        if col not in dt.columns:
            dt[col] = ""

    if not enable:
        empty = pd.DataFrame(columns=["topic", "label", "keywords", "size", "source"])
        dt.to_csv(out_doc_topics_csv, index=False)
        empty.to_csv(out_split_topics_csv, index=False)
        return dt, empty

    assigned = dt[dt["topic"] != -1].copy()
    assigned["emb_idx"] = assigned["doc_id"].map(doc_id_to_emb_idx).fillna(-1).astype(int)
    assigned = assigned[assigned["emb_idx"] != -1].copy()

    if assigned.empty:
        empty = pd.DataFrame(columns=["topic", "label", "keywords", "size", "source"])
        dt.to_csv(out_doc_topics_csv, index=False)
        empty.to_csv(out_split_topics_csv, index=False)
        return dt, empty

    max_tid = int(dt.loc[dt["topic"] != -1, "topic"].max()) if (dt["topic"] != -1).any() else -1
    next_tid = max_tid + 1

    E = embeddings
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    def _norm(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v) + 1e-12)
        return v / n

    topic_centroids: Dict[int, np.ndarray] = {}
    for t, grp in assigned.groupby("topic"):
        idxs = grp["emb_idx"].values.astype(int)
        if idxs.size >= 2:
            c = _norm(En[idxs].mean(axis=0))
            topic_centroids[int(t)] = c

    _GENERIC_LABEL_WORDS = {
        "analysis","approach","based","model","models","method","methods","methodology",
        "system","systems","framework","architecture","architectures","design",
        "application","applications","implementation","performance","evaluation","results",
        "optimization","control","management","detection","classification","prediction",
        "algorithm","algorithms","data","learning","network","networks","study","research"
    }

    def _label_from_kws(kws, max_terms: int = 3, min_rel_w: float = 0.35) -> str:
        if not kws:
            return ""
        top_w = float(kws[0][1]) if (kws and kws[0][1] is not None) else 0.0
        picked = []
        for w, wt in kws:
            if not w:
                continue
            wl = str(w).strip().lower()
            if not wl or len(wl) < 3:
                continue
            if wl in _GENERIC_LABEL_WORDS:
                continue
            if picked and wt is not None and top_w > 0 and float(wt) < (min_rel_w * top_w):
                continue
            picked.append(str(w).strip())
            if len(picked) >= max_terms:
                break
        if not picked:
            picked = [str(w).strip() for w, _ in (kws[:max_terms] if kws else []) if w]
        return " / ".join(picked[:max_terms])

    def _cluster_cso_purity(doc_ids: List[str]) -> Tuple[float, float, str]:
        if "cso_top1_label" not in df_idx.columns:
            return (float("nan"), 0.0, "")
        vals: List[str] = []
        for d in doc_ids:
            try:
                v = df_idx.loc[d, "cso_top1_label"]
            except Exception:
                continue
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            s = str(v).strip()
            if s:
                vals.append(s)
        if not vals:
            return (0.0, 0.0, "")
        from collections import Counter
        cnt = Counter(vals)
        top_lbl, top_n = cnt.most_common(1)[0]
        cov = float(len(vals)) / float(max(1, len(doc_ids)))
        purity = float(top_n) / float(len(vals))
        eff = purity * min(1.0, cov / 0.50)
        return (float(eff), float(cov), str(top_lbl))

    split_rows: List[Dict[str, Any]] = []
    topics_to_consider = assigned["topic"].value_counts()
    topics_to_consider = topics_to_consider[topics_to_consider >= int(min_topic_size)].index.tolist()

    if not topics_to_consider:
        empty = pd.DataFrame(columns=["topic", "label", "keywords", "size", "source"])
        dt.to_csv(out_doc_topics_csv, index=False)
        empty.to_csv(out_split_topics_csv, index=False)
        return dt, empty

    for t in tqdm(topics_to_consider, desc="Stage 7: Topic splitting", unit="topic"):
        grp = assigned[assigned["topic"] == t]
        idxs = grp["emb_idx"].values.astype(int)
        if idxs.size < int(min_topic_size):
            continue

        centroid = topic_centroids.get(int(t))
        if centroid is None:
            centroid = _norm(En[idxs].mean(axis=0))
            topic_centroids[int(t)] = centroid

        sims = En[idxs] @ centroid
        mean_sim = float(np.mean(sims))
        std_sim = float(np.std(sims))

        if (mean_sim >= float(trigger_mean_sim)) and (std_sim <= float(trigger_std_sim)):
            continue

        X = E[idxs]

        try:
            import umap
            reducer = umap.UMAP(
                n_neighbors=int(umap_n_neighbors),
                n_components=int(umap_n_components),
                min_dist=float(umap_min_dist),
                metric="cosine",
                random_state=int(seed),
            )
            X_red = reducer.fit_transform(X)
        except Exception:
            from sklearn.decomposition import PCA
            X_red = PCA(n_components=min(50, X.shape[1])).fit_transform(X)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(hdb_min_cluster_size),
            min_samples=int(hdb_min_samples),
            metric="euclidean",
            cluster_selection_method=str(cluster_selection_method),
        )
        labels = clusterer.fit_predict(X_red)

        uniq = [c for c in np.unique(labels) if c != -1]
        if len(uniq) < 2:
            continue

        doc_ids_in_topic = grp["doc_id"].astype(str).tolist()
        texts = []
        for d in doc_ids_in_topic:
            try:
                texts.append(str(df_idx.loc[d, "text"]))
            except Exception:
                texts.append("")
        kw = ctfidf_keywords(texts, labels, extra_stopwords=extra_stopwords)

        cand = []
        sizes = {}
        for c in uniq:
            n = int(np.sum(labels == c))
            sizes[c] = n
            if n < int(hdb_min_cluster_size):
                continue
            words = kw.get(c, [])
            if (not words) or (words[0][1] is None) or (float(words[0][1]) < float(quality_min_topword_weight)):
                continue
            doc_ids_c = [doc_ids_in_topic[i] for i in np.where(labels == c)[0]]
            eff_purity, cov, top_lbl = _cluster_cso_purity(doc_ids_c)
            if float(cso_purity_min) > 0.0:
                if (not np.isfinite(eff_purity)) or (float(eff_purity) < float(cso_purity_min)):
                    continue
            cand.append(c)

        if len(cand) < 2:
            continue

        main_c = max(cand, key=lambda c: sizes.get(c, 0))

        for c in cand:
            if c == main_c:
                continue

            sub_pos = np.where(labels == c)[0]
            if sub_pos.size == 0:
                continue

            sub_doc_ids = [doc_ids_in_topic[i] for i in sub_pos]
            sub_emb_idxs = idxs[sub_pos]

            sub_cent = _norm(En[sub_emb_idxs].mean(axis=0))
            best_sim = -1.0
            best_tid = None
            for tid0, cent0 in topic_centroids.items():
                if int(tid0) == int(t):
                    continue
                s = float(sub_cent @ cent0)
                if s > best_sim:
                    best_sim = s
                    best_tid = tid0
            if (best_tid is not None) and (best_sim >= float(duplicate_sim_threshold)):
                continue

            new_tid = int(next_tid)
            next_tid += 1

            vecs = E[sub_emb_idxs]
            cent_raw = E[sub_emb_idxs].mean(axis=0)
            cent_norm = float(np.linalg.norm(cent_raw) + 1e-12)
            vnorm = np.linalg.norm(vecs, axis=1) + 1e-12
            sims2 = (vecs @ cent_raw) / (vnorm * cent_norm)
            sims2 = np.clip(sims2, -1.0, 1.0)
            sim_map = {d: float(s) for d, s in zip(sub_doc_ids, sims2)}

            mask_dt = dt["doc_id"].isin(sub_doc_ids) & (dt["topic"] == int(t))
            idx_dt = dt.index[mask_dt]
            dt.loc[idx_dt, "topic"] = new_tid
            dt.loc[idx_dt, "assignment_source"] = "split"
            dt.loc[idx_dt, "similarity_score"] = dt.loc[idx_dt, "doc_id"].map(sim_map).astype(float)
            dt.loc[idx_dt, "assignment_confidence"] = dt.loc[idx_dt, "similarity_score"]

            topic_centroids[new_tid] = sub_cent

            words = kw.get(c, [])
            label = _label_from_kws(words) or f"Split {t}:{c}"
            kw_str = "; ".join([w for w, _ in (words[:15] if words else [])])

            eff_purity, cov, top_lbl = _cluster_cso_purity(sub_doc_ids)

            split_rows.append({
                "topic": new_tid,
                "label": label,
                "keywords": kw_str,
                "size": int(len(sub_doc_ids)),
                "source": f"split_from_{int(t)}",
                "parent_topic": int(t),
                "parent_mean_sim": mean_sim,
                "parent_std_sim": std_sim,
                "cso_purity_eff": float(eff_purity) if np.isfinite(eff_purity) else float("nan"),
                "cso_label_coverage": float(cov),
                "cso_top1_label": str(top_lbl) if top_lbl else "",
            })

    split_topics = pd.DataFrame(split_rows)
    if split_topics.empty:
        split_topics = pd.DataFrame(columns=["topic", "label", "keywords", "size", "source"])

    dt.to_csv(out_doc_topics_csv, index=False)
    split_topics.to_csv(out_split_topics_csv, index=False)
    return dt, split_topics




def stage07_merge_micro_topics(
    doc_topics: pd.DataFrame,
    embeddings: np.ndarray,
    doc_id_to_emb_idx: Dict[str, int],
    min_topic_size: int = 10,
    sim_threshold: float = 0.85,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge micro-topics (very small topics) into their nearest larger topic using embedding-centroid cosine similarity.

    This improves interpretability (reduces topic fragmentation) and tends to increase global coherence metrics
    that are sensitive to unstable keyword sets from tiny clusters.

    Returns:
      - updated doc_topics
      - merge_map dataframe with columns:
        from_topic, to_topic, sim, from_size, to_size_before, to_size_after
    """
    dt = doc_topics.copy()

    # Only assigned topics participate; keep outliers unchanged
    assigned = dt[dt["topic"] != -1][["doc_id", "topic"]].copy()
    if assigned.empty:
        return dt, pd.DataFrame(columns=["from_topic","to_topic","sim","from_size","to_size_before","to_size_after"])

    # Build per-topic embedding index lists
    # (robust to any missing doc_ids in embeddings)
    assigned["_emb_idx"] = assigned["doc_id"].astype(str).map(doc_id_to_emb_idx)
    assigned = assigned.dropna(subset=["_emb_idx"])
    if assigned.empty:
        return dt, pd.DataFrame(columns=["from_topic","to_topic","sim","from_size","to_size_before","to_size_after"])

    assigned["_emb_idx"] = assigned["_emb_idx"].astype(int)
    assigned["topic"] = assigned["topic"].astype(int)

    sizes = assigned["topic"].value_counts()
    small_topics = sizes[sizes < int(min_topic_size)].index.tolist()
    if not small_topics:
        return dt, pd.DataFrame(columns=["from_topic","to_topic","sim","from_size","to_size_before","to_size_after"])

    big_topics = sizes[sizes >= int(min_topic_size)].index.tolist()
    if not big_topics:
        # nothing to merge into
        return dt, pd.DataFrame(columns=["from_topic","to_topic","sim","from_size","to_size_before","to_size_after"])

    # Compute normalized centroids for all topics we need (small + big)
    # Use float64 for accumulation stability
    dim = int(embeddings.shape[1])
    centroids: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}

    # group indices per topic
    grp = assigned.groupby("topic")["_emb_idx"].apply(list)
    for tid, idxs in grp.items():
        idxs = [int(i) for i in idxs]
        if not idxs:
            continue
        vec = embeddings[idxs].astype(np.float64).mean(axis=0)
        nrm = np.linalg.norm(vec)
        if nrm > 0:
            vec = vec / nrm
        centroids[int(tid)] = vec.astype(np.float32)
        counts[int(tid)] = int(len(idxs))

    # Build big centroid matrix for fast similarity
    big_ids = [int(t) for t in big_topics if int(t) in centroids]
    if not big_ids:
        return dt, pd.DataFrame(columns=["from_topic","to_topic","sim","from_size","to_size_before","to_size_after"])

    big_mat = np.vstack([centroids[t] for t in big_ids]).astype(np.float32)
    big_pos = {tid: i for i, tid in enumerate(big_ids)}

    merge_rows = []

    # Process smallest topics first (more stable)
    small_topics_sorted = sorted([int(t) for t in small_topics], key=lambda t: counts.get(int(t), 0))

    for from_tid in small_topics_sorted:
        if from_tid not in centroids:
            continue
        from_size = counts.get(from_tid, 0)
        if from_size <= 0:
            continue

        # If it grew (due to earlier merges) above threshold, skip
        if from_size >= int(min_topic_size):
            continue

        # Compute cosine similarity to big topics
        v = centroids[from_tid]
        sims = big_mat @ v  # (B,)
        j = int(np.argmax(sims))
        best_sim = float(sims[j])
        to_tid = int(big_ids[j])

        if to_tid == from_tid:
            continue

        if best_sim < float(sim_threshold):
            continue

        # Apply merge in dt
        mask = (dt["topic"].astype(int) == int(from_tid))
        if not mask.any():
            continue

        to_size_before = counts.get(to_tid, 0)
        # update doc assignments
        dt.loc[mask, "topic"] = int(to_tid)
        dt.loc[mask, "assignment_source"] = "micro_merge"
        # Keep confidence conservative: don't inflate above 1.0
        if "assignment_confidence" in dt.columns:
            prev = dt.loc[mask, "assignment_confidence"].fillna(0.0).astype(float).values
            dt.loc[mask, "assignment_confidence"] = np.clip(np.maximum(prev, best_sim), 0.0, 1.0)
        if "similarity_score" in dt.columns:
            dt.loc[mask, "similarity_score"] = float(best_sim)
        if "probability" in dt.columns:
            dt.loc[mask, "probability"] = float(best_sim)
        if "membership_score" in dt.columns:
            dt.loc[mask, "membership_score"] = float(best_sim)

        # Update centroid and counts for target topic (weighted centroid)
        to_size_after = to_size_before + from_size
        if to_tid in centroids and to_size_before > 0:
            new_vec = (centroids[to_tid].astype(np.float64) * to_size_before) + (centroids[from_tid].astype(np.float64) * from_size)
            nrm = np.linalg.norm(new_vec)
            if nrm > 0:
                new_vec = new_vec / nrm
            centroids[to_tid] = new_vec.astype(np.float32)
            counts[to_tid] = int(to_size_after)
            # update big_mat row in place
            if to_tid in big_pos:
                big_mat[big_pos[to_tid], :] = centroids[to_tid]
        else:
            counts[to_tid] = int(to_size_after)

        # Mark from topic as "consumed"
        counts[from_tid] = 0

        merge_rows.append({
            "from_topic": int(from_tid),
            "to_topic": int(to_tid),
            "sim": float(best_sim),
            "from_size": int(from_size),
            "to_size_before": int(to_size_before),
            "to_size_after": int(to_size_after),
        })

    merge_df = pd.DataFrame(merge_rows)
    return dt, merge_df

def stage07_finalize(
    doc_topics: pd.DataFrame,
    base_topics: pd.DataFrame,
    new_topics: pd.DataFrame,
    out_doc_topics_csv: Path,
    out_topics_csv: Path,
    df_text: pd.DataFrame,
    ctfidf_topn: int,
    ctfidf_weighting: str,
    extra_stopwords: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    dt = doc_topics.copy()

    all_topics = base_topics.copy()
    if not new_topics.empty:
        new_topics = new_topics.reindex(columns=all_topics.columns, fill_value="")
        all_topics = pd.concat([all_topics, new_topics], ignore_index=True)

    print("   (Finalize) Recomputing global c-TF-IDF keywords...")
    merged = dt[dt["topic"] != -1].merge(df_text[["doc_id", "text"]], on="doc_id")

    weights = None
    if ctfidf_weighting == "confidence":
        weights = merged["assignment_confidence"].fillna(0.0).values

    final_kws = ctfidf_keywords(
        merged["text"].tolist(),
        merged["topic"].values,
        topn=ctfidf_topn,
        weights=weights,
        extra_stopwords=extra_stopwords
    )

    rows = []
    sizes = dt[dt["topic"] != -1]["topic"].value_counts()

    old_label_map = dict(zip(all_topics["topic"], all_topics["label"]))

    _GENERIC_LABEL_WORDS = {
        "analysis","approach","based","model","models","method","methods","methodology",
        "system","systems","framework","architecture","architectures","design",
        "application","applications","implementation","performance","evaluation","results",
        "optimization","control","management","detection","classification","prediction",
        "algorithm","algorithms","data","learning","network","networks","study","research"
    }
    _WEAK_LABELS = {"misc","other","others","unknown","general","topic"}

    def _is_weak_label(lbl: str) -> bool:
        if not lbl:
            return True
        norm = re.sub(r"[^a-z0-9\s/_\-]+", " ", str(lbl).lower()).strip()
        if not norm:
            return True
        toks = [t for t in re.split(r"[\s/_\-]+", norm) if t]
        if not toks:
            return True
        if norm in _WEAK_LABELS:
            return True
        if len(toks) == 1 and (toks[0] in _GENERIC_LABEL_WORDS or len(toks[0]) < 4):
            return True
        if len(toks) <= 2 and all(t in _GENERIC_LABEL_WORDS for t in toks):
            return True
        return False

    def _label_from_kws(kws, max_terms: int = 3, min_rel_w: float = 0.35) -> str:
        if not kws:
            return ""
        top_w = float(kws[0][1]) if (kws and kws[0][1] is not None) else 0.0
        picked = []
        for w, wt in kws:
            if not w:
                continue
            wl = str(w).strip().lower()
            if not wl or len(wl) < 3:
                continue
            if wl in _GENERIC_LABEL_WORDS:
                continue
            if picked and wt is not None and top_w > 0 and float(wt) < (min_rel_w * top_w):
                continue
            picked.append(str(w).strip())
            if len(picked) >= max_terms:
                break
        if not picked:
            picked = [str(w).strip() for w, _ in (kws[:max_terms] if kws else []) if w]
        return " / ".join(picked[:max_terms])

    unique_topics = sorted(sizes.index.tolist())
    for tid in unique_topics:
        kws = final_kws.get(tid, [])
        kw_str = "; ".join([w for w, _ in kws])

        lbl = old_label_map.get(tid, "")
        if (not lbl) or lbl.startswith("Topic ") or lbl.startswith("AugTopic") or _is_weak_label(lbl):
            lbl = _label_from_kws(kws) or (kws[0][0] if kws else f"Topic {tid}")

        rows.append({
            "topic": tid,
            "label": lbl,
            "keywords": kw_str,
            "size": sizes[tid],
            "source": "mixed"
        })

    final_topics_df = pd.DataFrame(rows)
    final_topics_df.to_csv(out_topics_csv, index=False)
    dt.to_csv(out_doc_topics_csv, index=False)

    
    return dt, final_topics_df


# -----------------------------
# Keyword refinement (embedded from refine_final_keywords.py)
# -----------------------------
def refine_final_keywords_in_pipeline(
    df_text: pd.DataFrame,
    doc_topics: pd.DataFrame,
    topics_df: pd.DataFrame,
    out_topics_csv: Path,
    *,
    extra_stopwords: str = "",
    core_k: int = 12,
    display_k: int = 12,
    candidate_pool: int = 60,
    min_df_global: int = 5,
    weighting: str = "none",
) -> pd.DataFrame:
    """Refine final topic keywords/labels to improve interpretability and coherence proxies.

    Outputs:
      - Writes `out_topics_csv` (refined).
      - Returns refined topics DataFrame.

    Notes:
      - Produces both coherence-friendly unigram *core* keywords and a richer *display* list.
      - Keeps your existing topic ids/sizes intact; only keywords and (weak) labels are adjusted.
    """
    from collections import Counter, defaultdict
    from itertools import combinations

    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    except Exception:
        ENGLISH_STOP_WORDS = set()
        CountVectorizer = None  # type: ignore

    extras = [w.strip().lower() for w in str(extra_stopwords or "").split(",") if w.strip()]
    stop = set(getattr(ENGLISH_STOP_WORDS, "union", lambda x: set(ENGLISH_STOP_WORDS) | set(x))(set(extras))) if ENGLISH_STOP_WORDS else set(extras)

    GENERIC = {
        "analysis","approach","based","model","models","method","methods","methodology",
        "system","systems","framework","architecture","architectures","design",
        "application","applications","implementation","performance","evaluation","results",
        "optimization","control","management","detection","classification","prediction",
        "algorithm","algorithms","data","learning","network","networks","study","research",
        "paper","thesis","dissertation","propose","proposes","proposed","present","presents","presented",
        "using","use","used","novel","new","existing","problem","problems","challenge","challenges",
    }
    WEAK_LABELS = {"misc","other","others","unknown","general","topic"}

    def _is_weak_label(lbl: str) -> bool:
        if not lbl:
            return True
        norm = re.sub(r"[^a-z0-9\s/_\-]+", " ", str(lbl).lower()).strip()
        if not norm:
            return True
        toks = [t for t in re.split(r"[\s/_\-]+", norm) if t]
        if not toks:
            return True
        if norm in WEAK_LABELS:
            return True
        if len(toks) == 1 and (toks[0] in GENERIC or len(toks[0]) < 4):
            return True
        if len(toks) <= 2 and all(t in GENERIC for t in toks):
            return True
        return False

    token_re = re.compile(r"[a-z][a-z0-9_\-]{2,}")
    def tokenize_unigrams(s: str) -> Set[str]:
        s = str(s or "").lower()
        toks = set(token_re.findall(s))
        if stop:
            toks = {t for t in toks if t not in stop}
        toks = {t for t in toks if not t.isdigit() and len(t) >= 3}
        return toks

    if "doc_id" not in df_text.columns or "text" not in df_text.columns:
        raise ValueError("df_text must contain columns: doc_id, text")

    dt = doc_topics.copy()
    merged = dt[dt["topic"] != -1].merge(df_text[["doc_id", "text"]], on="doc_id", how="inner").reset_index(drop=True)
    if merged.empty:
        topics_df.to_csv(out_topics_csv, index=False)
        return topics_df

    weights = None
    if weighting == "confidence" and "assignment_confidence" in merged.columns:
        w = pd.to_numeric(merged["assignment_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0).values
        weights = w.astype(float)

    doc_tokens: List[Set[str]] = []
    df_global: Counter = Counter()
    for txt in merged["text"].tolist():
        toks = tokenize_unigrams(txt)
        doc_tokens.append(toks)
        for t in toks:
            df_global[t] += 1
    N_docs = max(1, len(doc_tokens))

    topic_to_idxs: Dict[int, List[int]] = {}
    for i, tid in enumerate(merged["topic"].astype(int).tolist()):
        topic_to_idxs.setdefault(int(tid), []).append(i)

    def ctfidf_unigram_candidates(texts: List[str], labels: np.ndarray, topn: int) -> Dict[int, List[str]]:
        if CountVectorizer is None:
            return {}
        vec = CountVectorizer(ngram_range=(1, 1), stop_words=list(stop) if stop else None, min_df=2)
        try:
            X = vec.fit_transform([str(t) for t in texts])
        except Exception:
            return {}
        terms = np.array(vec.get_feature_names_out())
        labels_i = np.asarray(labels, dtype=int)
        clusters = sorted(list(set(labels_i) - {-1}))
        if not clusters:
            return {}
        rows = []
        for c in clusters:
            idx = np.where(labels_i == c)[0]
            if len(idx) == 0:
                rows.append(np.zeros(len(terms)))
            else:
                tf = np.asarray(X[idx].sum(axis=0)).reshape(-1)
                denom = float(len(idx))
                if denom > 0:
                    tf = tf / denom
                rows.append(tf)
        C = np.vstack(rows)
        df = np.asarray((X > 0).sum(axis=0)).reshape(-1)
        idf = np.log((1 + X.shape[0]) / (1 + df)) + 1.0
        ctfidf = C * idf
        out: Dict[int, List[str]] = {}
        for row_i, c in enumerate(clusters):
            scores = ctfidf[row_i]
            top_idx = np.argsort(scores)[::-1][:max(1, int(topn))]
            cand = []
            for j in top_idx:
                w = str(terms[j]).strip().lower()
                if (not w) or (w in stop) or (w in GENERIC) or (len(w) < 3) or w.isdigit():
                    continue
                cand.append(w)
            out[int(c)] = cand
        return out

    cand_map = ctfidf_unigram_candidates(merged["text"].tolist(), merged["topic"].astype(int).values, topn=max(30, candidate_pool))

    def rescue_tokens_from_keywords(kw_str: str) -> List[str]:
        if not kw_str:
            return []
        toks = []
        for part in [p.strip() for p in str(kw_str).split(";") if p.strip()]:
            for w in re.split(r"[\s/_\-]+", part.lower().strip()):
                w = re.sub(r"[^a-z0-9_\-]+", "", w)
                if w and (w not in stop) and (w not in GENERIC) and (len(w) >= 3) and (not w.isdigit()):
                    toks.append(w)
        return toks

    topics_ref = topics_df.copy()
    kw_by_topic_existing: Dict[int, str] = {}
    if "keywords" in topics_ref.columns:
        for rr in topics_ref[["topic", "keywords"]].itertuples(index=False):
            try:
                kw_by_topic_existing[int(rr.topic)] = str(rr.keywords or "")
            except Exception:
                pass

    def npmi(pi: float, pj: float, pij: float) -> float:
        if pij <= 0.0 or pi <= 0.0 or pj <= 0.0:
            return -1.0
        pmi = math.log(pij / (pi * pj) + 1e-12)
        denom = -math.log(pij + 1e-12)
        if denom <= 0:
            return -1.0
        return pmi / denom

    rows_out = []
    for rr in topics_ref.itertuples(index=False):
        tid = int(getattr(rr, "topic"))
        if tid == -1:
            continue
        idxs = topic_to_idxs.get(tid, [])
        if not idxs:
            continue
        n_t = max(1, len(idxs))

        cand = []
        cand.extend(cand_map.get(tid, [])[:candidate_pool])
        cand.extend(rescue_tokens_from_keywords(kw_by_topic_existing.get(tid, ""))[:candidate_pool])

        seen = set()
        cand2 = []
        for w in cand:
            w = str(w).strip().lower()
            if (not w) or (w in seen) or (w in stop) or (w in GENERIC):
                continue
            if df_global.get(w, 0) < int(min_df_global):
                continue
            seen.add(w)
            cand2.append(w)
        if not cand2:
            continue

        df_t: Counter = Counter()
        pair_t: Dict[Tuple[str, str], float] = defaultdict(float)

        for local_i in idxs:
            toks = doc_tokens[local_i]
            if not toks:
                continue
            inter = [t for t in toks if t in seen]
            if not inter:
                continue
            wgt = 1.0
            if weights is not None:
                wgt = float(weights[local_i])
            for t in set(inter):
                df_t[t] += wgt
            inter_sorted = sorted(set(inter))
            for a, b in combinations(inter_sorted, 2):
                pair_t[(a, b)] += wgt

        # Start with max DF (topic) then greedily maximize avg NPMI with selected
        remaining = sorted(cand2, key=lambda w: (df_t.get(w, 0.0), df_global.get(w, 0)), reverse=True)
        selected: List[str] = [remaining.pop(0)]

        while remaining and len(selected) < int(core_k):
            best_w, best_score = None, -1e18
            for w in remaining[:max(80, len(remaining))]:
                ssum, cnt = 0.0, 0
                for s in selected:
                    a, b = (w, s) if w < s else (s, w)
                    c = float(pair_t.get((a, b), 0.0))
                    if c <= 0.0:
                        continue
                    pij = c / float(max(1.0, n_t))
                    pi = float(df_t.get(w, 0.0)) / float(max(1.0, n_t))
                    pj = float(df_t.get(s, 0.0)) / float(max(1.0, n_t))
                    ssum += npmi(pi, pj, pij)
                    cnt += 1
                score = (ssum / max(1, cnt)) + 0.03 * math.log(1.0 + float(df_t.get(w, 0.0)))
                if score > best_score:
                    best_score = score
                    best_w = w
            if best_w is None:
                break
            selected.append(best_w)
            remaining = [w for w in remaining if w != best_w]

        # Display keywords: keep informative phrases that contain selected tokens, then pad with selected
        display_terms: List[str] = []
        orig_parts = [p.strip() for p in kw_by_topic_existing.get(tid, "").split(";") if p.strip()]
        for p in orig_parts:
            pl = p.lower()
            if any(tok in pl for tok in selected[:max(3, int(core_k)//2)]):
                toks = [t for t in re.split(r"[\s/_\-]+", pl) if t]
                if toks and all(t in GENERIC for t in toks):
                    continue
                display_terms.append(p.strip())
            if len(display_terms) >= int(display_k):
                break
        for w in selected:
            if len(display_terms) >= int(display_k):
                break
            display_terms.append(w)

        keywords_core = "; ".join(selected[:int(core_k)])
        keywords_display = "; ".join(display_terms[:int(display_k)])

        lbl = str(getattr(rr, "label", "") or "")
        if _is_weak_label(lbl) or lbl.startswith("Topic ") or lbl.startswith("AugTopic") or lbl.startswith("Split "):
            picked = []
            for p in display_terms:
                p2 = re.sub(r"\s+", " ", str(p).strip())
                if not p2:
                    continue
                low = p2.lower()
                toks = [t for t in re.split(r"[\s/_\-]+", low) if t]
                if toks and all(t in GENERIC for t in toks):
                    continue
                picked.append(p2)
                if len(picked) >= 3:
                    break
            if picked:
                lbl = " / ".join(picked[:3])

        row = dict(zip(topics_ref.columns, list(rr)))
        row["keywords_raw"] = kw_by_topic_existing.get(tid, row.get("keywords", ""))
        row["keywords_core"] = keywords_core
        row["keywords_display"] = keywords_display
        row["label"] = lbl
        rows_out.append(row)

    refined = pd.DataFrame(rows_out)
    for col in ["keywords_raw", "keywords_core", "keywords_display"]:
        if col not in refined.columns:
            refined[col] = ""
    if "keywords" in refined.columns:
        # Keep `keywords` coherence-friendly (unigram core). Use `keywords_display` for human-facing cards/reports.
        refined["keywords"] = refined["keywords_core"].fillna(refined["keywords"]).astype(str)

    refined.to_csv(out_topics_csv, index=False)
    return refined


# -----------------------------
# Stages 08-11: Trends, Shift, CSO Trends, Baselines
# -----------------------------


def stage08_trends_topics(
    doc_topics: pd.DataFrame,
    topics_df: pd.DataFrame,
    out_csv: Path,
    out_emerging_csv: Path,
    alpha: float,
    emerging_if_burst: bool,
    min_year_docs: int,
    mk_method: str = "classic",
    mk_lag_max: int = 8,
    prevalence_mode: str = "relative",
    min_topic_size: int = 25,
) -> pd.DataFrame:
    """Compute trend statistics for each topic.

    Notes
    -----
    - Writes BOTH absolute yearly counts and relative prevalence (counts / total docs per year)
      into the output CSV.
    - The MK/Sen summary columns (`mk_*`, `sen_slope`) are computed on the series selected by
      `prevalence_mode` ("relative" by default). The non-selected series is still reported as
      `*_rel` / `*_abs` columns for transparency.
    - `min_year_docs` applies to the TOTAL number of documents in that year (denominator robustness).
    - Years are counted over the full corpus (including outliers) so relative prevalence is not
      distorted by abstention rates.
    """
    dt_all = doc_topics.copy()
    if dt_all.empty:
        pd.DataFrame().to_csv(out_csv, index=False)
        pd.DataFrame().to_csv(out_emerging_csv, index=False)
        return pd.DataFrame()

    dt_all["year"] = dt_all["year"].astype(int)

    # Assigned docs for per-topic counts
    dt_assigned = dt_all[dt_all["topic"] != -1].copy()
    if dt_assigned.empty:
        pd.DataFrame().to_csv(out_csv, index=False)
        pd.DataFrame().to_csv(out_emerging_csv, index=False)
        return pd.DataFrame()

    years_full = list(range(int(dt_all["year"].min()), int(dt_all["year"].max()) + 1))
    year_totals = dt_all.groupby("year").size().reindex(years_full).fillna(0).astype(int)

    # Robustness: ignore years where the corpus is too small
    years_use = [y for y in years_full if int(year_totals.get(y, 0)) >= int(min_year_docs)]
    if len(years_use) < 3:
        # Not enough points for MK
        pd.DataFrame().to_csv(out_csv, index=False)
        pd.DataFrame().to_csv(out_emerging_csv, index=False)
        return pd.DataFrame()

    p_rows: List[Dict[str, Any]] = []
    for topic_id, grp in dt_assigned.groupby("topic"):
        topic_id_int = int(topic_id)

        # Total docs in this topic across all years
        topic_total = int(len(grp))
        if topic_total < int(min_topic_size):
            continue

        # Absolute counts per year (assigned docs)
        counts = grp["year"].value_counts().to_dict()
        counts_full = np.array([int(counts.get(y, 0)) for y in years_full], dtype=float)
        counts_use = np.array([int(counts.get(y, 0)) for y in years_use], dtype=float)

        # Relative prevalence per year (counts / total docs in year)
        prev_full = np.array([
            (counts.get(y, 0) / int(year_totals.get(y, 0))) if int(year_totals.get(y, 0)) > 0 else 0.0
            for y in years_full
        ], dtype=float)
        prev_use = np.array([
            (counts.get(y, 0) / int(year_totals.get(y, 0))) if int(year_totals.get(y, 0)) > 0 else 0.0
            for y in years_use
        ], dtype=float)

        # MK + Sen on BOTH series
        p_rel, tau_rel, trend_rel = mann_kendall_test(prev_use.tolist(), alpha=float(alpha), method=str(mk_method), lag_max=int(mk_lag_max))
        p_abs, tau_abs, trend_abs = mann_kendall_test(counts_use.tolist(), alpha=float(alpha), method=str(mk_method), lag_max=int(mk_lag_max))

        slope_rel = float(theil_sen_slope(prev_use.tolist(), years_use))
        slope_abs = float(theil_sen_slope(counts_use.tolist(), years_use))

        # Choose which series drives the summary columns
        mode = (prevalence_mode or "relative").strip().lower()
        if mode not in ("relative", "absolute"):
            mode = "relative"

        if mode == "absolute":
            mk_p, mk_tau, mk_trend, sen_slope = p_abs, tau_abs, trend_abs, slope_abs
        else:
            mk_p, mk_tau, mk_trend, sen_slope = p_rel, tau_rel, trend_rel, slope_rel

        # Burst (optional)
        burst_max = float("nan")
        try:
            burst_df = kleinberg_bursts(counts_use.astype(int).tolist(), s=2, gamma=1.0)
            if not burst_df.empty and "burst_state" in burst_df.columns:
                burst_max = float(burst_df["burst_state"].max())
        except Exception:
            burst_max = float("nan")

        # Merge label
        label = ""
        if "topic" in topics_df.columns:
            hit = topics_df[topics_df["topic"].astype(int) == topic_id_int]
            if not hit.empty and "name" in hit.columns:
                label = str(hit.iloc[0]["name"])

        p_rows.append({
            "topic": topic_id_int,
            "topic_label": label,
            "topic_total_docs": topic_total,
            "years_used_n": int(len(years_use)),
            "years_used_min": int(min(years_use)),
            "years_used_max": int(max(years_use)),
            "mk_method": str(mk_method),
            "prevalence_mode_used": mode,
            # Summary columns (series selected by prevalence_mode)
            "mk_p": float(mk_p),
            "mk_tau": float(mk_tau),
            "mk_trend": str(mk_trend),
            "sen_slope": float(sen_slope),
            # Relative series outputs
            "mk_p_rel": float(p_rel),
            "mk_tau_rel": float(tau_rel),
            "mk_trend_rel": str(trend_rel),
            "sen_slope_rel": float(slope_rel),
            # Absolute series outputs
            "mk_p_abs": float(p_abs),
            "mk_tau_abs": float(tau_abs),
            "mk_trend_abs": str(trend_abs),
            "sen_slope_abs": float(slope_abs),
            # Burst
            "burst_max": burst_max,
            # Also store the first/last relative prevalence for quick inspection
            "prev_rel_first": float(prev_full[0]) if len(prev_full) else 0.0,
            "prev_rel_last": float(prev_full[-1]) if len(prev_full) else 0.0,
            "count_abs_first": float(counts_full[0]) if len(counts_full) else 0.0,
            "count_abs_last": float(counts_full[-1]) if len(counts_full) else 0.0,
        })

    out = pd.DataFrame(p_rows)
    if out.empty:
        out.to_csv(out_csv, index=False)
        out.to_csv(out_emerging_csv, index=False)
        return out

    # FDR correction for BOTH series
    out["mk_q_rel"] = bh_fdr(out["mk_p_rel"].to_numpy(dtype=float))
    out["mk_q_abs"] = bh_fdr(out["mk_p_abs"].to_numpy(dtype=float))

    # FDR for the selected series, aligned with `mk_p`
    if (out["prevalence_mode_used"].iloc[0] == "absolute"):
        out["mk_q"] = out["mk_q_abs"]
    else:
        out["mk_q"] = out["mk_q_rel"]

    out.to_csv(out_csv, index=False)

    # Emerging topics summary (uses selected series columns)
    emerging = out.copy()
    emerging = emerging[emerging["mk_trend"] == "increasing"].copy()
    emerging = emerging[emerging["mk_q"] <= float(alpha)].copy()

    if emerging_if_burst:
        try:
            emerging = emerging[(emerging["burst_max"].fillna(0.0) >= 2.0)].copy()
        except Exception:
            pass

    emerging = emerging.sort_values(["mk_q", "sen_slope"], ascending=[True, False])
    emerging.to_csv(out_emerging_csv, index=False)
    return out

def stage09_semantic_shift(
    doc_topics: pd.DataFrame,
    embeddings: np.ndarray,
    doc_id_to_emb_idx: Dict[str, int],
    topics_df: pd.DataFrame,
    out_csv: Path,
    min_docs_per_year: int = 5,
    require_consecutive_years: bool = False,
    normalize_by_year_gap: bool = False,
) -> None:
    """
    Calculates topic semantic shift over time using embedding centroids.

    For each topic and each year with >= min_docs_per_year documents (and available embeddings),
    compute an L2-normalized centroid. Then compute cosine distance between successive years.

    If normalize_by_year_gap=True, distances are divided by the year gap (e.g., 2005->2008
    distance is treated as a 3-year drift and normalized per-year). Raw (un-normalized) drift
    metrics are also written for transparency.
    """

    cols_out = [
        "topic", "label", "years", "first_year", "last_year",
        "drift_total_dist", "drift_step_mean", "drift_robust_score",
        "mean_year_gap", "missing_emb_rate",
        "drift_total_dist_raw", "drift_step_mean_raw", "drift_mode",
    ]

    dt = doc_topics.copy()
    dt = dt[dt["topic"] != -1]
    if dt.empty:
        pd.DataFrame(columns=cols_out).to_csv(out_csv, index=False)
        return

    # Ensure integer years
    dt["year"] = pd.to_numeric(dt["year"], errors="coerce")
    dt = dt.dropna(subset=["year"])
    dt["year"] = dt["year"].astype(int)
    dt = dt[dt["year"] > 0]

    # Map embeddings once (vectorized)
    dt["doc_id_str"] = dt["doc_id"].astype(str)
    dt["emb_idx"] = dt["doc_id_str"].map(doc_id_to_emb_idx)

    # Build a stable topic->label map once (avoids per-topic DataFrame filtering)
    topic_label_map: Dict[int, str] = {}
    if topics_df is not None and hasattr(topics_df, "empty") and (not topics_df.empty) and ("topic" in topics_df.columns):
        tdf = topics_df.copy()
        try:
            tdf["topic"] = pd.to_numeric(tdf["topic"], errors="coerce").astype("Int64")
        except Exception:
            pass

        # Try a few common label fields
        label_cols = [c for c in ("label", "name", "Name", "topic_name", "Representation") if c in tdf.columns]
        if label_cols:
            lc = label_cols[0]
            try:
                tdf2 = tdf.dropna(subset=["topic"])
                for _, r in tdf2.iterrows():
                    try:
                        topic_label_map[int(r["topic"])] = str(r.get(lc, "") or "")
                    except Exception:
                        continue
            except Exception:
                pass

    rows: List[Dict[str, Any]] = []

    # group once by topic (avoid O(topics*docs) slicing)
    for tid, sub in tqdm(dt.groupby("topic", sort=False), desc="Stage 10: Semantic Shift", unit="topic"):
        # missing embedding diagnostics
        total_docs = int(len(sub))
        sub_ok = sub.dropna(subset=["emb_idx"]).copy()
        ok_docs = int(len(sub_ok))
        if ok_docs == 0:
            continue
        missing_emb_rate = 1.0 - (ok_docs / max(total_docs, 1))

        # Ensure integer indices where possible
        try:
            sub_ok["emb_idx"] = sub_ok["emb_idx"].astype(int)
        except Exception:
            pass

        centroids: Dict[int, np.ndarray] = {}

        for year, grp in sub_ok.groupby("year"):
            if int(len(grp)) < int(min_docs_per_year):
                continue

            idxs = grp["emb_idx"].to_numpy(dtype=int, copy=False)
            if idxs.size:
                # Guard against stale/out-of-range indices
                idxs = idxs[(idxs >= 0) & (idxs < int(embeddings.shape[0]))]
            if idxs.size == 0:
                continue

            vecs = embeddings[idxs]
            if vecs.ndim != 2 or vecs.shape[0] == 0:
                continue

            cent = vecs.mean(axis=0)
            norm = float(np.linalg.norm(cent))
            if not (np.isfinite(norm) and norm > 0):
                continue
            cent = cent / (norm + 1e-12)

            centroids[int(year)] = cent

        valid_years = sorted(centroids.keys())
        if len(valid_years) < 2:
            continue

        dists = []
        dists_raw = []
        gaps = []

        for y1, y2 in zip(valid_years[:-1], valid_years[1:]):
            gap = int(y2 - y1)
            if gap <= 0:
                continue
            if require_consecutive_years and gap != 1:
                continue

            c1, c2 = centroids[y1], centroids[y2]
            sim = float(np.dot(c1, c2))
            sim = max(-1.0, min(1.0, sim))
            dist_raw = 1.0 - sim
            # Numerical safety: cosine distance should be in [0, 2]
            dist_raw = float(max(0.0, min(2.0, dist_raw)))

            dist = dist_raw / float(gap) if bool(normalize_by_year_gap) else dist_raw

            dists.append(float(dist))
            dists_raw.append(float(dist_raw))
            gaps.append(gap)

        if not dists:
            continue

        total_dist = float(np.sum(dists))
        mean_step = float(np.mean(dists))
        total_dist_raw = float(np.sum(dists_raw))
        mean_step_raw = float(np.mean(dists_raw))
        mean_gap = float(np.mean(gaps)) if gaps else 1.0

        try:
            tid_i = int(tid)
        except Exception:
            continue

        label = topic_label_map.get(tid_i, "") or ""

        # Robust score should reflect the number of drift steps actually used.
        n_steps = int(len(dists))
        years_used = n_steps + 1

        rows.append({
            "topic": tid_i,
            "label": label,
            "years": int(len(valid_years)),
            "first_year": int(valid_years[0]),
            "last_year": int(valid_years[-1]),
            "drift_total_dist": total_dist,
            "drift_step_mean": mean_step,
            "drift_robust_score": float(mean_step * np.log(years_used + 1)),
            "mean_year_gap": mean_gap,
            "missing_emb_rate": float(missing_emb_rate),
            "drift_total_dist_raw": total_dist_raw,
            "drift_step_mean_raw": mean_step_raw,
            "drift_mode": ("per_year" if bool(normalize_by_year_gap) else "raw_step"),
        })

    out = pd.DataFrame(rows, columns=cols_out)
    if not out.empty:
        out = out.sort_values("drift_robust_score", ascending=False)
    out.to_csv(out_csv, index=False)

def stage10_cso_concept_trends(
    df: pd.DataFrame,
    cso_df: pd.DataFrame,
    out_csv: Path,
    alpha: float,
    min_concept_docs: int = 50,
    min_year_docs: int = 10,
) -> None:
    df = df.copy()
    cso_df = cso_df.copy()
    df["doc_id"] = df["doc_id"].astype(str)
    cso_df["doc_id"] = cso_df["doc_id"].astype(str)

    if "union" not in cso_df.columns:
        pd.DataFrame().to_csv(out_csv, index=False)
        return

    cso_df["union"] = cso_df["union"].apply(_parse_list_maybe)

    merged = df.merge(cso_df[["doc_id", "union"]], on="doc_id", how="left")
    merged["union"] = merged["union"].apply(lambda x: x if isinstance(x, list) else [])

    years_full = sorted(merged["year"].unique())
    year_counts_full = merged.groupby("year")["doc_id"].count().reindex(years_full, fill_value=0)
    years_use = [y for y in years_full if int(year_counts_full.loc[y]) >= int(min_year_docs)]
    if len(years_use) < 3:
        years_use = years_full
    # Fast concept counting / per-year series via explode (avoids O(#concepts * #docs) .apply)
    expl = merged[["year", "union"]].explode("union")
    if expl.empty:
        pd.DataFrame().to_csv(out_csv, index=False)
        return
    expl = expl[expl["union"].notna()]
    if expl.empty:
        pd.DataFrame().to_csv(out_csv, index=False)
        return
    expl["union"] = expl["union"].astype(str).str.strip()
    expl = expl[expl["union"].str.len() > 0]
    if expl.empty:
        pd.DataFrame().to_csv(out_csv, index=False)
        return

    counts = expl["union"].value_counts()
    concepts = counts[counts >= int(min_concept_docs)].index.tolist()
    if not concepts:
        pd.DataFrame().to_csv(out_csv, index=False)
        return

    concept_year_counts = (
        expl.groupby(["union", "year"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=years_full, fill_value=0)
    )

    rows = []
    for c in tqdm(concepts, desc="Calculating CSO Trends"):
        sub = concept_year_counts.loc[c].reindex(years_full, fill_value=0)
        prev = (sub / year_counts_full.replace(0, np.nan)).fillna(0.0).reindex(years_use, fill_value=0.0)
        mk_p, mk_tau, mk_trend = mann_kendall_test(prev.tolist(), float(alpha))
        starts, burst, _ = kleinberg_bursts(sub.tolist())
        rows.append({
            "concept": c,
            "size": int(counts[c]),
            "mk_p": float(mk_p),
            "mk_tau": float(mk_tau),
            "mk_trend": mk_trend,
            "burst_max": int(burst),
            "burst_years": ";".join(map(str, [years_full[i] for i in starts if 0 <= int(i) < len(years_full)])),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["mk_q"] = bh_fdr(out["mk_p"].to_numpy())
        out.sort_values(["mk_q", "burst_max"], ascending=[True, False]).to_csv(out_csv, index=False)
    else:
        out.to_csv(out_csv, index=False)


# -----------------------------
# Substep 11a: Prevalence baseline (Robust)
# -----------------------------
def stage11a_prevalence_baseline(doc_topics: pd.DataFrame, out_csv: Path, min_topic_docs: int = 50) -> None:
    dt = doc_topics[doc_topics["topic"] != -1].copy()
    if dt.empty:
        pd.DataFrame().to_csv(out_csv, index=False)
        return

    t = dt["thesis_type"].astype(str).str.lower()
    dt["is_doctorate"] = (t.str.contains("doctor") | t.str.contains("dokto") | t.str.contains("phd")).astype(int)

    year = pd.to_numeric(dt["year"], errors="coerce")
    dt["year_z"] = (year - year.mean()) / (year.std(ddof=0) if year.std(ddof=0) > 0 else 1.0)

    topic_counts = dt["topic"].value_counts()
    topics = topic_counts[topic_counts >= int(min_topic_docs)].index.tolist()

    rows = []
    use_statsmodels = False
    try:
        import statsmodels.api as sm  # type: ignore
        from statsmodels.tools.sm_exceptions import ConvergenceWarning  # type: ignore
        import warnings
        use_statsmodels = True
    except Exception:
        use_statsmodels = False

    try:
        from scipy.stats import spearmanr  # type: ignore
    except Exception as e:
        raise RuntimeError("stage11a_prevalence_baseline requires scipy (for spearmanr) or statsmodels.") from e

    # Precompute arrays for speed (avoid repeated pandas allocations inside the topic loop)
    topic_arr = dt["topic"].to_numpy()
    year_arr = pd.to_numeric(dt["year"], errors="coerce").to_numpy()
    year_z_arr = dt["year_z"].to_numpy(dtype=float)
    is_doc_arr = dt["is_doctorate"].to_numpy(dtype=float)

    if use_statsmodels:
        n = int(len(dt))
        # columns: [const, year_z, is_doctorate]
        X_base = np.column_stack([np.ones(n, dtype=float), year_z_arr, is_doc_arr])

        iterator = tqdm(topics, desc="Prevalence Baseline")
        for tid in iterator:
            y = (topic_arr == tid).astype(np.int8)

            if int(y.sum()) < 5:
                r, p = spearmanr(year_arr, y)
                rows.append({"topic": int(tid), "method": "spearman (sparse)", "beta_year_z": float(r), "p": float(p)})
                continue

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    mdl = sm.Logit(y, X_base).fit(disp=False, maxiter=2000)

                if not mdl.mle_retvals.get("converged", True):
                    raise RuntimeError("Failed to converge")

                beta = float(mdl.params[1]) if len(mdl.params) > 1 else float("nan")
                pv = float(mdl.pvalues[1]) if len(mdl.pvalues) > 1 else float("nan")
                rows.append({"topic": int(tid), "method": "logit", "beta_year_z": beta, "p": pv})
            except Exception:
                r, p = spearmanr(year_arr, y)
                rows.append({"topic": int(tid), "method": "spearman (fallback)", "beta_year_z": float(r), "p": float(p)})
    else:
        for tid in tqdm(topics, desc="Prevalence Baseline"):
            y = (topic_arr == tid).astype(np.int8)
            r, p = spearmanr(year_arr, y)
            rows.append({"topic": int(tid), "method": "spearman", "beta_year_z": float(r), "p": float(p)})

    out = pd.DataFrame(rows)
    if not out.empty:
        if "p" in out.columns and out["p"].notna().any():
            out["q"] = bh_fdr(out["p"].fillna(1.0).to_numpy())
            out = out.sort_values("q")
        out.to_csv(out_csv, index=False)
    else:
        out.to_csv(out_csv, index=False)

def stage11b_model_stability(
    df: pd.DataFrame,
    run_dir: Path,
    embed_model: str,
    embeddings_path: Path,
    runs: int,
    batch_size: int,
    umap_n_neighbors: int,
    umap_n_components: int,
    umap_min_dist: float,
    hdb_min_cluster_size: int,
    hdb_min_samples: int,
    extra_stopwords: str,
    reduce_topics: str,
    out_csv: Path,
    vectorizer_min_df: int = 5,
    seed: int = 32,
) -> None:
    if int(runs) <= 1:
        pd.DataFrame([{"runs": int(runs), "mean_ami": 1.0, "std_ami": 0.0}]).to_csv(out_csv, index=False)
        return

    from sklearn.metrics import adjusted_mutual_info_score  # type: ignore

    assignments = []
    for i in tqdm(range(int(runs)), desc="Stability Runs"):
        out_d = run_dir / f"_stab_doc_topics_{i}.csv"
        out_t = run_dir / f"_stab_topics_{i}.csv"
        mdl = run_dir / f"_stab_model_{i}"
        stage03_bertopic(
            df=df,
            out_doc_topics_csv=out_d,
            out_topics_csv=out_t,
            model_dir=mdl,
            embeddings_path=embeddings_path,
            embed_model=embed_model,
            umap_n_neighbors=umap_n_neighbors,
            umap_n_components=umap_n_components,
            umap_min_dist=umap_min_dist,
            hdb_min_cluster_size=hdb_min_cluster_size,
            hdb_min_samples=hdb_min_samples,
            seed=int(seed) + i * 17,
            batch_size=batch_size,
            extra_stopwords=extra_stopwords,
            reduce_topics=reduce_topics,
        )
        assignments.append(pd.read_csv(out_d)["topic"].to_numpy())

    amis = [
        adjusted_mutual_info_score(assignments[i], assignments[j])
        for i in range(int(runs)) for j in range(i + 1, int(runs))
    ]
    pd.DataFrame([{"runs": int(runs), "mean_ami": float(np.mean(amis)), "std_ami": float(np.std(amis))}]).to_csv(out_csv, index=False)


# -----------------------------
# Main Pipeline Orchestrator
# -----------------------------
def pipeline_parse_args(argv: List[str]):
    ap = argparse.ArgumentParser(description="Neuro-Symbolic Topic Trends Pipeline")

    ap.add_argument("--data", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_tr_chars", type=int, default=0)

    ap.add_argument("--skip_cso", action="store_true")
    ap.add_argument("--cso_ttl", default="CSO.3.5.ttl")
    ap.add_argument("--cso_model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--cso_topk", type=int, default=10)
    ap.add_argument("--cso_sim_threshold", type=float, default=0.6)
    ap.add_argument("--cso_doc_batch_size", type=int, default=512)
    ap.add_argument("--cso_entropy_topm", type=int, default=15)
    ap.add_argument("--cso_entropy_margin", type=float, default=0.05)
    ap.add_argument("--cso_entropy_softmax_temp", type=float, default=0.07)

    ap.add_argument("--embed_model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--batch_size", type=int, default=32)

    ap.add_argument("--umap_neighbors", type=int, default=15)
    ap.add_argument("--umap_components", type=int, default=5)
    ap.add_argument("--umap_min_dist", type=float, default=0.0)
    ap.add_argument("--min_cluster_size", type=int, default=50)
    ap.add_argument("--min_samples", type=int, default=10)
    ap.add_argument("--reduce_topics", type=str, default="none")
    ap.add_argument("--extra_stopwords", default="")
    ap.add_argument("--vectorizer_min_df", type=int, default=5,
                    help="Min document frequency for terms (lower=2 to catch niche topics)")

    ap.add_argument("--max_outliers", type=int, default=0)
    ap.add_argument("--reassign_sim_threshold", type=float, default=0.75)
    ap.add_argument("--reassign_margin_threshold", type=float, default=0.0)
    ap.add_argument("--reassign_margin_auto", action="store_true")
    ap.add_argument("--reassign_margin_quantile", type=float, default=0.85)
    ap.add_argument("--entropy_threshold", type=float, default=1.0)
    ap.add_argument("--entropy_auto", action="store_true")
    ap.add_argument("--entropy_quantile", type=float, default=0.85)

    ap.add_argument("--llm_mode", choices=["none", "ollama", "transformers"], default="none")
    ap.add_argument("--llm_model", default="mistral")
    ap.add_argument("--llm_topk", type=int, default=5)
    ap.add_argument("--llm_sim_floor", type=float, default=0.55)
    ap.add_argument("--ollama_url", default="http://localhost:11434")
    ap.add_argument("--ollama_workers", type=int, default=4)
    ap.add_argument("--ollama_timeout", type=int, default=600)
    ap.add_argument("--entropy_gate", action="store_true")
    ap.add_argument("--cso_sim_threshold_gate", type=float, default=0.85)

    ap.add_argument("--augment_min_cluster_size", type=int, default=8)
    ap.add_argument("--augment_min_samples", type=int, default=3)
    ap.add_argument("--augment_quality_min_topword_weight", type=float, default=0.0035)
    ap.add_argument("--augment_cso_purity_min", type=float, default=0.70)

    # Augmentation-specific UMAP/HDBSCAN knobs (so Stage 6 can be tuned independently)
    ap.add_argument("--augment_umap_neighbors", type=int, default=15)
    ap.add_argument("--augment_umap_components", type=int, default=5)
    ap.add_argument("--augment_umap_min_dist", type=float, default=0.0)
    ap.add_argument("--augment_cluster_selection_method", choices=["eom", "leaf"], default="leaf")

    # Symbolic topic discovery (CSO-driven)
    ap.add_argument("--symbolic_min_cluster_size", type=int, default=6)
    ap.add_argument("--symbolic_top1_sim_min", type=float, default=0.65)
    ap.add_argument("--symbolic_max_topics", type=int, default=250)

    # Topic splitting pass (between Stage 6 and finalization)
    ap.add_argument("--split_topics", dest="split_topics", action="store_true", default=True)
    ap.add_argument("--no_split_topics", dest="split_topics", action="store_false")
    ap.add_argument("--split_min_topic_size", type=int, default=200)
    ap.add_argument("--split_trigger_mean_sim", type=float, default=0.60)
    ap.add_argument("--split_trigger_std_sim", type=float, default=0.12)
    ap.add_argument("--split_umap_neighbors", type=int, default=15)
    ap.add_argument("--split_umap_components", type=int, default=5)
    ap.add_argument("--split_umap_min_dist", type=float, default=0.0)
    ap.add_argument("--split_min_cluster_size", type=int, default=6)
    ap.add_argument("--split_min_samples", type=int, default=2)
    ap.add_argument("--split_cluster_selection_method", choices=["eom", "leaf"], default="leaf")
    ap.add_argument("--split_quality_min_topword_weight", type=float, default=0.0035)
    ap.add_argument("--split_duplicate_sim_threshold", type=float, default=0.90)

    # Final cleanup pass: centroid reassignment on remaining outliers (post-splitting)
    ap.add_argument("--final_cleanup", dest="final_cleanup", action="store_true", default=True)
    ap.add_argument("--no_final_cleanup", dest="final_cleanup", action="store_false")
    ap.add_argument("--split_cso_purity_min", type=float, default=0.0)


    ap.add_argument("--ctfidf_topn", type=int, default=15)
    
    # Keyword refinement (post-finalization): improves interpretability and tends to lift global NPMI by removing generic/noisy tokens.
    # Enabled by default; produces:
    #   - 09_final_topics_raw.csv   (original Stage 8 output)
    #   - 09_final_topics.csv       (refined keywords/labels)
    ap.add_argument("--kw_refine", dest="kw_refine", action="store_true", default=True)
    ap.add_argument("--no_kw_refine", dest="kw_refine", action="store_false")
    ap.add_argument("--kw_refine_core_k", type=int, default=12, help="How many core (coherence-friendly) unigrams to keep per topic")
    ap.add_argument("--kw_refine_display_k", type=int, default=12, help="How many display keywords to keep per topic (may include phrases)")
    ap.add_argument("--kw_refine_candidate_pool", type=int, default=60, help="Candidate pool size per topic for greedy coherence selection")
    ap.add_argument("--kw_refine_min_df_global", type=int, default=5, help="Minimum document frequency for a token to be eligible")
    ap.add_argument("--kw_refine_weighting", "--ctfidf_weighting", choices=["none", "confidence"], default="none",
                help="If 'confidence', downweights low-confidence documents when building topic token stats")

    # Micro-topic merge (post-splitting, pre-finalization)
    ap.add_argument("--merge_micro_topics", action="store_true", help="Merge tiny topics into nearest larger topics using embedding centroid cosine similarity")
    ap.add_argument("--no_merge_micro_topics", action="store_true", help="Disable micro-topic merge")
    ap.add_argument("--micro_merge_min_topic_size", type=int, default=10, help="Topics with size < this will be considered micro-topics for merging")
    ap.add_argument("--micro_merge_sim_threshold", type=float, default=0.85, help="Cosine similarity threshold to merge a micro-topic into a target topic")


    ap.add_argument("--trend_alpha", type=float, default=0.05)

    ap.add_argument("--mk_method", choices=["classic", "tfpw", "hamed_rao"], default="classic",
                    help="Mann-Kendall variant: classic, tfpw (Trend-Free Pre-Whitening), or hamed_rao")
    ap.add_argument("--mk_lag_max", type=int, default=10,
                    help="Max lag for autocorrelation correction in MK test")
    ap.add_argument("--prevalence_mode", choices=["relative", "absolute"], default="relative",
                    help="Calculate trends on relative share (relative) or absolute counts (absolute)")
    ap.add_argument("--trend_min_topic_size", type=int, default=15,
                    help="Minimum total topic size to calculate trends")

    ap.add_argument("--emerging_if_burst", action="store_true")
    ap.add_argument("--min_year_docs", type=int, default=10)
    ap.add_argument("--min_docs_shift", type=int, default=5, help="Min docs per year (per-topic) required for semantic shift.")
    ap.add_argument("--stability_runs", type=int, default=0)

    ap.add_argument("--ablation_out_dir", default="")
    ap.add_argument("--ablation_runs", nargs="+", default=[])
    ap.add_argument("--ablation_label", default="")
    ap.add_argument("--ablation_stage14_py", default="03_evaluate_topic_model.py")
    ap.add_argument("--ablation_py", default="04_ablation_compare_with_entropy.py")
    ap.add_argument("--ablation_copy_artifacts", action="store_true")
    ap.add_argument("--ablation_top_n", type=int, default=20)
    ap.add_argument("--ablation_silhouette_sample", type=int, default=2000)
    ap.add_argument("--ablation_silhouette_per_topic_max", type=int, default=300)
    args = ap.parse_args(argv)
    # Backward-compatible alias: some pipeline code/runner expects args.ctfidf_weighting
    if not hasattr(args, "ctfidf_weighting"):
        args.ctfidf_weighting = getattr(args, "kw_refine_weighting", "none")
    return args


# -----------------------------
# Pipeline Main (11 stages: 1 → 11)
# -----------------------------
def pipeline_main(argv: List[str]) -> int:
    args = pipeline_parse_args(argv)
    # Backward-compatible alias (runner may pass --ctfidf_weighting)
    if (not hasattr(args, 'ctfidf_weighting')) and hasattr(args, 'kw_refine_weighting'):
        setattr(args, 'ctfidf_weighting', getattr(args, 'kw_refine_weighting'))


    run_dir = Path(args.run_dir)
    ensure_dir(run_dir)
    print(f"🚀 Starting Pipeline. Output: {run_dir}")

    run_started_at_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_cmd = "python " + " ".join([Path(sys.argv[0]).name] + list(argv))
    wall_t0 = time.perf_counter()
    stage_times_sec: Dict[str, float] = {}
    _current_stage_label: Optional[str] = None
    _current_stage_t0: Optional[float] = None

    try:
        _pbar = tqdm(total=11, unit="stage", dynamic_ncols=True)
    except Exception:
        _pbar = None

    def pb_set(stage_num: int, desc: str) -> None:
        nonlocal _current_stage_label, _current_stage_t0
        now = time.perf_counter()
        if _current_stage_label is not None and _current_stage_t0 is not None:
            stage_times_sec[_current_stage_label] = stage_times_sec.get(_current_stage_label, 0.0) + (now - _current_stage_t0)
        _current_stage_label = f"{stage_num:02d}:{desc}"
        _current_stage_t0 = now
        if _pbar is not None:
            _pbar.set_description(f"Stage {stage_num}: {desc}")

    def pb_tick(**postfix) -> None:
        if _pbar is not None:
            if postfix:
                postfix = {k: v for k, v in postfix.items() if v is not None}
                if postfix:
                    _pbar.set_postfix(postfix, refresh=False)
            _pbar.update(1)

    pb_set(1, "Preprocessing")
    print(">>> Stage 1: Preprocessing...")
    df_raw = robust_read_csv(Path(args.data))
    f_pre = run_dir / "01_preprocessed_documents.csv"
    df = stage01_preprocess(df_raw, f_pre, max_tr_chars=int(args.max_tr_chars))
    pb_tick(docs=int(len(df)))

    f_cso_gz = run_dir / "02_cso_concepts.jsonl.gz"
    f_cso_stats = run_dir / "03_cso_entropy_stats.csv"
    cso_df: Optional[pd.DataFrame] = None

    if not args.skip_cso:
        pb_set(2, "CSO extraction")
        print(">>> Stage 2: CSO Extraction...")
        stage02_cso(
            df=df,
            out_jsonl_gz=f_cso_gz,
            cso_ttl=Path(args.cso_ttl),
            cso_model=args.cso_model,
            doc_batch_size=int(args.cso_doc_batch_size),
            topk=int(args.cso_topk),
            sim_threshold=float(args.cso_sim_threshold),
        )
        cso_df = load_cso_jsonl_gz(
            f_cso_gz,
            int(args.cso_entropy_topm),
            float(args.cso_entropy_margin),
            float(args.cso_entropy_softmax_temp),
        )
        cso_df.to_csv(f_cso_stats, index=False)
        pb_tick(cso_pairs=int(len(cso_df)))
    else:
        pb_set(2, "CSO extraction (skipped)")
        print(">>> Skipping CSO (flag set).")
        if f_cso_stats.exists():
            try:
                cso_df = pd.read_csv(f_cso_stats)
            except Exception:
                cso_df = None
        pb_tick(cso_pairs=int(len(cso_df)) if cso_df is not None else 0)

    # --- FIX: Merge CSO stats into main DF so Stages 6 & 7 can see them ---
    if cso_df is not None:
        df["doc_id"] = df["doc_id"].astype(str)
        cso_df["doc_id"] = cso_df["doc_id"].astype(str)

        # Only merge relevant columns
        cols_to_merge = ["doc_id"]
        for c in ["cso_top1_label", "cso_top1_sim", "entropy"]:
            if c in cso_df.columns:
                cols_to_merge.append(c)

        df = df.merge(cso_df[cols_to_merge], on="doc_id", how="left")
        print(f"   (Pipeline) Merged CSO stats. Documents with CSO label: {df['cso_top1_label'].notna().sum()}")
    # ----------------------------------------------------------------------

    pb_set(3, "BERTopic")
    print(">>> Stage 3: BERTopic...")
    f_dt = run_dir / "04_bertopic_doc_topics.csv"
    f_topics = run_dir / "04_bertopic_topics.csv"
    f_model = run_dir / "04_bertopic_model"
    f_emb = run_dir / "embeddings.npy"

    res04 = stage03_bertopic(
        df=df,
        out_doc_topics_csv=f_dt,
        out_topics_csv=f_topics,
        model_dir=f_model,
        embeddings_path=f_emb,
        embed_model=args.embed_model,
        umap_n_neighbors=int(args.umap_neighbors),
        umap_n_components=int(args.umap_components),
        umap_min_dist=float(args.umap_min_dist),
        hdb_min_cluster_size=int(args.min_cluster_size),
        hdb_min_samples=int(args.min_samples),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        extra_stopwords=args.extra_stopwords,
        reduce_topics=args.reduce_topics,
        vectorizer_min_df=int(args.vectorizer_min_df), # Passes new arg
    )

    dt = pd.read_csv(f_dt)
    topics = pd.read_csv(f_topics)
    embeddings = res04.embeddings if res04.embeddings is not None else np.load(f_emb)
    doc_ids = res04.embeddings_doc_ids if res04.embeddings_doc_ids is not None else dt["doc_id"].astype(str).tolist()
    doc_id_to_emb_idx = make_doc_id_to_emb_idx(doc_ids)

    pb_tick(
        topics=int(dt["topic"].nunique() - (1 if (dt["topic"] == -1).any() else 0)),
        outliers=int((dt["topic"] == -1).sum()),
    )

    pb_set(4, "Outlier reassignment")
    print(">>> Stage 4: Outlier Reassignment...")
    f_dt_re = run_dir / "06_reassigned_doc_topics.csv"
    dt = stage04_outlier_reassign(
        doc_topics=dt,
        embeddings=embeddings,
        doc_id_to_emb_idx=doc_id_to_emb_idx,
        cso_df=cso_df,
        out_csv=f_dt_re,
        max_outliers=int(args.max_outliers),
        sim_threshold=float(args.reassign_sim_threshold),
        entropy_threshold=float(args.entropy_threshold),
        margin_threshold=float(args.reassign_margin_threshold),
        margin_auto=bool(args.reassign_margin_auto),
        margin_quantile=float(args.reassign_margin_quantile),
        entropy_gate=bool(args.entropy_gate),
        entropy_auto=bool(args.entropy_auto),
        entropy_quantile=float(args.entropy_quantile),
        cso_sim_threshold_gate=float(args.cso_sim_threshold_gate),
    )
    if dt is None:
        dt = pd.read_csv(f_dt_re)

    pb_tick(outliers=int((dt["topic"] == -1).sum()))

    pb_set(5, f"LLM gating ({args.llm_mode})")
    print(f">>> Stage 5: LLM Gating (Mode: {args.llm_mode})...")
    f_dt_llm = run_dir / "07_llm_gated_doc_topics.csv"
    dt, outliers_post_llm = stage05_llm_gate(
        doc_topics=dt,
        topics_df=topics,
        df_text=df,
        embeddings=embeddings,
        doc_id_to_emb_idx=doc_id_to_emb_idx,
        out_csv=f_dt_llm,
        llm_mode=args.llm_mode,
        llm_model=args.llm_model,
        max_outliers=int(args.max_outliers),
        top_k_candidates=int(args.llm_topk),
        sim_floor=float(args.llm_sim_floor),
        cso_df=cso_df,
        entropy_threshold=float(args.entropy_threshold),
        low_sim_threshold=float(args.reassign_sim_threshold),
        reassign_margin_threshold=float(args.reassign_margin_threshold),
        entropy_auto=bool(args.entropy_auto),
        entropy_quantile=float(args.entropy_quantile),
        cso_sim_threshold=float(args.cso_sim_threshold_gate),
        cso_entropy_margin=0.05,
        entropy_gate=bool(args.entropy_gate),
        ollama_url=args.ollama_url,
        ollama_workers=int(args.ollama_workers),
        ollama_timeout=int(args.ollama_timeout),
    )
    pb_tick(outliers=int((dt["topic"] == -1).sum()), llm_left=int(len(outliers_post_llm) if outliers_post_llm is not None else 0))

    pb_set(6, "Augmentation")
    print(">>> Stage 6: Augmentation (Symbolic & Clustering)...")
    f_dt_aug = run_dir / "08_augmented_doc_topics.csv"
    f_topics_aug = run_dir / "08_new_topics_metadata.csv"
    residuals = dt[dt["topic"] == -1].copy()

    dt, new_topics = stage06_augment_residuals(
        doc_topics=dt,
        residual=residuals,
        df_text=df,
        embeddings=embeddings,
        doc_id_to_emb_idx=doc_id_to_emb_idx,
        out_doc_topics_csv=f_dt_aug,
        out_new_topics_csv=f_topics_aug,
        min_cluster_size=int(args.augment_min_cluster_size),
        min_samples=int(args.augment_min_samples),
        quality_min_topword_weight=float(args.augment_quality_min_topword_weight),
        run_dir=run_dir,
        augment_cso_purity_min=float(args.augment_cso_purity_min),
        symbolic_min_cluster_size=int(args.symbolic_min_cluster_size),
        symbolic_top1_sim_min=float(args.symbolic_top1_sim_min),
        symbolic_max_topics=int(args.symbolic_max_topics),
        extra_stopwords=args.extra_stopwords,
        umap_n_neighbors=int(args.augment_umap_neighbors),
        umap_n_components=int(args.augment_umap_components),
        umap_min_dist=float(args.augment_umap_min_dist),
        cluster_selection_method=str(args.augment_cluster_selection_method),
        seed=int(args.seed),
    )
    pb_tick(outliers=int((dt["topic"] == -1).sum()), new_topics=int(len(new_topics) if isinstance(new_topics, pd.DataFrame) else 0))

    pb_set(7, "Topic splitting")
    print(">>> Stage 7: Topic splitting...")
    f_dt_split = run_dir / "08_split_doc_topics.csv"
    f_topics_split = run_dir / "08_split_topics_metadata.csv"
    dt, split_topics = stage07_split_topics(
        doc_topics=dt,
        df_text=df,
        embeddings=embeddings,
        doc_id_to_emb_idx=doc_id_to_emb_idx,
        out_doc_topics_csv=f_dt_split,
        out_split_topics_csv=f_topics_split,
        enable=bool(args.split_topics),
        min_topic_size=int(args.split_min_topic_size),
        trigger_mean_sim=float(args.split_trigger_mean_sim),
        trigger_std_sim=float(args.split_trigger_std_sim),
        umap_n_neighbors=int(args.split_umap_neighbors),
        umap_n_components=int(args.split_umap_components),
        umap_min_dist=float(args.split_umap_min_dist),
        hdb_min_cluster_size=int(args.split_min_cluster_size),
        hdb_min_samples=int(args.split_min_samples),
        cluster_selection_method=str(args.split_cluster_selection_method),
        quality_min_topword_weight=float(args.split_quality_min_topword_weight),
        duplicate_sim_threshold=float(args.split_duplicate_sim_threshold),
        cso_purity_min=float(args.split_cso_purity_min),
        extra_stopwords=args.extra_stopwords,
        seed=int(args.seed),
    )

    # Merge split topics into new_topics metadata so they appear in final topics
    if isinstance(split_topics, pd.DataFrame) and (not split_topics.empty):
        if isinstance(new_topics, pd.DataFrame) and (not new_topics.empty):
            new_topics = pd.concat([new_topics, split_topics], ignore_index=True)
        else:
            new_topics = split_topics

    pb_tick(split_new_topics=int(len(split_topics) if isinstance(split_topics, pd.DataFrame) else 0))


    # Optional: merge micro-topics (tiny clusters) into nearest larger topics to reduce fragmentation
    merge_enable = bool(getattr(args, "merge_micro_topics", False)) and (not bool(getattr(args, "no_merge_micro_topics", False)))
    if merge_enable:
        print(">>> Micro-topic merge...")
        f_dt_micro = run_dir / "08_micro_merged_doc_topics.csv"
        f_merge_map = run_dir / "08_micro_merge_map.csv"
        dt, merge_map = stage07_merge_micro_topics(
            doc_topics=dt,
            embeddings=embeddings,
            doc_id_to_emb_idx=doc_id_to_emb_idx,
            min_topic_size=int(getattr(args, "micro_merge_min_topic_size", 10)),
            sim_threshold=float(getattr(args, "micro_merge_sim_threshold", 0.85)),
        )
        try:
            dt.to_csv(f_dt_micro, index=False)
        except Exception:
            pass
        try:
            merge_map.to_csv(f_merge_map, index=False)
        except Exception:
            pass
        try:
            merged_topics_n = int(len(merge_map))
            merged_docs_n = int(merge_map["from_size"].sum()) if (not merge_map.empty and "from_size" in merge_map.columns) else 0
        except Exception:
            merged_topics_n, merged_docs_n = 0, 0
        print(f"   (Micro-merge) merged_topics={merged_topics_n}, merged_docs={merged_docs_n}")
        pb_tick(micro_merged_topics=merged_topics_n, micro_merged_docs=merged_docs_n)

    
    # Optional final cleanup: centroid reassignment on remaining outliers (no LLM cost)
    if bool(getattr(args, "final_cleanup", True)):
        try:
            n_out_rem = int((dt["topic"] == -1).sum())
        except Exception:
            n_out_rem = 0
        if n_out_rem > 0:
            print(f">>> Final cleanup: centroid reassignment on {n_out_rem} remaining outliers...")
            f_dt_cleanup = run_dir / "08b_final_cleanup_doc_topics.csv"
            dt2 = stage04_outlier_reassign(
                doc_topics=dt,
                embeddings=embeddings,
                doc_id_to_emb_idx=doc_id_to_emb_idx,
                cso_df=cso_df,
                out_csv=f_dt_cleanup,
                max_outliers=int(args.max_outliers),
                sim_threshold=float(args.reassign_sim_threshold),
                entropy_threshold=float(args.entropy_threshold),
                margin_threshold=float(args.reassign_margin_threshold),
                margin_auto=bool(args.reassign_margin_auto),
                margin_quantile=float(args.reassign_margin_quantile),
                entropy_gate=bool(args.entropy_gate),
                entropy_auto=bool(args.entropy_auto),
                entropy_quantile=float(args.entropy_quantile),
                cso_sim_threshold_gate=float(args.cso_sim_threshold_gate),
            )
            if dt2 is None:
                dt2 = pd.read_csv(f_dt_cleanup)
            dt = dt2
    pb_set(8, "Finalize & keywords")
    print(">>> Stage 8: Finalizing...")
    f_dt_final = run_dir / "09_final_doc_topics.csv"
    f_topics_final = run_dir / "09_final_topics.csv"
    f_topics_final_raw = run_dir / "09_final_topics_raw.csv"

    final_dt, final_topics = stage07_finalize(
        doc_topics=dt,
        base_topics=topics,
        new_topics=new_topics,
        out_doc_topics_csv=f_dt_final,
        out_topics_csv=(f_topics_final_raw if bool(args.kw_refine) else f_topics_final),
        df_text=df,
        ctfidf_topn=int(args.ctfidf_topn),
        ctfidf_weighting=args.ctfidf_weighting,
        extra_stopwords=args.extra_stopwords,
    )

    if bool(args.kw_refine):
        try:
            final_topics = refine_final_keywords_in_pipeline(
                df_text=df,
                doc_topics=final_dt,
                topics_df=final_topics,
                out_topics_csv=f_topics_final,
                extra_stopwords=args.extra_stopwords,
                core_k=int(args.kw_refine_core_k),
                display_k=int(args.kw_refine_display_k),
                candidate_pool=int(args.kw_refine_candidate_pool),
                min_df_global=int(args.kw_refine_min_df_global),
                weighting=str(args.kw_refine_weighting),
            )
            try:
                final_topics.to_csv(run_dir / "09_final_topics_refined.csv", index=False)
            except Exception:
                pass
        except Exception as e:
            print(f"   (Finalize) Keyword refinement failed; keeping raw keywords. Reason: {e}")
            try:
                if (not f_topics_final.exists()) and f_topics_final_raw.exists():
                    import shutil
                    shutil.copyfile(f_topics_final_raw, f_topics_final)
            except Exception:
                pass
    pb_tick(
        coverage=round(float((final_dt["topic"] != -1).mean()), 3),
        topics=int(final_dt["topic"].nunique() - (1 if (final_dt["topic"] == -1).any() else 0)),
        outliers=int((final_dt["topic"] == -1).sum()),
    )

    pb_set(9, "Topic trends")
    print(">>> Stage 9: Topic Trends...")
    f_trends = run_dir / "10_topic_trends.csv"
    f_emerging = run_dir / "10_emerging_topics.csv"
    stage08_trends_topics(
        doc_topics=final_dt,
        topics_df=final_topics,
        out_csv=f_trends,
        out_emerging_csv=f_emerging,
        alpha=float(args.trend_alpha),
        emerging_if_burst=bool(args.emerging_if_burst),
        min_year_docs=int(args.min_year_docs),
        mk_method=str(args.mk_method),
        mk_lag_max=int(args.mk_lag_max),
        prevalence_mode=str(args.prevalence_mode),
        min_topic_size=int(args.trend_min_topic_size),
    )
    try:
        _em = pd.read_csv(f_emerging) if f_emerging.exists() else None
        em_n = int(len(_em)) if _em is not None else None
    except Exception:
        em_n = None
    pb_tick(emerging=em_n)

    pb_set(10, "Semantic shift")
    print(">>> Stage 10: Semantic Shift...")
    f_shift = run_dir / "11_semantic_shift.csv"
    stage09_semantic_shift(
        doc_topics=final_dt,
        embeddings=embeddings,
        doc_id_to_emb_idx=doc_id_to_emb_idx,
        topics_df=final_topics,
        out_csv=f_shift,
        min_docs_per_year=int(getattr(args, "min_docs_shift", 5)),
        normalize_by_year_gap=True,
    )
    pb_tick(shift=int(f_shift.exists()))

    if cso_df is not None:
        pb_set(11, "Trends & stability")
    print(">>> Stage 11: Trends & Stability...")
    cso_trends_ok = 0
    if (not bool(args.skip_cso)) and (cso_df is not None) and (not getattr(cso_df, "empty", True)):
        f_cso_trends = run_dir / "12_cso_concept_trends.csv"
        stage10_cso_concept_trends(final_dt, cso_df, f_cso_trends, float(args.trend_alpha))
        cso_trends_ok = int(f_cso_trends.exists())
    else:
        print("   (Stage 11) CSO trends skipped (no CSO concepts available).")

    stage11a_prevalence_baseline(final_dt, run_dir / "13_prevalence_baseline.csv")
    stage11b_model_stability(
        df=df,
        run_dir=run_dir,
        embed_model=args.embed_model,
        embeddings_path=f_emb,
        runs=int(args.stability_runs),
        batch_size=int(args.batch_size),
        umap_n_neighbors=int(args.umap_neighbors),
        umap_n_components=int(args.umap_components),
        umap_min_dist=float(args.umap_min_dist),
        hdb_min_cluster_size=int(args.min_cluster_size),
        hdb_min_samples=int(args.min_samples),
        extra_stopwords=args.extra_stopwords,
        reduce_topics=args.reduce_topics,
        out_csv=run_dir / "13_model_stability.csv",
        vectorizer_min_df=int(args.vectorizer_min_df),
        seed=int(args.seed),
    )
    pb_tick(cso_trends=cso_trends_ok, stab_runs=int(args.stability_runs))

    if _pbar is not None:
        _pbar.set_description("Done")
        _pbar.close()

    # Finalize stage timing
    now = time.perf_counter()
    if _current_stage_label is not None and _current_stage_t0 is not None:
        stage_times_sec[_current_stage_label] = stage_times_sec.get(_current_stage_label, 0.0) + (now - _current_stage_t0)

    wall_clock_sec = float(now - wall_t0)
    llm_calls = 0
    llm_assigned = 0
    try:
        if isinstance(final_dt, pd.DataFrame):
            if "llm_called" in final_dt.columns:
                llm_calls = int(final_dt["llm_called"].fillna(False).astype(bool).sum())
            if "assignment_source" in final_dt.columns:
                llm_assigned = int((final_dt["assignment_source"].astype(str) == "llm_gate").sum())
    except Exception:
        pass

    run_metrics = {
        "run_started_at_utc": run_started_at_utc,
        "command": run_cmd,
        "run_dir": str(run_dir),
        "wall_clock_seconds": wall_clock_sec,
        "stage_times_seconds": stage_times_sec,
        "llm_calls": llm_calls,
        "llm_assigned": llm_assigned,
        "mk_method": str(getattr(args, "mk_method", "classic")),
        "prevalence_mode_used": str(getattr(args, "prevalence_mode", "relative")),
        "seed": int(getattr(args, "seed", 0)),
        "stability_runs": int(getattr(args, "stability_runs", 0)),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    try:
        if torch.cuda.is_available():
            run_metrics["cuda_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        (run_dir / "14_run_metrics.json").write_text(json.dumps(run_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"WARNING: failed to write 14_run_metrics.json: {e}")

    print(f"✅ Pipeline complete.\nOutputs in: {run_dir}")
    return 0


# -----------------------------
# Extra: Evaluation Wrapper
# -----------------------------



def pipeline_main_legacy(argv: List[str]) -> int:
    args = pipeline_parse_args(argv)
    run_dir = Path(args.run_dir)
    ensure_dir(run_dir)

    print(f"🚀 Starting Pipeline. Output: {run_dir}")

    # --- Stage 1 ---
    print(">>> Stage 1: Preprocessing...")
    f_pre = run_dir / "01_preprocessed_documents.csv"
    if not f_pre.exists():
        df_raw = robust_read_csv(Path(args.data))
        df = stage01_preprocess(df_raw, f_pre, max_tr_chars=int(args.max_tr_chars))
    else:
        df = pd.read_csv(f_pre)
        # Ensure doc_id is string
        if "doc_id" in df.columns:
            df["doc_id"] = df["doc_id"].astype(str)
        print("   (Skipped) Loaded preprocessed data.")

    # --- Stage 2 ---
    f_cso_gz = run_dir / "02_cso_concepts.jsonl.gz"
    f_cso_stats = run_dir / "03_cso_entropy_stats.csv"
    cso_df = None

    if not args.skip_cso:
        if not f_cso_gz.exists():
            print(">>> Stage 2: CSO Extraction...")
            stage02_cso(
                df=df,
                out_jsonl_gz=f_cso_gz,
                cso_ttl=Path(args.cso_ttl),
                cso_model=args.cso_model,
                doc_batch_size=int(args.cso_doc_batch_size),
                topk=int(args.cso_topk),
                sim_threshold=float(args.cso_sim_threshold),
            )

        if not f_cso_stats.exists():
            cso_df = load_cso_jsonl_gz(f_cso_gz, int(args.cso_entropy_topm), float(args.cso_entropy_margin))
            cso_df.to_csv(f_cso_stats, index=False)
        else:
            cso_df = pd.read_csv(f_cso_stats)
            cso_df["doc_id"] = cso_df["doc_id"].astype(str)

    # --- Stage 3 ---
    print(">>> Stage 3: BERTopic...")
    f_dt = run_dir / "04_bertopic_doc_topics.csv"
    f_topics = run_dir / "04_bertopic_topics.csv"
    f_model = run_dir / "04_bertopic_model"
    f_emb = run_dir / "embeddings.npy"

    if not f_dt.exists():
        res03 = stage03_bertopic(
            df=df,
            out_doc_topics_csv=f_dt,
            out_topics_csv=f_topics,
            model_dir=f_model,
            embeddings_path=f_emb,
            embed_model=args.embed_model,
            umap_n_neighbors=int(args.umap_neighbors),
            umap_n_components=int(args.umap_components),
            umap_min_dist=float(args.umap_min_dist),
            hdb_min_cluster_size=int(args.min_cluster_size),
            hdb_min_samples=int(args.min_samples),
            seed=int(args.seed),
            batch_size=int(args.batch_size),
            extra_stopwords=args.extra_stopwords,
            reduce_topics=args.reduce_topics,
        )
        embeddings = res03.embeddings
        doc_ids = res03.embeddings_doc_ids
    else:
        print("   (Skipped) Loaded BERTopic results.")
        embeddings, doc_ids = load_embeddings_with_doc_ids(f_emb)

    dt = pd.read_csv(f_dt)
    dt["doc_id"] = dt["doc_id"].astype(str)
    topics = pd.read_csv(f_topics)
    doc_id_to_emb_idx = make_doc_id_to_emb_idx(doc_ids)

    # --- Stage 4 ---
    print(">>> Stage 4: Outlier Reassignment...")
    f_dt_re = run_dir / "06_reassigned_doc_topics.csv"
    if not f_dt_re.exists():
        dt = stage04_outlier_reassign(
            doc_topics=dt,
            embeddings=embeddings,
            doc_id_to_emb_idx=doc_id_to_emb_idx,
            cso_df=cso_df,
            out_csv=f_dt_re,
            max_outliers=int(args.max_outliers),
            sim_threshold=float(args.reassign_sim_threshold),
            entropy_threshold=float(args.entropy_threshold),
            margin_threshold=float(args.reassign_margin_threshold),
            margin_auto=bool(args.reassign_margin_auto),
            margin_quantile=float(args.reassign_margin_quantile),
                entropy_gate=bool(args.entropy_gate),
                entropy_auto=bool(args.entropy_auto),
                entropy_quantile=float(args.entropy_quantile),
                cso_sim_threshold_gate=float(args.cso_sim_threshold_gate),
        )
    else:
        dt = pd.read_csv(f_dt_re)
        dt["doc_id"] = dt["doc_id"].astype(str)

    # --- Stage 5 ---
    print(f">>> Stage 5: LLM Gating (Mode: {args.llm_mode})...")
    f_dt_llm = run_dir / "07_llm_gated_doc_topics.csv"
    if not f_dt_llm.exists() and args.llm_mode != "none":
        dt, _ = stage05_llm_gate(
            doc_topics=dt,
            topics_df=topics,
            df_text=df,
            embeddings=embeddings,
            doc_id_to_emb_idx=doc_id_to_emb_idx,
            out_csv=f_dt_llm,
            llm_mode=args.llm_mode,
            llm_model=args.llm_model,
            max_outliers=int(args.max_outliers),
            top_k_candidates=int(args.llm_topk),
            sim_floor=float(args.llm_sim_floor),
            cso_df=cso_df,
            entropy_threshold=float(args.entropy_threshold),
            low_sim_threshold=float(args.reassign_sim_threshold),
            reassign_margin_threshold=float(args.reassign_margin_threshold),
            entropy_auto=bool(args.entropy_auto),
            entropy_quantile=float(args.entropy_quantile),
            cso_sim_threshold=float(args.cso_sim_threshold_gate),
            entropy_gate=bool(args.entropy_gate),
            ollama_url=args.ollama_url,
            ollama_workers=int(args.ollama_workers),
            ollama_timeout=int(args.ollama_timeout),
        )
    elif f_dt_llm.exists():
        dt = pd.read_csv(f_dt_llm)
        dt["doc_id"] = dt["doc_id"].astype(str)

    # --- Stage 6 ---
    print(">>> Stage 6: Augmentation (Symbolic & Clustering)...")
    f_dt_aug = run_dir / "08_augmented_doc_topics.csv"
    f_topics_aug = run_dir / "08_new_topics_metadata.csv"
    residuals = dt[dt["topic"] == -1].copy()

    # We always run Stage 6 logic to ensure new_topics variable exists,
    # but we can save time if files exist by reloading.
    if not f_dt_aug.exists():
        dt, new_topics = stage06_augment_residuals(
            doc_topics=dt,
            residual=residuals,
            df_text=df,
            embeddings=embeddings,
            doc_id_to_emb_idx=doc_id_to_emb_idx,
            out_doc_topics_csv=f_dt_aug,
            out_new_topics_csv=f_topics_aug,
            min_cluster_size=int(args.augment_min_cluster_size),
            min_samples=int(args.augment_min_samples),
            quality_min_topword_weight=float(args.augment_quality_min_topword_weight),
            run_dir=run_dir,
            augment_cso_purity_min=float(args.augment_cso_purity_min),
            symbolic_min_cluster_size=int(args.symbolic_min_cluster_size),
            symbolic_top1_sim_min=float(args.symbolic_top1_sim_min),
            symbolic_max_topics=int(args.symbolic_max_topics),
            extra_stopwords=args.extra_stopwords,
            umap_n_neighbors=int(args.augment_umap_neighbors),
            umap_n_components=int(args.augment_umap_components),
            umap_min_dist=float(args.augment_umap_min_dist),
            cluster_selection_method=str(args.augment_cluster_selection_method),
            seed=int(args.seed),
        )
    else:
        dt = pd.read_csv(f_dt_aug)
        dt["doc_id"] = dt["doc_id"].astype(str)
        if f_topics_aug.exists():
            new_topics = pd.read_csv(f_topics_aug)
        else:
            new_topics = pd.DataFrame()

    # --- Stage 7 ---
    # --- Stage 7 ---
    print(">>> Stage 7: Finalizing...")
    f_dt_final = run_dir / "09_final_doc_topics.csv"
    f_topics_final = run_dir / "09_final_topics.csv"
    f_topics_final_raw = run_dir / "09_final_topics_raw.csv"

    final_dt, final_topics = stage07_finalize(
        doc_topics=dt,
        base_topics=topics,
        new_topics=new_topics,
        out_doc_topics_csv=f_dt_final,
        out_topics_csv=(f_topics_final_raw if bool(args.kw_refine) else f_topics_final),
        df_text=df,
        ctfidf_topn=int(args.ctfidf_topn),
        ctfidf_weighting=args.ctfidf_weighting,
        extra_stopwords=args.extra_stopwords,
    )

    if bool(args.kw_refine):
        try:
            final_topics = refine_final_keywords_in_pipeline(
                df_text=df,
                doc_topics=final_dt,
                topics_df=final_topics,
                out_topics_csv=f_topics_final,
                extra_stopwords=args.extra_stopwords,
                core_k=int(args.kw_refine_core_k),
                display_k=int(args.kw_refine_display_k),
                candidate_pool=int(args.kw_refine_candidate_pool),
                min_df_global=int(args.kw_refine_min_df_global),
                weighting=str(args.kw_refine_weighting),
            )
            try:
                final_topics.to_csv(run_dir / "09_final_topics_refined.csv", index=False)
            except Exception:
                pass
        except Exception as e:
            print(f"   (Finalize) Keyword refinement failed; keeping raw keywords. Reason: {e}")
            try:
                if (not f_topics_final.exists()) and f_topics_final_raw.exists():
                    import shutil
                    shutil.copyfile(f_topics_final_raw, f_topics_final)
            except Exception:
                pass

    # --- Stage 8 ---
    print(">>> Stage 9: Topic Trends...")
    f_trends = run_dir / "10_topic_trends.csv"
    f_emerging = run_dir / "10_emerging_topics.csv"

    # Generate trends
    # (Helper function logic embedded here for brevity since it's stats-heavy)
    dt_clean = final_dt[final_dt["topic"] != -1].copy()
    years_full = list(range(int(dt_clean["year"].min()), int(dt_clean["year"].max()) + 1))
    year_counts = dt_clean.groupby("year")["doc_id"].count().reindex(years_full, fill_value=0)

    rows = []
    label_map = dict(zip(final_topics["topic"], final_topics["label"]))

    for tid, size in tqdm(dt_clean["topic"].value_counts().items(), desc="Calculating Trends"):
        if size < 25: continue
        counts = dt_clean[dt_clean["topic"] == tid].groupby("year")["doc_id"].count().reindex(years_full, fill_value=0)
        norm_counts = (counts / year_counts.replace(0, np.nan)).fillna(0.0)

        mk_p, mk_tau, mk_trend = mann_kendall_test(norm_counts.tolist(), alpha=float(args.trend_alpha))
        starts, burst_max, _ = kleinberg_bursts(counts.tolist())

        rows.append({
            "topic": tid,
            "label": label_map.get(tid, str(tid)),
            "size": size,
            "mk_p": mk_p,
            "mk_tau": mk_tau,
            "mk_trend": mk_trend,
            "burst_max": burst_max
        })

    trends_df = pd.DataFrame(rows)
    if not trends_df.empty:
        trends_df.to_csv(f_trends, index=False)
        mask = (trends_df["mk_trend"] == "increasing")
        if args.emerging_if_burst:
            mask = mask | (trends_df["burst_max"] >= 1)
        trends_df[mask].to_csv(f_emerging, index=False)
    else:
        pd.DataFrame().to_csv(f_trends, index=False)
        pd.DataFrame().to_csv(f_emerging, index=False)

    print(f"✅ Pipeline complete. Results in {run_dir}")
    return 0

def stage12_evaluate_main(argv: List[str]) -> int:
    import subprocess

    script_path = Path("03_evaluate_topic_model.py")
    if not script_path.exists():
        script_path = Path(__file__).parent / "03_evaluate_topic_model.py"

    if not script_path.exists():
        print(f"❌ Error: Could not find evaluation script at {script_path}")
        return 1

    cmd = [sys.executable, str(script_path)] + argv
    print(f"🚀 Running evaluation wrapper: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation wrapper failed with code {e.returncode}")
        return int(e.returncode)


# -----------------------------
# Utility: Extended Report Wrapper (optional)
# -----------------------------
def stage13_extended_report_main(argv: List[str]) -> int:
    """Generate an extended, self-contained report from an existing --run_dir.

    This is a *post-hoc* wrapper: it does not re-run modeling. It only reads artifacts
    produced by the main pipeline and emits:
      - 14_extended_report.json
      - 14_extended_report.md
      - 14_topic_cards.csv
      - 14_assignment_source_stats.csv (if available)
      - 14_topic_examples.csv (optional, when --include_examples)

    Usage:
      python thesis_topic_trends_pipeline.py report --run_dir <RUN_DIR>
    """

    import argparse
    import random
    from datetime import datetime, timezone

    def _parse_args(a: List[str]):
        ap = argparse.ArgumentParser(
            prog="report",
            description="Generate an extended report from an existing pipeline run directory.",
        )
        ap.add_argument("--run_dir", required=True, help="Directory containing pipeline outputs")
        ap.add_argument("--out_dir", default="", help="Where to write report files (default: run_dir)")
        ap.add_argument("--out_json", default="", help="Override JSON output path")
        ap.add_argument("--out_md", default="", help="Override Markdown output path")
        ap.add_argument("--top_n", type=int, default=20, help="How many topics to show in top lists")
        ap.add_argument("--include_examples", action="store_true", help="Include example docs per topic")
        ap.add_argument("--examples_per_topic", type=int, default=3)
        ap.add_argument("--min_topic_size_for_examples", type=int, default=25)
        ap.add_argument("--sample_seed", type=int, default=42)
        ap.add_argument("--prefer_recent", action="store_true", help="Prefer recent-year docs for examples")
        ap.add_argument("--max_example_chars", type=int, default=300, help="Max chars per example snippet")
        return ap.parse_args(a)

    args = _parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"❌ report: run_dir does not exist: {run_dir}")
        return 1

    out_dir = Path(args.out_dir).expanduser() if str(args.out_dir).strip() else run_dir
    ensure_dir(out_dir)

    out_json = Path(args.out_json).expanduser() if str(args.out_json).strip() else (out_dir / "14_extended_report.json")
    out_md = Path(args.out_md).expanduser() if str(args.out_md).strip() else (out_dir / "14_extended_report.md")
    out_cards_csv = out_dir / "14_topic_cards.csv"
    out_src_csv = out_dir / "14_assignment_source_stats.csv"
    out_ex_csv = out_dir / "14_topic_examples.csv"

    def _read_first_csv(names: List[str], required: bool = True) -> Optional[pd.DataFrame]:
        for nm in names:
            p = run_dir / nm
            if p.exists():
                try:
                    return robust_read_csv(p)
                except Exception:
                    try:
                        return pd.read_csv(p)
                    except Exception:
                        continue
        if required:
            raise FileNotFoundError(f"Missing required CSV in {run_dir}. Tried: {names}")
        return None

    def _read_optional_csv(name: str) -> Optional[pd.DataFrame]:
        p = run_dir / name
        if not p.exists():
            return None
        try:
            return robust_read_csv(p)
        except Exception:
            try:
                return pd.read_csv(p)
            except Exception:
                return None

    # ---- Load core artifacts ----
    try:
        dt = _read_first_csv([
            "09_final_doc_topics.csv",
            "09_final_doc_topics_enriched.csv",
            "08_augmented_doc_topics.csv",
            "07_llm_gated_doc_topics.csv",
            "06_reassigned_doc_topics.csv",
            "04_bertopic_doc_topics.csv",
        ], required=True)
    except Exception as e:
        print(f"❌ report: could not load doc_topics: {e}")
        return 1

    # topics metadata is nice-to-have; we can still report without it
    topics_df = _read_first_csv([
        "09_final_topics.csv",
        "08_new_topics_metadata.csv",
        "04_bertopic_topics.csv",
    ], required=False)

    # preprocessed docs for titles/abstracts/examples
    docs_df = _read_optional_csv("01_preprocessed_documents.csv")

    # trend/shift/baselines
    trends_df = _read_optional_csv("10_topic_trends.csv")
    emerging_df = _read_optional_csv("10_emerging_topics.csv")
    shift_df = _read_optional_csv("11_semantic_shift.csv")
    cso_trends_df = _read_optional_csv("12_cso_concept_trends.csv")
    prev_df = _read_optional_csv("13_prevalence_baseline.csv")
    stab_df = _read_optional_csv("13_model_stability.csv")
    cso_stats_df = _read_optional_csv("03_cso_entropy_stats.csv")

    # ---- Normalize types ----
    dt = dt.copy()
    if "doc_id" in dt.columns:
        dt["doc_id"] = dt["doc_id"].astype(str)
    if "topic" not in dt.columns and "topic_id" in dt.columns:
        dt["topic"] = dt["topic_id"]
    if "topic" not in dt.columns:
        print("❌ report: doc_topics has no 'topic' column.")
        return 1
    dt["topic"] = pd.to_numeric(dt["topic"], errors="coerce").fillna(-1).astype(int)

    if "year" in dt.columns:
        dt["year"] = pd.to_numeric(dt["year"], errors="coerce").fillna(0).astype(int)

    # Ensure label/keywords in topics_df if present
    if topics_df is not None and not topics_df.empty:
        topics_df = topics_df.copy()
        if "topic" not in topics_df.columns and "Topic" in topics_df.columns:
            topics_df["topic"] = topics_df["Topic"]
        topics_df["topic"] = pd.to_numeric(topics_df.get("topic"), errors="coerce").fillna(-1).astype(int)
        for c in ("label", "keywords"):
            if c not in topics_df.columns:
                topics_df[c] = ""
        if "size" not in topics_df.columns:
            topics_df["size"] = 0
    else:
        topics_df = pd.DataFrame(columns=["topic", "label", "keywords", "size"])

    # ---- Core summary metrics ----
    n_docs = int(len(dt))
    outliers = int((dt["topic"] == -1).sum())
    assigned = int(n_docs - outliers)
    coverage = float(assigned / n_docs) if n_docs > 0 else 0.0

    topic_sizes = dt[dt["topic"] != -1]["topic"].value_counts().sort_index()
    n_topics = int(topic_sizes.shape[0])

    # Global balance entropy (normalized 0..1)
    if n_topics <= 1 or topic_sizes.sum() <= 0:
        balance_entropy = 0.0
    else:
        p = (topic_sizes.to_numpy(dtype=float) / float(topic_sizes.sum()))
        H = float(-np.sum(p * np.log(p + 1e-12)))
        balance_entropy = float(H / (math.log(float(n_topics)) + 1e-12))

    # Simple inequality proxy (Gini on topic sizes)
    def _gini(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0
        if np.all(x == 0):
            return 0.0
        x = np.sort(x)
        n = x.size
        cum = np.cumsum(x)
        # Gini = (n+1 - 2*sum(cum)/cum[-1]) / n
        return float((n + 1 - 2.0 * np.sum(cum) / (cum[-1] + 1e-12)) / n)

    gini_sizes = _gini(topic_sizes.to_numpy(dtype=float))

    # Assignment source + confidence stats (if present)
    source_counts = {}
    source_conf_stats = None
    if "assignment_source" in dt.columns:
        sc = dt["assignment_source"].fillna("").astype(str).value_counts()
        source_counts = {str(k): int(v) for k, v in sc.items()}

        if "assignment_confidence" in dt.columns:
            tmp = dt.copy()
            tmp["assignment_confidence"] = pd.to_numeric(tmp["assignment_confidence"], errors="coerce")
            tmp = tmp[(tmp["topic"] != -1) & (tmp["assignment_source"].notna())].copy()
            if not tmp.empty:
                grp = tmp.groupby("assignment_source")["assignment_confidence"].agg(
                    n="count",
                    mean="mean",
                    median="median",
                    p10=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 10)),
                    p25=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 25)),
                    p75=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 75)),
                    p90=lambda s: float(np.nanpercentile(s.to_numpy(dtype=float), 90)),
                ).reset_index()
                source_conf_stats = grp

    # Uncertainty proxy = 1 - confidence (where available)
    unc_stats = None
    if "assignment_confidence" in dt.columns:
        conf = pd.to_numeric(dt["assignment_confidence"], errors="coerce")
        conf = conf.where(np.isfinite(conf), np.nan)
        unc = (1.0 - conf).clip(lower=0.0, upper=1.0)
        unc_assigned = unc[dt["topic"] != -1]
        if unc_assigned.notna().any():
            unc_stats = {
                "mean": float(np.nanmean(unc_assigned.to_numpy(dtype=float))),
                "median": float(np.nanmedian(unc_assigned.to_numpy(dtype=float))),
                "p10": float(np.nanpercentile(unc_assigned.to_numpy(dtype=float), 10)),
                "p90": float(np.nanpercentile(unc_assigned.to_numpy(dtype=float), 90)),
            }

    # CSO stats (optional)
    cso_summary = None
    if cso_stats_df is not None and not cso_stats_df.empty:
        cso_stats_df = cso_stats_df.copy()
        if "doc_id" in cso_stats_df.columns:
            cso_stats_df["doc_id"] = cso_stats_df["doc_id"].astype(str)
        cso_summary = {
            "rows": int(len(cso_stats_df)),
            "mean_entropy": float(pd.to_numeric(cso_stats_df.get("entropy"), errors="coerce").mean())
            if "entropy" in cso_stats_df.columns else None,
            "mean_top1_sim": float(pd.to_numeric(cso_stats_df.get("cso_top1_sim"), errors="coerce").mean())
            if "cso_top1_sim" in cso_stats_df.columns else None,
        }

    # ---- Topic cards (merge multiple sources) ----
    cards = pd.DataFrame({"topic": topic_sizes.index.astype(int), "size": topic_sizes.values.astype(int)})

    # attach labels/keywords
    if topics_df is not None and not topics_df.empty:
        if "keywords_display" in topics_df.columns:
            tmeta = topics_df[["topic", "label", "keywords_display"]].drop_duplicates("topic").rename(columns={"keywords_display": "keywords"})
        else:
            tmeta = topics_df[["topic", "label", "keywords"]].drop_duplicates("topic")
        cards = cards.merge(tmeta, on="topic", how="left")
    else:
        cards["label"] = ""
        cards["keywords"] = ""

    # trend metrics
    if trends_df is not None and not trends_df.empty and "topic" in trends_df.columns:
        tr = trends_df.copy()
        tr["topic"] = pd.to_numeric(tr["topic"], errors="coerce").fillna(-1).astype(int)
        keep_cols = [c for c in ["topic", "mk_q", "mk_p", "mk_tau", "mk_trend", "slope", "burst_max", "burst_years"] if c in tr.columns]
        tr = tr[keep_cols].drop_duplicates("topic")
        cards = cards.merge(tr, on="topic", how="left")

    # shift metrics
    if shift_df is not None and not shift_df.empty and "topic" in shift_df.columns:
        sh = shift_df.copy()
        sh["topic"] = pd.to_numeric(sh["topic"], errors="coerce").fillna(-1).astype(int)
        keep_cols = [c for c in ["topic", "drift_total_dist", "drift_step_mean", "drift_robust_score"] if c in sh.columns]
        sh = sh[keep_cols].drop_duplicates("topic")
        cards = cards.merge(sh, on="topic", how="left")

    # prevalence baseline
    if prev_df is not None and not prev_df.empty and "topic" in prev_df.columns:
        pv = prev_df.copy()
        pv["topic"] = pd.to_numeric(pv["topic"], errors="coerce").fillna(-1).astype(int)
        if "p" in pv.columns and "q" not in pv.columns:
            try:
                pv["q"] = bh_fdr(pd.to_numeric(pv["p"], errors="coerce").fillna(1.0).to_numpy())
            except Exception:
                pv["q"] = np.nan
        keep_cols = [c for c in ["topic", "method", "beta_year_z", "p", "q"] if c in pv.columns]
        pv = pv[keep_cols].drop_duplicates("topic")
        cards = cards.merge(pv, on="topic", how="left")

    # fill label fallback
    cards["label"] = cards["label"].fillna("")
    cards["keywords"] = cards["keywords"].fillna("")
    cards.loc[cards["label"].astype(str).str.len() == 0, "label"] = cards.loc[cards["label"].astype(str).str.len() == 0, "topic"].map(lambda t: f"Topic {int(t)}")

    # ---- Examples (optional) ----
    examples = {}
    examples_df = None
    if bool(args.include_examples) and docs_df is not None and not docs_df.empty and "doc_id" in docs_df.columns:
        docs_df = docs_df.copy()
        docs_df["doc_id"] = docs_df["doc_id"].astype(str)

        # choose best available columns
        title_col = "english_title" if "english_title" in docs_df.columns else ("english_title_clean" if "english_title_clean" in docs_df.columns else None)
        abs_col = "abstract_en" if "abstract_en" in docs_df.columns else ("abstract_en_clean" if "abstract_en_clean" in docs_df.columns else ("text" if "text" in docs_df.columns else None))

        if title_col and abs_col:
            merged = dt.merge(docs_df[["doc_id", title_col, abs_col]], on="doc_id", how="left")
            merged = merged[merged["topic"] != -1].copy()

            # only topics above threshold and top by size
            top_topics = cards.sort_values("size", ascending=False)
            top_topics = top_topics[top_topics["size"] >= int(args.min_topic_size_for_examples)]
            top_topics = top_topics.head(int(args.top_n))["topic"].astype(int).tolist()

            random.seed(int(args.sample_seed))
            rows_ex = []

            for tid in top_topics:
                sub = merged[merged["topic"] == int(tid)].copy()
                if sub.empty:
                    continue

                # deterministic preference: highest confidence; optionally prefer recent years
                if "assignment_confidence" in sub.columns:
                    sub["assignment_confidence"] = pd.to_numeric(sub["assignment_confidence"], errors="coerce").fillna(0.0)
                else:
                    sub["assignment_confidence"] = 0.0

                sort_cols = ["assignment_confidence"]
                asc = [False]
                if bool(args.prefer_recent) and "year" in sub.columns:
                    sort_cols.append("year"); asc.append(False)

                sub = sub.sort_values(sort_cols, ascending=asc)

                k = int(max(1, args.examples_per_topic))
                picked = sub.head(k).copy()
                ex_list = []
                for _, r in picked.iterrows():
                    title = str(r.get(title_col, "") or "").strip()
                    abst = str(r.get(abs_col, "") or "").strip()
                    abst = abst.replace("\\n", " ").strip()
                    if len(abst) > int(args.max_example_chars):
                        abst = abst[: int(args.max_example_chars)].rstrip() + "..."
                    ex_list.append({
                        "doc_id": str(r.get("doc_id")),
                        "year": int(r.get("year", 0)) if pd.notna(r.get("year", 0)) else None,
                        "confidence": float(r.get("assignment_confidence", 0.0)) if pd.notna(r.get("assignment_confidence", 0.0)) else None,
                        "title": title,
                        "snippet": abst,
                    })
                    rows_ex.append({
                        "topic": int(tid),
                        "doc_id": str(r.get("doc_id")),
                        "year": int(r.get("year", 0)) if pd.notna(r.get("year", 0)) else None,
                        "confidence": float(r.get("assignment_confidence", 0.0)) if pd.notna(r.get("assignment_confidence", 0.0)) else None,
                        "title": title,
                        "snippet": abst,
                    })

                examples[str(int(tid))] = ex_list

            if rows_ex:
                examples_df = pd.DataFrame(rows_ex)

    # ---- Top lists for the report ----
    top_n = int(max(1, args.top_n))

    top_by_size = cards.sort_values("size", ascending=False).head(top_n).copy()

    top_emerging = None
    if emerging_df is not None and not emerging_df.empty:
        top_emerging = emerging_df.copy()
        # Try to sort by mk_q then burst_max (common columns)
        if "mk_q" in top_emerging.columns:
            top_emerging["mk_q"] = pd.to_numeric(top_emerging["mk_q"], errors="coerce")
        if "burst_max" in top_emerging.columns:
            top_emerging["burst_max"] = pd.to_numeric(top_emerging["burst_max"], errors="coerce")
        sort_cols = [c for c in ["mk_q", "burst_max"] if c in top_emerging.columns]
        if sort_cols:
            asc = [True if c == "mk_q" else False for c in sort_cols]
            top_emerging = top_emerging.sort_values(sort_cols, ascending=asc)
        top_emerging = top_emerging.head(top_n)

    top_shift = None
    if shift_df is not None and not shift_df.empty:
        top_shift = shift_df.copy()
        if "drift_robust_score" in top_shift.columns:
            top_shift["drift_robust_score"] = pd.to_numeric(top_shift["drift_robust_score"], errors="coerce")
            top_shift = top_shift.sort_values("drift_robust_score", ascending=False)
        top_shift = top_shift.head(top_n)

    top_cso = None
    if cso_trends_df is not None and not cso_trends_df.empty:
        top_cso = cso_trends_df.copy()
        if "mk_q" in top_cso.columns:
            top_cso["mk_q"] = pd.to_numeric(top_cso["mk_q"], errors="coerce")
            top_cso = top_cso.sort_values("mk_q", ascending=True)
        top_cso = top_cso.head(top_n)

    # ---- Assemble JSON ----
    report = {
        "run_dir": str(run_dir),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "docs_total": n_docs,
            "docs_assigned": assigned,
            "docs_outliers": outliers,
            "coverage": coverage,
            "topics": n_topics,
            "topic_size_min": int(topic_sizes.min()) if n_topics > 0 else 0,
            "topic_size_median": float(topic_sizes.median()) if n_topics > 0 else 0.0,
            "topic_size_max": int(topic_sizes.max()) if n_topics > 0 else 0,
            "global_balance_entropy_norm": balance_entropy,
            "topic_size_gini": gini_sizes,
            "assignment_source_counts": source_counts,
            "assignment_uncertainty_proxy": unc_stats,
            "cso_summary": cso_summary,
            "stability": (stab_df.iloc[0].to_dict() if stab_df is not None and not stab_df.empty else None),
        },
        "top_topics_by_size": top_by_size.to_dict(orient="records"),
        "top_emerging_topics": (top_emerging.to_dict(orient="records") if top_emerging is not None else None),
        "top_semantic_shift": (top_shift.to_dict(orient="records") if top_shift is not None else None),
        "top_cso_concepts": (top_cso.to_dict(orient="records") if top_cso is not None else None),
        "examples": examples if examples else None,
    }

    # ---- Write CSV artifacts ----
    try:
        cards.sort_values("size", ascending=False).to_csv(out_cards_csv, index=False)
    except Exception as e:
        print(f"⚠️ report: could not write {out_cards_csv.name}: {e}")

    if source_conf_stats is not None and isinstance(source_conf_stats, pd.DataFrame):
        try:
            source_conf_stats.to_csv(out_src_csv, index=False)
        except Exception as e:
            print(f"⚠️ report: could not write {out_src_csv.name}: {e}")

    if examples_df is not None and isinstance(examples_df, pd.DataFrame) and not examples_df.empty:
        try:
            examples_df.to_csv(out_ex_csv, index=False)
        except Exception as e:
            print(f"⚠️ report: could not write {out_ex_csv.name}: {e}")

    # ---- Write JSON ----
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ report: failed to write JSON: {e}")
        return 1

    # ---- Write Markdown ----
    def _fmt(x, nd=3):
        try:
            if x is None or (isinstance(x, float) and not np.isfinite(x)):
                return "NA"
            if isinstance(x, float):
                return str(round(float(x), nd))
            return str(x)
        except Exception:
            return str(x)

    md_lines: List[str] = []
    md_lines.append("# Extended Report\n")
    md_lines.append(f"\nRun: `{run_dir}`\n\n")
    md_lines.append(f"Generated: `{report['generated_at']}`\n\n")

    md_lines.append("## Summary\n\n")
    md_lines.append(f"- Documents: **{n_docs}**\n")
    md_lines.append(f"- Assigned: **{assigned}**\n")
    md_lines.append(f"- Outliers: **{outliers}**\n")
    md_lines.append(f"- Coverage: **{_fmt(coverage, 4)}**\n")
    md_lines.append(f"- Topics (excluding outlier): **{n_topics}**\n")
    md_lines.append(f"- Global balance entropy (norm): **{_fmt(balance_entropy, 4)}**\n")
    md_lines.append(f"- Topic size Gini: **{_fmt(gini_sizes, 4)}**\n")
    if unc_stats is not None:
        md_lines.append(
            "- Assignment uncertainty proxy (1-confidence): "
            f"mean={_fmt(unc_stats.get('mean'),4)}, "
            f"median={_fmt(unc_stats.get('median'),4)}, "
            f"p10={_fmt(unc_stats.get('p10'),4)}, "
            f"p90={_fmt(unc_stats.get('p90'),4)}\n"
        )
    if stab_df is not None and not stab_df.empty:
        row0 = stab_df.iloc[0].to_dict()
        md_lines.append(
            f"- Stability: mean_ami={_fmt(row0.get('mean_ami'))}, "
            f"std_ami={_fmt(row0.get('std_ami'))}, runs={row0.get('runs')}\n"
        )

    if source_counts:
        md_lines.append("\n## Assignment sources\n\n")
        md_lines.append("| source | docs |\n|---|---:|\n")
        for k, v in sorted(source_counts.items(), key=lambda kv: kv[1], reverse=True):
            md_lines.append(f"| {k} | {int(v)} |\n")

    if source_conf_stats is not None and not source_conf_stats.empty:
        md_lines.append("\n## Confidence by source\n\n")
        md_lines.append("| source | n | mean | median | p10 | p25 | p75 | p90 |\n")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in source_conf_stats.iterrows():
            md_lines.append(
                "| {src} | {n} | {mean} | {median} | {p10} | {p25} | {p75} | {p90} |\n".format(
                    src=str(r.get("assignment_source")),
                    n=int(r.get("n", 0)),
                    mean=_fmt(r.get("mean")),
                    median=_fmt(r.get("median")),
                    p10=_fmt(r.get("p10")),
                    p25=_fmt(r.get("p25")),
                    p75=_fmt(r.get("p75")),
                    p90=_fmt(r.get("p90")),
                )
            )

    md_lines.append("\n## Top topics by size\n\n")
    md_lines.append("| topic | size | label | keywords |\n|---:|---:|---|---|\n")
    for _, r in top_by_size.iterrows():
        md_lines.append(
            f"| {int(r['topic'])} | {int(r['size'])} | {str(r.get('label',''))} | {str(r.get('keywords',''))} |\n"
        )

    if top_emerging is not None and not top_emerging.empty:
        md_lines.append("\n## Emerging topics\n\n")
        cols = [c for c in ["topic", "label", "mk_q", "mk_trend", "slope", "burst_max", "burst_years"] if c in top_emerging.columns]
        if cols:
            md_lines.append("| " + " | ".join(cols) + " |\n")
            md_lines.append("|" + "|".join(["---"] * len(cols)) + "|\n")
            for _, r in top_emerging[cols].iterrows():
                md_lines.append("| " + " | ".join([_fmt(r.get(c)) for c in cols]) + " |\n")

    if top_shift is not None and not top_shift.empty:
        md_lines.append("\n## Semantic shift leaders\n\n")
        cols = [c for c in ["topic", "label", "years", "drift_robust_score", "drift_total_dist", "drift_step_mean"] if c in top_shift.columns]
        if cols:
            md_lines.append("| " + " | ".join(cols) + " |\n")
            md_lines.append("|" + "|".join(["---"] * len(cols)) + "|\n")
            for _, r in top_shift[cols].iterrows():
                md_lines.append("| " + " | ".join([_fmt(r.get(c)) for c in cols]) + " |\n")

    if top_cso is not None and not top_cso.empty:
        md_lines.append("\n## CSO concept trends (baseline)\n\n")
        cols = [c for c in ["concept", "size", "mk_q", "mk_trend", "mk_tau", "burst_max", "burst_years"] if c in top_cso.columns]
        if cols:
            md_lines.append("| " + " | ".join(cols) + " |\n")
            md_lines.append("|" + "|".join(["---"] * len(cols)) + "|\n")
            for _, r in top_cso[cols].iterrows():
                md_lines.append("| " + " | ".join([_fmt(r.get(c)) for c in cols]) + " |\n")

    if examples:
        md_lines.append("\n## Topic examples (sample)\n\n")
        shown = 0
        for tid in [str(int(x)) for x in top_by_size["topic"].head(top_n).tolist()]:
            exs = examples.get(tid, [])
            if not exs:
                continue
            shown += 1
            md_lines.append(f"\n### Topic {tid}\n\n")
            for ex in exs:
                md_lines.append(f"- ({_fmt(ex.get('year'))}) conf={_fmt(ex.get('confidence'),4)} — **{ex.get('title','')}**\n")
                sn = str(ex.get("snippet", "")).strip()
                if sn:
                    md_lines.append(f"  - {sn}\n")
            if shown >= top_n:
                break

    md_text = "".join(md_lines)
    try:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md_text)
    except Exception as e:
        print(f"❌ report: failed to write Markdown: {e}")
        return 1

    print("✅ Extended report written:")
    print(f"- {out_json}")
    print(f"- {out_md}")
    print(f"- {out_cards_csv}")
    if source_conf_stats is not None:
        print(f"- {out_src_csv}")
    if examples_df is not None and not examples_df.empty:
        print(f"- {out_ex_csv}")
    return 0



# -----------------------------
# Utility: Ablation Logic
# -----------------------------
def stage14_ablation_compare(
    current_run_dir: Path,
    out_dir: Path,
    extra_runs: List[str],
    self_label: str,
    stage14_py: Optional[Path],
    ablation_py: Optional[Path],
    copy_artifacts: bool,
    top_n: int,
    silhouette_sample: int,
    silhouette_per_topic_max: int,
) -> None:
    import subprocess

    ensure_dir(out_dir)

    script_path = ablation_py
    if not script_path or not script_path.exists():
        candidates = [
            Path("04_ablation_compare_with_entropy.py"),
            Path(__file__).parent / "04_ablation_compare_with_entropy.py",
        ]
        for c in candidates:
            if c.exists():
                script_path = c
                break

    if not script_path or not script_path.exists():
        raise FileNotFoundError("Could not find 04_ablation_compare_with_entropy.py")

    runs_arg: List[str] = []
    if current_run_dir and current_run_dir.exists():
        runs_arg.append(f"{self_label}={current_run_dir}")
    runs_arg.extend(extra_runs)

    if not runs_arg:
        raise ValueError("No valid runs provided for ablation comparison.")

    cmd = [
        sys.executable, str(script_path),
        "--out_dir", str(out_dir),
        "--runs",
    ] + runs_arg + [
        "--top_n", str(int(top_n)),
        "--silhouette_sample", str(int(silhouette_sample)),
        "--silhouette_per_topic_max", str(int(silhouette_per_topic_max)),
    ]

    if stage14_py:
        cmd.extend(["--stage14_py", str(stage14_py)])

    if copy_artifacts:
        cmd.append("--copy_artifacts")

    print(f"🚀 Running ablation wrapper: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def stage14_ablation_compare_main(argv: Optional[List[str]] = None) -> int:
    args = pipeline_parse_args(argv or [])

    if not (isinstance(args.ablation_out_dir, str) and args.ablation_out_dir.strip()):
        print("ERROR: Ablation wrapper requires --ablation_out_dir.")
        return 1

    run_dir = Path(args.run_dir) if args.run_dir else Path(".")
    stage14_py = Path(args.ablation_stage14_py).expanduser() if str(args.ablation_stage14_py).strip() else None
    ablation_py = Path(args.ablation_py).expanduser() if str(args.ablation_py).strip() else None

    try:
        stage14_ablation_compare(
            current_run_dir=run_dir,
            out_dir=Path(args.ablation_out_dir).expanduser(),
            extra_runs=list(args.ablation_runs or []),
            self_label=str(args.ablation_label or "main"),
            stage14_py=stage14_py,
            ablation_py=ablation_py,
            copy_artifacts=bool(args.ablation_copy_artifacts),
            top_n=int(args.ablation_top_n),
            silhouette_sample=int(args.ablation_silhouette_sample),
            silhouette_per_topic_max=int(args.ablation_silhouette_per_topic_max),
        )
        print("\n✅ Ablation complete.")
        return 0
    except Exception as e:
        print(f"\n❌ Ablation failed: {e}")
        return 1


# -----------------------------
# Unified Main
# -----------------------------
def unified_main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        pipeline_parse_args(["--help"])
        return 0

    cmd = argv[0]
    sub_argv = argv[1:]

    if cmd in {"evaluate", "stage12"}:
        return stage12_evaluate_main(sub_argv)

    if cmd in {"report", "stage13"}:
        return stage13_extended_report_main(sub_argv)

    if cmd in {"ablation", "stage14", "stage17"}:
        return stage14_ablation_compare_main(sub_argv)

    return pipeline_main(argv)


if __name__ == "__main__":
    raise SystemExit(unified_main())
