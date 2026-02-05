#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
03_evaluate_topic_model.py
- Adds Performance Metrics (Wall Clock, LLM Calls)
'''
from __future__ import annotations

import os
# Disable Tokenizers Parallelism to prevent deadlock/warnings with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def _running_in_ipython() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        return get_ipython() is not None
    except Exception:
        return False


def _safe_read_csv(p: Path) -> Optional[pd.DataFrame]:
    try:
        # standard robust read
        return pd.read_csv(p, on_bad_lines='skip', engine='python')
    except Exception:
        return None


def _mk_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pick_metric(d: dict, keys: Iterable[str], default=float("nan")):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _as_float(x: Any, default=float("nan")) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


# ----------------------------------------------------------------------------
# Run dir inference / discovery
# ----------------------------------------------------------------------------

def infer_run_dir(provided: Optional[str]) -> Optional[Path]:
    if provided:
        p = Path(provided)
        if p.exists() and p.is_dir():
            return p
    # Heuristic search
    markers = ["09_final_doc_topics.csv", "04_bertopic_doc_topics.csv"]
    roots = [Path("."), Path("runs"), Path("../runs")]

    for root in roots:
        if not root.exists(): continue
        # Check root itself
        if any((root/m).exists() for m in markers):
            return root
        # Check subdirs
        for sub in root.iterdir():
            if sub.is_dir() and any((sub/m).exists() for m in markers):
                print(f" inferred run_dir: {sub}")
                return sub
    return None


# ----------------------------------------------------------------------------
# File loading
# ----------------------------------------------------------------------------

DOC_TOPICS_PRIORITY = [
    "09_final_doc_topics.csv",
    "09_doc_topics.csv",
    "08_augmented_doc_topics.csv",
    "07_llm_gated_doc_topics.csv",
    "06_reassigned_doc_topics.csv",
    "04_bertopic_doc_topics.csv",
]

TOPICS_PRIORITY = [
    "09_final_topics.csv",
    "09_topics.csv",
    "08_new_topics_metadata.csv",
    "04_bertopic_topics.csv",
]


def _normalize_topic_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "topic" not in df.columns and "topic_id" in df.columns:
        df = df.rename(columns={"topic_id": "topic"})
    if "topic" in df.columns:
        # Ensure numeric
        df["topic"] = pd.to_numeric(df["topic"], errors="coerce").fillna(-1).astype(int)
    return df


def _load_doc_topics(run_dir: Path) -> pd.DataFrame:
    for name in DOC_TOPICS_PRIORITY:
        p = run_dir / name
        if p.exists():
            print(f" loading doc_topics from: {name}")
            df = _safe_read_csv(p)
            if df is None or df.empty:
                continue
            df = _normalize_topic_cols(df)
            if "doc_id" not in df.columns:
                for c in ("document_id", "id", "docid"):
                    if c in df.columns:
                        df = df.rename(columns={c: "doc_id"})
                        break
            if "doc_id" in df.columns:
                df["doc_id"] = df["doc_id"].astype(str)
            return df
    print(f" WARNING: No doc_topics CSV found in {run_dir}")
    return pd.DataFrame()


def _load_topics(run_dir: Path) -> pd.DataFrame:
    for name in TOPICS_PRIORITY:
        p = run_dir / name
        if p.exists():
            df = _safe_read_csv(p)
            if df is None or df.empty:
                continue
            df = _normalize_topic_cols(df)
            return df
    return pd.DataFrame()


def _infer_preproc_path(run_dir: Path, preproc_csv: Optional[Path]) -> Optional[Path]:
    if preproc_csv is not None:
        if preproc_csv.exists():
            return preproc_csv
    p = run_dir / "01_preprocessed_documents.csv"
    if p.exists():
        return p
    return None


def _load_preproc_df(run_dir: Path, preproc_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    p = _infer_preproc_path(run_dir, preproc_csv)
    if p is None:
        return None
    df = _safe_read_csv(p)
    if df is None or "doc_id" not in df.columns:
        return None

    # Construct text field if missing
    if "text" not in df.columns:
        t = df["english_title"].astype(str) if "english_title" in df.columns else ""
        a = df["abstract_en"].astype(str) if "abstract_en" in df.columns else ""
        if "english_title" not in df.columns:
             # Try clean columns
             t = df["english_title_clean"].astype(str) if "english_title_clean" in df.columns else ""
             a = df["abstract_en_clean"].astype(str) if "abstract_en_clean" in df.columns else ""

        df["text"] = (t + "\n" + a).str.strip()

    df["doc_id"] = df["doc_id"].astype(str)
    df["text"] = df["text"].astype(str)
    return df


def _load_texts(run_dir: Path, preproc_csv: Optional[Path]) -> Dict[str, str]:
    df = _load_preproc_df(run_dir, preproc_csv)
    if df is None or df.empty:
        print(" WARNING: Preprocessed text not found. Coherence will be skipped.")
        return {}
    return {r["doc_id"]: r["text"] for _, r in df.iterrows()}


def _compute_embeddings_sentence_transformers(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(f"Install sentence-transformers to auto-compute embeddings. Error: {e}")

    model = SentenceTransformer(model_name)
    try:
        emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    except TypeError:
        emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return np.asarray(emb)


def _load_embeddings(
    run_dir: Path,
    preproc_df: Optional[pd.DataFrame],
    embed_model: Optional[str],
    embed_batch_size: int,
    auto_compute: bool,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:

    # 1. Try loading existing
    for name in ["embeddings.npy", "02_embeddings.npy"]:
        p = run_dir / name
        if not p.exists(): continue

        try:
            emb = np.load(p)
        except Exception:
            continue

        # Look for sidecar
        doc_ids = None
        sidecars = [
            p.with_name(p.stem + "_doc_ids.json"),
            run_dir / "embeddings_doc_ids.json",
            run_dir / "02_embeddings_doc_ids.json"
        ]
        for sc in sidecars:
            if sc.exists():
                try:
                    doc_ids = json.loads(sc.read_text(encoding="utf-8"))
                    doc_ids = [str(d) for d in doc_ids]
                    break
                except: pass

        if doc_ids:
            return emb, doc_ids

        # Fallback: align with preproc if count matches
        if preproc_df is not None and len(preproc_df) == emb.shape[0]:
            print("   (Embeddings) Sidecar missing, inferred doc_ids from preproc CSV.")
            return emb, preproc_df["doc_id"].astype(str).tolist()

    # 2. Auto-compute
    if auto_compute and embed_model and preproc_df is not None and not preproc_df.empty:
        try:
            print(f" [auto] Computing embeddings for {len(preproc_df)} docs using: {embed_model}")
            texts = preproc_df["text"].astype(str).tolist()
            emb = _compute_embeddings_sentence_transformers(texts, embed_model, embed_batch_size)
            doc_ids = preproc_df["doc_id"].astype(str).tolist()

            # Cache them
            np.save(run_dir / "embeddings.npy", emb)
            (run_dir / "embeddings_doc_ids.json").write_text(json.dumps(doc_ids), encoding="utf-8")
            return emb, doc_ids
        except Exception as e:
            print(f" WARNING: auto-embedding failed: {e}")

    return None, None


# ----------------------------------------------------------------------------
# Keyword Parsing & Diversity
# ----------------------------------------------------------------------------

def parse_top_words(s: str, top_n: int) -> List[str]:
    s = str(s or "").strip()
    if not s: return []

    # 1. AST literal eval for ['a','b'] format
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("('") and s.endswith("')")):
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(i).strip().lower() for i in parsed if str(i).strip()][:top_n]
        except: pass

    # 2. Split by delimiters
    # Clean quotes
    s_clean = s.replace('"', '').replace("'", "").replace("[", "").replace("]", "")
    # Split by ; , or newline
    toks = re.split(r"[,;\n]\s*", s_clean)

    out = []
    seen = set()
    for t in toks:
        t = t.strip().lower()
        # Remove weird symbols but allow some
        t = re.sub(r"^[^a-z0-9]+", "", t)
        if t and t not in seen:
            out.append(t)
            seen.add(t)

    return out[:top_n]


def _topic_keywords_row(r: pd.Series) -> str:
    # Prefer refined keyword columns if present (pipeline post-finalization refinement)
    for c in ("keywords_core", "keywords", "keywords_display", "representation", "topic_label", "label", "name", "Name"):
        if c in r and pd.notna(r[c]):
            return str(r[c])
    return ""


def topic_diversity(topics: pd.DataFrame, top_n: int) -> float:
    if topics is None or topics.empty: return float("nan")
    if "topic" not in topics.columns: return float("nan")

    valid = topics[topics["topic"] != -1]
    if valid.empty: return float("nan")

    all_words = []
    for _, r in valid.iterrows():
        ws = parse_top_words(_topic_keywords_row(r), top_n)
        all_words.extend(ws)

    if not all_words: return 0.0
    return len(set(all_words)) / len(all_words)


# ----------------------------------------------------------------------------
# Entropy
# ----------------------------------------------------------------------------

def compute_topic_entropy(doc_topics_df: pd.DataFrame) -> Tuple[float, float, int, int]:
    if doc_topics_df is None or doc_topics_df.empty:
        return float("nan"), float("nan"), 0, 0

    if "topic" not in doc_topics_df.columns:
        return float("nan"), float("nan"), 0, 0

    df = doc_topics_df[doc_topics_df["topic"] != -1].copy()
    if df.empty:
        return float("nan"), float("nan"), 0, 0

    counts = df["topic"].value_counts()
    n = int(counts.sum())
    if n <= 0: return float("nan"), float("nan"), 0, 0

    probs = counts.values.astype(float) / n
    ent = float(-(probs * np.log(probs + 1e-12)).sum())

    k = len(counts)
    ent_norm = ent / math.log(k) if k > 1 else 0.0
    return ent, ent_norm, k, n


# ----------------------------------------------------------------------------
# Coherence (Gensim)
# ----------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", re.I)

def _prepare_gensim_corpus(texts: Dict[str, str], min_df: int=2, max_df: float=0.95):
    try:
        from gensim.corpora import Dictionary
    except ImportError:
        return None, None

    # Tokenize
    tokenized = []
    for t in texts.values():
        if not t.strip(): continue
        tokens = _TOKEN_RE.findall(t.lower())
        if tokens: tokenized.append(tokens)

    if not tokenized: return None, None

    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=min_df, no_above=max_df)
    dictionary.compactify()

    if len(dictionary) == 0: return None, None

    return tokenized, dictionary

def compute_gensim_coherence_both(topics: pd.DataFrame, texts: Dict[str, str], top_n: int, min_df: int=2, max_df: float=0.95):
    try:
        from gensim.models import CoherenceModel
    except ImportError:
        print("   (Coherence) Skipping: gensim not installed.")
        return float("nan"), float("nan"), 0

    tokenized, dictionary = _prepare_gensim_corpus(texts, min_df, max_df)
    if not dictionary:
        return float("nan"), float("nan"), 0

    # Build topic word lists
    topic_words = []
    for _, r in topics.iterrows():
        if r.get("topic", -1) == -1: continue

        raw_kws = parse_top_words(_topic_keywords_row(r), top_n)

        # FIX 1: Explicitly tokenize phrase-keywords into unigrams.
        expanded_kws = []
        for kw in raw_kws:
            sub_tokens = _TOKEN_RE.findall(kw.lower())
            expanded_kws.extend(sub_tokens)

        # Filter to vocab and dedup preserving order
        valid = []
        seen = set()
        for w in expanded_kws:
            if w in dictionary.token2id and w not in seen:
                valid.append(w)
                seen.add(w)

        # FIX 2: STRICT REQUIREMENT FOR NPMI.
        # NPMI requires pairs. A topic with only 1 word cannot have a score.
        # We enforce at least 2 valid words to prevent "Mean of empty slice" division by zero.
        if len(valid) >= 2:
            topic_words.append(valid)

    if not topic_words:
        print("   (Coherence) WARNING: No valid topic keywords (size >= 2) found. Scores will be NaN.")
        return float("nan"), float("nan"), 0

    try:
        cm_cv = CoherenceModel(topics=topic_words, texts=tokenized, dictionary=dictionary, coherence="c_v", window_size=20)
        score_cv = cm_cv.get_coherence()
    except: score_cv = float("nan")

    try:
        cm_npmi = CoherenceModel(topics=topic_words, texts=tokenized, dictionary=dictionary, coherence="c_npmi", window_size=20)
        score_npmi = cm_npmi.get_coherence()
    except: score_npmi = float("nan")

    return score_cv, score_npmi, len(topics) - len(topic_words)


# ----------------------------------------------------------------------------
# Metrics: Silhouette & Certainty
# ----------------------------------------------------------------------------

def silhouette(doc_topics: pd.DataFrame, emb: np.ndarray, doc_ids: List[str], sample: int=2000, per_topic_max: int=200, seed: int=42) -> float:
    from sklearn.metrics import silhouette_score

    if emb is None or not doc_ids: return float("nan")

    # Align
    d2i = {d: i for i, d in enumerate(doc_ids)}
    valid = doc_topics[doc_topics["topic"] != -1].copy()
    valid["emb_idx"] = valid["doc_id"].map(d2i)
    valid = valid.dropna(subset=["emb_idx"])

    if valid.empty: return float("nan")

    # Stratified downsample
    subsets = []
    for _, grp in valid.groupby("topic"):
        if len(grp) > per_topic_max:
            subsets.append(grp.sample(n=per_topic_max, random_state=seed))
        else:
            subsets.append(grp)

    sample_df = pd.concat(subsets)
    if len(sample_df) > sample:
        sample_df = sample_df.sample(n=sample, random_state=seed)

    labels = sample_df["topic"].values
    if len(set(labels)) < 2: return float("nan")

    X = emb[sample_df["emb_idx"].astype(int).values]
    try:
        return float(silhouette_score(X, labels, metric="cosine"))
    except:
        return float("nan")

def certainty_proxy(doc_topics: pd.DataFrame, emb: np.ndarray, doc_ids: List[str], sample: int=2000, seed: int=42) -> float:
    # Simplified Certainty: Mean distance to centroid of assigned topic
    # (Lower distance = higher certainty)
    if emb is None or not doc_ids: return float("nan")

    d2i = {d: i for i, d in enumerate(doc_ids)}
    valid = doc_topics[doc_topics["topic"] != -1].copy()
    valid["emb_idx"] = valid["doc_id"].map(d2i).dropna().astype(int)

    if valid.empty: return float("nan")

    # Compute centroids
    centroids = {}
    for tid, grp in valid.groupby("topic"):
        idxs = grp["emb_idx"].values
        # Only computing for reasonable topics
        if len(idxs) < 3: continue
        vecs = emb[idxs]
        cent = vecs.mean(axis=0)
        cent = cent / (np.linalg.norm(cent) + 1e-12)
        centroids[tid] = cent

    if not centroids: return float("nan")

    # Sample docs for eval
    if len(valid) > sample:
        eval_df = valid.sample(n=sample, random_state=seed)
    else:
        eval_df = valid

    # Calculate cosine sim to own centroid
    sims = []
    for _, row in eval_df.iterrows():
        tid = row["topic"]
        if tid not in centroids: continue

        idx = int(row["emb_idx"])
        vec = emb[idx]
        vec = vec / (np.linalg.norm(vec) + 1e-12)

        sim = float(np.dot(vec, centroids[tid]))
        sims.append(sim)

    if not sims: return float("nan")
    return float(np.mean(sims))


# ----------------------------------------------------------------------------
# Main Evaluate Logic
# ----------------------------------------------------------------------------

def evaluate_run(
    run_dir: Path,
    out_dir: Optional[Path] = None,
    top_n: int = 10,
    min_df: int = 2,
    seed: int = 42,
    preproc_csv: Optional[Path] = None,
    embed_model: Optional[str] = None,
    embed_batch_size: int = 256,
    no_auto_embeddings: bool = False,
    quiet: bool = False
) -> Dict[str, Any]:

    if not quiet: print(f"Evaluating: {run_dir}")

    dt = _load_doc_topics(run_dir)
    topics = _load_topics(run_dir)
    preproc = _load_preproc_df(run_dir, preproc_csv)
    texts = _load_texts(run_dir, preproc_csv)
    emb, doc_ids = _load_embeddings(run_dir, preproc, embed_model, embed_batch_size, not no_auto_embeddings)

    # 1. Basic Counts
    n_docs = len(dt)
    n_out = (dt["topic"] == -1).sum() if not dt.empty else 0
    n_topics = dt[dt["topic"] != -1]["topic"].nunique() if not dt.empty else 0
    cov = 1.0 - (n_out / n_docs) if n_docs else 0.0

    if not quiet: print(f" Docs: {n_docs}, Topics: {n_topics}, Coverage: {cov:.3f}")

    # 2. Diversity
    div = topic_diversity(topics, top_n)

    # 3. Coherence
    cv, cnpmi, skip = compute_gensim_coherence_both(topics, texts, top_n, min_df=min_df)
    if not quiet: print(f" Coherence: c_v={cv:.3f}, c_npmi={cnpmi:.3f}")

    # 4. Silhouette & Certainty
    sil = silhouette(dt, emb, doc_ids, seed=seed)
    cert = certainty_proxy(dt, emb, doc_ids, seed=seed)

    # 5. Trend & Shift (Load external CSVs)
    trends_csv = run_dir / "10_topic_trends.csv"
    emerging_count = 0
    if trends_csv.exists():
        try:
            tdf = pd.read_csv(trends_csv)
            if "mk_trend" in tdf.columns:
                emerging_count = int(tdf[tdf["mk_trend"] == "increasing"].shape[0])
        except: pass

    shift_csv = run_dir / "11_semantic_shift.csv"
    avg_drift = float("nan")
    if shift_csv.exists():
        try:
            sdf = pd.read_csv(shift_csv)
            if "drift_robust_score" in sdf.columns:
                avg_drift = float(sdf["drift_robust_score"].mean())
        except: pass

    # 6. Entropy
    ent, ent_norm, _, _ = compute_topic_entropy(dt)

    # 7. Performance Metrics (Load from 14_run_metrics.json)
    wall_clock = float("nan")
    llm_calls = 0
    metrics_path = run_dir / "14_run_metrics.json"
    if metrics_path.exists():
        try:
            rm = json.loads(metrics_path.read_text(encoding="utf-8"))
            wall_clock = float(rm.get("wall_clock_seconds", float("nan")))
            llm_calls = int(rm.get("llm_calls", 0))
        except Exception:
            pass

    results = {
        "run_dir": str(run_dir),
        "docs": n_docs,
        "topics": n_topics,
        "coverage": cov,
        "outliers": int(n_out),
        "diversity": div,
        "coherence_cv": cv,
        "coherence_npmi": cnpmi,
        "silhouette": sil,
        "certainty": cert,
        "entropy": ent,
        "entropy_norm": ent_norm,
        "emerging_topics_count": emerging_count,
        "avg_semantic_drift": avg_drift,
        "wall_clock_seconds": wall_clock,
        "llm_calls": llm_calls
    }

    # Save
    _write_json(run_dir / "14_topic_model_evaluation.json", results)
    if out_dir:
        _mk_dir(out_dir)
        _write_json(out_dir / "14_topic_model_evaluation.json", results)

    return results


# ----------------------------------------------------------------------------
# Comparison Mode
# ----------------------------------------------------------------------------

def _load_assignments_for_ami(run_dir: Path) -> pd.Series:
    """Load doc->topic assignments for AMI stability (final doc topics preferred)."""
    dt = _load_doc_topics(run_dir)
    if dt is None or dt.empty:
        raise FileNotFoundError(f"No doc-topics file found under {run_dir}")
    if "doc_id" not in dt.columns:
        raise ValueError(f"doc_id column missing in doc-topics under {run_dir}")
    if "topic" not in dt.columns:
        raise ValueError(f"topic column missing in doc-topics under {run_dir}")
    ser = dt[["doc_id", "topic"]].copy()
    ser["doc_id"] = ser["doc_id"].astype(str)
    ser["topic"] = ser["topic"].astype(int)
    return ser.drop_duplicates("doc_id").set_index("doc_id")["topic"]


def _pairwise_ami_stats(run_dirs: List[Path]) -> Dict[str, Any]:
    """Compute pairwise Adjusted Mutual Information (AMI) across multiple seed runs."""
    from sklearn.metrics import adjusted_mutual_info_score

    if len(run_dirs) < 2:
        return {
            "ami_pairs_n": 0,
            "ami_all_mean": float("nan"),
            "ami_all_std": float("nan"),
            "ami_assigned_mean": float("nan"),
            "ami_assigned_std": float("nan"),
            "ami_doc_overlap_min": 0,
        }

    assigns: List[pd.Series] = []
    for rd in run_dirs:
        try:
            assigns.append(_load_assignments_for_ami(rd))
        except Exception as e:
            print(f"WARNING: Skipping AMI for {rd} due to error: {e}")

    ami_all: List[float] = []
    ami_assigned: List[float] = []
    overlaps: List[int] = []

    if len(assigns) < 2:
         return {
            "ami_pairs_n": 0,
            "ami_all_mean": float("nan"),
            "ami_all_std": float("nan"),
            "ami_assigned_mean": float("nan"),
            "ami_assigned_std": float("nan"),
            "ami_doc_overlap_min": 0,
        }

    for i in range(len(assigns)):
        for j in range(i + 1, len(assigns)):
            a = assigns[i]
            b = assigns[j]
            common = a.index.intersection(b.index)
            overlaps.append(int(len(common)))
            if len(common) < 5:
                continue

            la = a.loc[common].to_numpy(dtype=int)
            lb = b.loc[common].to_numpy(dtype=int)

            # AMI including outliers (-1 is treated as its own cluster)
            ami_all.append(float(adjusted_mutual_info_score(la, lb)))

            # AMI restricted to docs assigned to a non-outlier topic in BOTH runs
            mask = (la != -1) & (lb != -1)
            if int(mask.sum()) >= 5:
                ami_assigned.append(float(adjusted_mutual_info_score(la[mask], lb[mask])))

    def _mean_std(vals: List[float]) -> Tuple[float, float]:
        if not vals:
            return float("nan"), float("nan")
        v = np.asarray(vals, dtype=float)
        return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

    all_mean, all_std = _mean_std(ami_all)
    asg_mean, asg_std = _mean_std(ami_assigned)
    return {
        "ami_pairs_n": int(len(ami_all)),
        "ami_all_mean": all_mean,
        "ami_all_std": all_std,
        "ami_assigned_mean": asg_mean,
        "ami_assigned_std": asg_std,
        "ami_doc_overlap_min": int(min(overlaps)) if overlaps else 0,
    }


def compare_runs(
    out_dir: Path,
    runs: Dict[str, List[Path]],
    preproc_csv: Optional[Path],
    embed_model: Optional[str],
    embed_batch_size: int,
    coh_min_docs: int,
    coh_max_topics: int,
    coh_topk_words: int,
    coh_sample_docs: int,
    prefer_recent: bool,
    disable_embeddings_autofill: bool,
) -> None:
    """Compare multiple models; each model may have multiple seed runs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for label, run_dirs in runs.items():
        run_dirs = [Path(p).expanduser().resolve() for p in run_dirs]
        run_results: List[Dict[str, Any]] = []

        # Evaluate each run directory (per-seed)
        for rd in run_dirs:
            try:
                # Aligned arguments with evaluate_run signature
                res = evaluate_run(
                    run_dir=rd,
                    preproc_csv=preproc_csv,
                    embed_model=embed_model,
                    embed_batch_size=embed_batch_size,
                    top_n=coh_topk_words,
                    min_df=coh_min_docs,
                    no_auto_embeddings=disable_embeddings_autofill,
                )
            except Exception as e:
                print(f"WARNING: evaluation failed for {label} @ {rd}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Attach run_metrics if available
            metrics_path = rd / "14_run_metrics.json"
            if metrics_path.exists():
                try:
                    rm = json.loads(metrics_path.read_text(encoding="utf-8"))
                    res["wall_clock_seconds"] = float(rm.get("wall_clock_seconds", float("nan")))
                    res["llm_calls"] = int(rm.get("llm_calls", 0))
                except Exception:
                    pass

            run_results.append(res)

        if not run_results:
            continue

        df = pd.DataFrame(run_results)

        # Aggregate numeric metrics
        agg: Dict[str, Any] = {
            "label": label,
            "n_runs": int(len(run_results)),
            "run_dirs": ",".join([str(p) for p in run_dirs]),
        }

        for col in df.columns:
            if col in ("run_dir", "label", "run_dirs"):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                vals = pd.to_numeric(df[col], errors="coerce")
                agg[f"{col}_mean"] = float(vals.mean())
                agg[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        # Stability across seeds (AMI)
        try:
            ami_stats = _pairwise_ami_stats(run_dirs)
            agg.update(ami_stats)
        except Exception as e:
            print(f"WARNING: AMI computation failed for {label}: {e}")

        rows.append(agg)

    if not rows:
        print("No valid runs found to compare. Exiting.")
        return

    out = pd.DataFrame(rows)
    out_path = out_dir / "17_ablation_summary.csv"
    out.to_csv(out_path, index=False)

    # Pretty table: keep a focused set of columns when available
    prefer_cols = [
        "label", "n_runs",
        "topics_mean", "coverage_mean", "outliers_mean", "diversity_mean",
        "coherence_npmi_mean", "coherence_cv_mean",
        "ami_all_mean", "ami_assigned_mean",
        "wall_clock_seconds_mean", "llm_calls_mean",
    ]
    # Check what columns actually exist
    show_cols = [c for c in prefer_cols if c in out.columns]

    # Fallback to all columns if prefer_cols are missing
    if not show_cols:
        show_cols = out.columns.tolist()

    if show_cols:
        md = out[show_cols].sort_values("label").to_markdown(index=False)
    else:
        md = out.sort_values("label").to_markdown(index=False)

    print("\n=== Compare Summary (mean across seeds; std columns in CSV) ===")
    print(md)
    print(f"\nWrote: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", help="Single run directory to evaluate")
    ap.add_argument("--out_dir", help="Output directory for comparison")
    ap.add_argument("--runs", nargs="+", help="label=path items for comparison")

    # Config
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--min_df", type=int, default=2, help="Min DF for coherence dictionary (set low for niche topics)")
    ap.add_argument("--embed_model", default="local_bge-large-en-v1.5")
    ap.add_argument("--preproc_csv", help="Path to shared preprocessed documents")
    ap.add_argument("--no_auto_embeddings", action="store_true")
    # Added seed argument here
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling-based metrics (silhouette/certainty)")

    args = ap.parse_args()

    if args.runs and args.out_dir:
        # Compare Mode
        rdict = {}
        for x in args.runs:
            k,v = x.split("=", 1)
            # FIXED: Append to list instead of overwriting
            if k not in rdict:
                rdict[k] = []
            rdict[k].append(Path(v))

        compare_runs(
            Path(args.out_dir), rdict,
            # Fix: Map arguments to the names expected by compare_runs definition
            coh_topk_words=args.top_n,
            # Pass defaults for params not exposed in CLI but required by definition
            coh_min_docs=args.min_df,
            coh_max_topics=100,
            coh_sample_docs=2000,
            embed_batch_size=256,
            prefer_recent=False,
            preproc_csv=Path(args.preproc_csv) if args.preproc_csv else None,
            embed_model=args.embed_model,
            disable_embeddings_autofill=args.no_auto_embeddings
        )
    elif args.run_dir:
        evaluate_run(
            Path(args.run_dir),
            top_n=args.top_n,
            min_df=args.min_df,
            seed=args.seed,
            preproc_csv=Path(args.preproc_csv) if args.preproc_csv else None,
            embed_model=args.embed_model,
            no_auto_embeddings=args.no_auto_embeddings
        )
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
