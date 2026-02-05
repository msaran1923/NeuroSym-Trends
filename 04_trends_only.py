#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_trends_only.py

Post-run trend analysis for ANY run_dir that contains doc_topics + topics metadata.
This is designed to make bare BERTopic and pure-LLM baselines comparable on the same trend metrics used in the main pipeline:

  - Stage 10: Topic Trends -> 10_topic_trends.csv + 10_emerging_topics.csv
  - Stage 11: Semantic Shift -> 11_semantic_shift.csv (only if embeddings are available)
  - Stage 13: Prevalence Baseline -> 13_prevalence_baseline.csv
  - Stage 13: Model Stability -> 13_model_stability.csv (BERTopic-based; meaningful for BERTopic runs)

It reuses the exact stage implementations from the pipeline file via dynamic import.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


DOC_TOPICS_PRIORITY = [
    "09_final_doc_topics.csv",
    "09_final_doc_topics_enriched.csv",
    "08_augmented_doc_topics.csv",
    "07_llm_gated_doc_topics.csv",
    "06_reassigned_doc_topics.csv",
    "04_bertopic_doc_topics.csv",
]

TOPICS_PRIORITY = [
    "09_final_topics.csv",
    "08_new_topics_metadata.csv",
    "04_bertopic_topics.csv",
]


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        # Try common encodings
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
            try:
                return pd.read_csv(p, encoding=enc)
            except Exception:
                pass
    raise RuntimeError(f"Failed to read CSV: {p}")


def _normalize_topic_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "topic" not in df.columns and "topic_id" in df.columns:
        df = df.rename(columns={"topic_id": "topic"})
    if "doc_id" not in df.columns:
        for c in ("document_id", "id", "docid"):
            if c in df.columns:
                df = df.rename(columns={c: "doc_id"})
                break
    return df


def load_doc_topics(run_dir: Path) -> Tuple[pd.DataFrame, str]:
    for name in DOC_TOPICS_PRIORITY:
        p = run_dir / name
        if p.exists():
            df = _safe_read_csv(p)
            df = _normalize_topic_cols(df)
            if "doc_id" in df.columns and "topic" in df.columns:
                df["doc_id"] = df["doc_id"].astype(str)
                return df, name
    raise FileNotFoundError(f"No doc_topics found under {run_dir}. Expected one of: {DOC_TOPICS_PRIORITY}")


def load_topics(run_dir: Path) -> Tuple[pd.DataFrame, str]:
    for name in TOPICS_PRIORITY:
        p = run_dir / name
        if p.exists():
            df = _safe_read_csv(p)
            df = _normalize_topic_cols(df)
            if "topic" in df.columns:
                return df, name
    return pd.DataFrame(), ""


def load_preproc(preproc_csv: Path) -> pd.DataFrame:
    df = _safe_read_csv(preproc_csv)
    if "doc_id" not in df.columns:
        raise ValueError(f"Preprocessed CSV must have doc_id: {preproc_csv}")
    df["doc_id"] = df["doc_id"].astype(str)
    # Normalize year
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df[df["year"].notna()].copy()
        df["year"] = df["year"].astype(int)

    # Ensure text column exists
    if "text" not in df.columns:
        t = df["english_title"].astype(str) if "english_title" in df.columns else ""
        a = df["abstract_en"].astype(str) if "abstract_en" in df.columns else ""
        if "english_title" not in df.columns:
             # Try clean columns
             t = df["english_title_clean"].astype(str) if "english_title_clean" in df.columns else ""
             a = df["abstract_en_clean"].astype(str) if "abstract_en_clean" in df.columns else ""
        df["text"] = (t + "\n" + a).str.strip()

    return df


def _compute_embeddings_sentence_transformers(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is required to auto-compute embeddings. "
            f"Install it or disable with --no_auto_embeddings. Import error: {e}"
        )

    model = SentenceTransformer(model_name)
    try:
        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except TypeError:
        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    return np.asarray(emb)


def load_embeddings(
    run_dir: Path,
    preproc: pd.DataFrame,
    embed_model: str,
    batch_size: int,
    auto_compute: bool,
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:

    # 1. Try loading existing
    for name in ["embeddings.npy", "02_embeddings.npy"]:
        p = run_dir / name
        if not p.exists(): continue

        try:
            emb = np.load(p)
        except Exception as e:
            print(f"WARNING: Failed to load {name}: {e}")
            continue

        doc_ids: Optional[List[str]] = None
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
                except Exception as e:
                    print(f"WARNING: Failed to read {sc.name}: {e}")

        # If aligned by size, assume coherence
        if emb is not None and (not doc_ids) and preproc is not None and len(preproc) == int(getattr(emb, "shape", [0])[0]):
            try:
                doc_ids = preproc["doc_id"].astype(str).tolist()
                # Auto-heal sidecar
                (run_dir / "embeddings_doc_ids.json").write_text(json.dumps(doc_ids, ensure_ascii=False), encoding="utf-8")
            except Exception: pass

        return emb, doc_ids

    # 2. Auto-compute
    if not auto_compute:
        return None, None

    if preproc is None or preproc.empty:
        print("WARNING: Cannot auto-compute embeddings: preproc is empty.")
        return None, None

    if embed_model.lower() == "none" or not embed_model.strip():
        return None, None

    try:
        texts = preproc["text"].astype(str).tolist()
        doc_ids = preproc["doc_id"].astype(str).tolist()
        print(f"[trends] [auto] Computing embeddings for {len(doc_ids)} docs using: {embed_model}")
        emb = _compute_embeddings_sentence_transformers(texts, embed_model, int(batch_size)).astype(np.float32)

        # Save
        np.save(run_dir / "embeddings.npy", emb)
        (run_dir / "embeddings_doc_ids.json").write_text(json.dumps(doc_ids, ensure_ascii=False), encoding="utf-8")
        return emb, doc_ids
    except Exception as e:
        print(f"WARNING: Auto-embedding failed: {e}")
        return None, None


def make_doc_id_to_emb_idx(preproc: pd.DataFrame, emb: np.ndarray, emb_doc_ids: Optional[List[str]]) -> Optional[Dict[str, int]]:
    if emb is None:
        return None
    if emb_doc_ids:
        return {str(d): i for i, d in enumerate(emb_doc_ids)}
    # Fallback: only if shapes line up with preproc row order
    if len(preproc) == int(emb.shape[0]):
        return {str(d): i for i, d in enumerate(preproc["doc_id"].tolist())}
    return None


def import_pipeline(pipeline_py: Path):
    import importlib.util
    # Unique name to prevent caching issues across runs
    mod_name = f"pipeline_mod_{int(time.time()*1000)}"
    spec = importlib.util.spec_from_file_location(mod_name, str(pipeline_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import pipeline module from {pipeline_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def ensure_year_and_type(doc_topics: pd.DataFrame, preproc: pd.DataFrame) -> pd.DataFrame:
    dt = doc_topics.copy()
    dt["doc_id"] = dt["doc_id"].astype(str)

    # Merge missing metadata
    cols_to_add = []
    if "year" not in dt.columns and "year" in preproc.columns:
        cols_to_add.append("year")
    if "thesis_type" not in dt.columns and "thesis_type" in preproc.columns:
        cols_to_add.append("thesis_type")

    if cols_to_add:
        # Preproc lookup
        lookup = preproc[["doc_id"] + cols_to_add].drop_duplicates("doc_id").set_index("doc_id")
        dt = dt.join(lookup, on="doc_id", how="left")

    if "year" in dt.columns:
        dt["year"] = pd.to_numeric(dt["year"], errors="coerce")
        dt = dt[dt["year"].notna()].copy()
        dt["year"] = dt["year"].astype(int)

    if "topic" in dt.columns:
        dt["topic"] = pd.to_numeric(dt["topic"], errors="coerce").fillna(-1).astype(int)

    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Run directory to write 10/11/13 outputs into")
    ap.add_argument("--pipeline_py", default="", help="Path to 01_thesis_topic_trends_pipeline.py. If omitted, defaults to a sibling of this script.")
    ap.add_argument("--preproc_csv", default="", help="Path to 01_preprocessed_documents.csv. If omitted, defaults to <run_dir>/01_preprocessed_documents.csv.")

    ap.add_argument("--trend_alpha", type=float, default=0.05)

    ap.add_argument("--mk_method", choices=["classic", "tfpw", "hamed_rao"], default="classic")
    ap.add_argument("--mk_lag_max", type=int, default=10)
    ap.add_argument("--prevalence_mode", choices=["relative", "absolute"], default="relative")
    ap.add_argument("--trend_min_topic_size", type=int, default=15)

    ap.add_argument("--emerging_if_burst", action="store_true", help="Mark emerging if burst detected, even if mk not significant")
    ap.add_argument("--min_year_docs", type=int, default=10)

    ap.add_argument("--min_docs_shift", type=int, default=5)

    # Stability (optional; BERTopic-only meaningful)
    ap.add_argument("--stability_runs", type=int, default=0)

    # Needed if stability_runs > 0 (mirrors pipeline args)
    ap.add_argument("--embed_model", default="local_bge-large-en-v1.5")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--embed_batch_size", type=int, default=None, help="Alias for --batch_size (backward compatible).")

    ap.add_argument("--no_auto_embeddings", action="store_true",
                help="Disable auto-computation of embeddings when embeddings.npy is missing (will be skipped).")
    ap.add_argument("--umap_neighbors", type=int, default=15)
    ap.add_argument("--umap_components", type=int, default=5)
    ap.add_argument("--umap_min_dist", type=float, default=0.0)
    ap.add_argument("--min_cluster_size", type=int, default=30)
    ap.add_argument("--min_samples", type=int, default=10)
    ap.add_argument("--extra_stopwords", default="")
    ap.add_argument("--reduce_topics", default="none")

    # --- Fix: Support min_df for stability check ---
    ap.add_argument("--vectorizer_min_df", type=int, default=5, help="Min DF for stability runs.")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Backward-compatible alias
    if getattr(args, 'embed_batch_size', None) is not None:
        args.batch_size = int(args.embed_batch_size)

    pipeline_py = Path(args.pipeline_py).expanduser() if str(args.pipeline_py).strip() else (Path(__file__).parent / '01_thesis_topic_trends_pipeline.py')
    preproc_csv = Path(args.preproc_csv).expanduser() if str(args.preproc_csv).strip() else (run_dir / '01_preprocessed_documents.csv')

    if not pipeline_py.exists():
        raise FileNotFoundError(f"pipeline_py not found: {pipeline_py}")
    if not preproc_csv.exists():
        raise FileNotFoundError(f"preproc_csv not found: {preproc_csv}")

    mod = import_pipeline(pipeline_py)

    # Load inputs
    preproc = load_preproc(preproc_csv)
    dt, dt_name = load_doc_topics(run_dir)
    topics, topics_name = load_topics(run_dir)

    print(f"[trends] run_dir={run_dir}")
    print(f"[trends] using doc_topics={dt_name}")
    print(f"[trends] using topics={topics_name or '(none)'}")
    print(f"[trends] preproc={preproc_csv}")

    dt = ensure_year_and_type(dt, preproc)

    # -------------------------
    # Stage: trends
    # -------------------------
    f_trends = run_dir / "10_topic_trends.csv"
    f_emerging = run_dir / "10_emerging_topics.csv"
    print("[trends] Topic Trends")
    try:
        mod.stage08_trends_topics(
            doc_topics=dt,
            topics_df=topics,
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
    except Exception as e:
        print(f"❌ Trends failed: {e}")

    # -------------------------
    # Stage: semantic shift
    # -------------------------
    emb, emb_doc_ids = load_embeddings(run_dir, preproc, str(args.embed_model), int(args.batch_size), auto_compute=(not bool(args.no_auto_embeddings)))
    doc_id_to_emb_idx = make_doc_id_to_emb_idx(preproc, emb, emb_doc_ids) if emb is not None else None
    f_shift = run_dir / "11_semantic_shift.csv"

    if emb is None or doc_id_to_emb_idx is None:
        print("[trends] Stage: Semantic Shift (SKIP) - embeddings missing or mapping unavailable.")
        pd.DataFrame().to_csv(f_shift, index=False)
    else:
        print("[trends] Stage: Semantic Shift")
        try:
            mod.stage09_semantic_shift(dt, emb, doc_id_to_emb_idx, topics, f_shift, min_docs_per_year=int(args.min_docs_shift))
        except Exception as e:
            print(f"❌ Semantic shift failed: {e}")

    # -------------------------
    # Stage: prevalence baseline
    # -------------------------
    print("[trends] Stage: Prevalence Baseline")
    try:
        mod.stage11a_prevalence_baseline(dt, run_dir / "13_prevalence_baseline.csv")
    except Exception as e:
        print(f"❌ Prevalence baseline failed: {e}")

    # -------------------------
    # Stage: model stability
    # -------------------------
    if int(args.stability_runs) > 0:
        print(f"[trends] Stage: Model Stability (runs={args.stability_runs})")
        emb_path = run_dir / "embeddings.npy"
        if not emb_path.exists():
            print("[trends]  - SKIP stability: embeddings.npy missing in run_dir.")
        else:
            # Robust call: handle updated signature with vectorizer_min_df if present
            try:
                mod.stage11b_model_stability(
                    preproc, run_dir,
                    args.embed_model, emb_path,
                    int(args.stability_runs),
                    int(args.batch_size),
                    int(args.umap_neighbors),
                    int(args.umap_components),
                    float(args.umap_min_dist),
                    int(args.min_cluster_size),
                    int(args.min_samples),
                    str(args.extra_stopwords),
                    str(args.reduce_topics),
                    run_dir / "13_model_stability.csv",
                    # Pass the new arg if the function supports it
                    vectorizer_min_df=int(args.vectorizer_min_df)
                )
            except TypeError:
                print("[trends] ⚠️  Warning: stage11b_model_stability does not accept vectorizer_min_df. Using default.")
                mod.stage11b_model_stability(
                    preproc, run_dir,
                    args.embed_model, emb_path,
                    int(args.stability_runs),
                    int(args.batch_size),
                    int(args.umap_neighbors),
                    int(args.umap_components),
                    float(args.umap_min_dist),
                    int(args.min_cluster_size),
                    int(args.min_samples),
                    str(args.extra_stopwords),
                    str(args.reduce_topics),
                    run_dir / "13_model_stability.csv",
                )

    print("[trends] ✅ Done.")


if __name__ == "__main__":
    main()
