#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_bare_bertopic_baseline.py

"Fair" BERTopic-only baseline for neuro-symbolic pipeline.
This script imports the main pipeline to reuse preprocessing and evaluation functions but runs standard BERTopic logic for the modeling stage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import datetime
import inspect

try:
    import torch  
except Exception: 
    torch = None  
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

# -----------------------------------------------------------------------------
# Module Loader
# -----------------------------------------------------------------------------
def _load_pipeline_module(pipeline_path: Path):
    """Dynamically import the pipeline module so we can reuse its stages."""
    import importlib.util

    pipeline_path = Path(pipeline_path).resolve()
    # Create a unique module name to avoid cache collisions
    module_name = f"_pipeline_for_baseline_{pipeline_path.stem}_{int(time.time()*1e6)}"

    spec = importlib.util.spec_from_file_location(module_name, str(pipeline_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for: {pipeline_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    # We remove 'conflict_handler' to ensure strict parsing (no duplicates allowed)
    ap = argparse.ArgumentParser(description="Bare BERTopic baseline (fair finalize).")

    # --- Inputs ---
    ap.add_argument("--data", default="",
                    help="Raw input CSV. Runs pipeline Stage 1 preprocessing if --preproc_csv is missing.")
    ap.add_argument("--preproc_csv", default="",
                    help="Path to an existing 01_preprocessed_documents.csv. If omitted, --data must be provided.")
    ap.add_argument("--run_dir", required=True, help="Output directory for this baseline run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pipeline_py", default="01_thesis_topic_trends_pipeline.py",
                    help="Pipeline script path to import stages from.")

    # --- BERTopic Params ---
    ap.add_argument("--embed_model", default="local_bge-large-en-v1.5")
    ap.add_argument("--batch_size", type=int, default=512)

    # UMAP
    ap.add_argument("--umap_neighbors", type=int, default=15)
    ap.add_argument("--umap_components", type=int, default=5)
    ap.add_argument("--umap_min_dist", type=float, default=0.0)

    # HDBSCAN
    ap.add_argument("--min_cluster_size", type=int, default=30)
    ap.add_argument("--min_samples", type=int, default=10)

    # Topic Reduction & Stopwords
    ap.add_argument("--reduce_topics", default="none",
                    help="Topic reduction mode: none | auto | <int>.")
    ap.add_argument("--extra_stopwords", default="")

    # --- CRITICAL FIX: Vocabulary Control ---
    ap.add_argument("--vectorizer_min_df", type=int, default=5,
                    help="Min document frequency (must match main pipeline for fair comparison).")

    # --- Finalize Params (c-TF-IDF) ---
    ap.add_argument("--ctfidf_topn", type=int, default=15)
    ap.add_argument("--ctfidf_weighting", default="confidence",
                    help="Finalize c-TF-IDF weighting: confidence|none")

    # --- Analysis Flags ---
    ap.add_argument("--run_trends", action="store_true")
    ap.add_argument("--trend_alpha", type=float, default=0.05)
    ap.add_argument("--emerging_if_burst", action="store_true")
    ap.add_argument("--min_year_docs", type=int, default=10)

    # --- Semantic Shift ---
    ap.add_argument("--run_shift", action="store_true")
    ap.add_argument("--min_docs_shift", type=int, default=5,
                    help="Min docs per year (per-topic) required for semantic shift.")

    return ap.parse_args(argv)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _resolve_pipeline_path(p: str) -> Path:
    pipeline_path = Path(p)
    if pipeline_path.exists():
        return pipeline_path
    # Fall back to script dir if relative path fails
    candidate = Path(__file__).parent / Path(p).name
    return candidate
def _ordered_dict_from_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Build a dict from (key, value) pairs preserving first occurrence."""
    out: Dict[str, Any] = {}
    for k, v in pairs:
        if k not in out:
            out[k] = v
    return out


def _call_with_supported_kwargs(fn, kv_pairs: List[Tuple[str, Any]], *, context: str = ""):
    """
    Call `fn` using only the keyword arguments it actually supports (via inspect.signature).
    This keeps the baseline compatible with small signature differences across pipeline versions
    without changing any core logic.
    """
    cand = _ordered_dict_from_pairs(kv_pairs)

    try:
        sig = inspect.signature(fn)
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_varkw:
            return fn(**cand)

        filtered = {k: v for k, v in cand.items() if k in params}

        # If there are positional-only params, kwargs-only calls are unsafe; fall back to a direct call.
        if any(p.kind == inspect.Parameter.POSITIONAL_ONLY for p in params.values()):
            return fn(**filtered)

        # Validate required params
        missing = [
            name for name, p in params.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and p.default is inspect._empty
            and name not in filtered
        ]
        if missing:
            raise TypeError(
                f"{context}: missing required params {missing} for {getattr(fn, '__name__', 'callable')}. "
                f"Provided keys: {sorted(filtered.keys())}"
            )

        return fn(**filtered)
    except TypeError:
        # Let caller decide whether to try additional fallbacks.
        raise
    except Exception:
        # If signature introspection fails for any reason, try the full candidate kwargs.
        return fn(**cand)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    wall_t0 = time.perf_counter()
    run_started_at_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if argv is None:
        argv = sys.argv[1:]
    # Capture the *resolved* argv for reproducible run metadata
    run_cmd = "python " + " ".join([Path(sys.argv[0]).name] + list(argv))
    args = parse_args(argv)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Bare BERTopic Baseline | Run Dir: {run_dir}")

    # 1. Load pipeline module
    pipeline_path = _resolve_pipeline_path(args.pipeline_py)
    if not pipeline_path.exists():
        raise SystemExit(f"ERROR: pipeline script not found: {pipeline_path}")
    pipe = _load_pipeline_module(pipeline_path)

    # 2. Determine Input Data
    preproc: Optional[Path] = None
    data_csv: Optional[Path] = None

    if str(args.preproc_csv).strip():
        preproc = Path(args.preproc_csv)
        if not preproc.exists():
            raise SystemExit(f"ERROR: preproc_csv not found: {preproc}")
        if preproc.is_dir():
            raise SystemExit(f"ERROR: preproc_csv is a directory: {preproc}")
    elif str(args.data).strip():
        data_csv = Path(args.data)
        if not data_csv.exists():
            raise SystemExit(f"ERROR: data CSV not found: {data_csv}")
        if data_csv.is_dir():
            raise SystemExit(f"ERROR: --data is a directory: {data_csv}")

        # Run Stage 1 Preprocessing via pipeline
        if not hasattr(pipe, "stage01_preprocess"):
            raise SystemExit("ERROR: stage01_preprocess not found in pipeline module.")

        # Helper to read CSV
        if hasattr(pipe, "robust_read_csv"):
            df_raw = pipe.robust_read_csv(data_csv)
        else:
            import pandas as pd
            df_raw = pd.read_csv(data_csv)

        preproc = run_dir / "01_preprocessed_documents.csv"
        # Run preprocessing
        df = pipe.stage01_preprocess(df_raw, preproc, max_tr_chars=0)
    else:
        raise SystemExit("ERROR: Provide --preproc_csv (existing) or --data (raw CSV).")

    assert preproc is not None

    # 3. Load Dataframe (if not already loaded from preprocessing step)
    if "df" not in locals():
        if hasattr(pipe, "robust_read_csv"):
            df = pipe.robust_read_csv(preproc)
        else:
            import pandas as pd
            df = pd.read_csv(preproc)

    # Ensure doc_id is string
    if "doc_id" in df.columns:
        df["doc_id"] = df["doc_id"].astype(str)

    # Copy preproc file to run_dir for evaluation script compatibility
    dst_preproc = run_dir / "01_preprocessed_documents.csv"
    if dst_preproc.resolve() != preproc.resolve():
        shutil.copy2(preproc, dst_preproc)

    # 4. Run Stage 3: BERTopic
    f_dt04 = run_dir / "04_bertopic_doc_topics.csv"
    f_topics04 = run_dir / "04_bertopic_topics.csv"
    f_model04 = run_dir / "04_bertopic_model"
    f_emb = run_dir / "embeddings.npy"

    print(">>> Stage 03: BERTopic...")

    if not hasattr(pipe, "stage03_bertopic"):
        raise SystemExit("ERROR: stage03_bertopic not found in pipeline module.")


    # Signature-aware call first (keeps compatibility across small pipeline signature differences)
    stage03_kv = [
        ("df", df),
        ("out_doc_topics_csv", f_dt04),
        ("out_doc_topics_path", f_dt04),
        ("out_topics_csv", f_topics04),
        ("out_topics_path", f_topics04),
        ("model_dir", f_model04),
        ("embeddings_path", f_emb),
        ("embeddings_file", f_emb),
        ("embed_model", args.embed_model),
        ("umap_n_neighbors", args.umap_neighbors),
        ("umap_neighbors", args.umap_neighbors),
        ("umap_n_components", args.umap_components),
        ("umap_components", args.umap_components),
        ("umap_min_dist", args.umap_min_dist),
        ("hdb_min_cluster_size", args.min_cluster_size),
        ("min_cluster_size", args.min_cluster_size),
        ("hdb_min_samples", args.min_samples),
        ("min_samples", args.min_samples),
        ("seed", args.seed),
        ("batch_size", args.batch_size),
        ("extra_stopwords", args.extra_stopwords),
        ("reduce_topics", args.reduce_topics),
        ("vectorizer_min_df", args.vectorizer_min_df),
    ]

    try:
        _call_with_supported_kwargs(pipe.stage03_bertopic, stage03_kv, context="stage03_bertopic")
    except TypeError:
        # Legacy positional fallbacks (kept for older pipeline variants)
        # Try calling stage03 with various signatures to handle pipeline version differences
        # Priority: Newest signature with vectorizer_min_df -> Standard -> Fallbacks
        try:
            pipe.stage03_bertopic(
                df, f_dt04, f_topics04, f_model04, f_emb,
                args.embed_model, args.umap_neighbors, args.umap_components, args.umap_min_dist,
                args.min_cluster_size, args.min_samples,
                args.seed, args.batch_size,
                args.extra_stopwords, args.reduce_topics,
                vectorizer_min_df=args.vectorizer_min_df  # <--- CRITICAL FIX
            )
        except TypeError:
            try:
                # Fallback: old signature (no min_df arg)
                print("⚠️  Warning: Pipeline stage03_bertopic does not accept vectorizer_min_df. Using default.")
                pipe.stage03_bertopic(
                    df, f_dt04, f_topics04, f_model04, f_emb,
                    args.embed_model, args.umap_neighbors, args.umap_components, args.umap_min_dist,
                    args.min_cluster_size, args.min_samples,
                    args.seed, args.batch_size,
                    args.extra_stopwords, args.reduce_topics,
                )
            except TypeError:
                try:
                    # Fallback: no reduce_topics
                    pipe.stage03_bertopic(
                        df, f_dt04, f_topics04, f_model04, f_emb,
                        args.embed_model, args.umap_neighbors, args.umap_components, args.umap_min_dist,
                        args.min_cluster_size, args.min_samples,
                        args.seed, args.batch_size,
                        args.extra_stopwords,
                    )
                except TypeError:
                    # Fallback: no extra_stopwords
                    pipe.stage03_bertopic(
                        df, f_dt04, f_topics04, f_model04, f_emb,
                        args.embed_model, args.umap_neighbors, args.umap_components, args.umap_min_dist,
                        args.min_cluster_size, args.min_samples,
                        args.seed, args.batch_size,
                    )

    # 5. Finalize (Comparison Step)
    print(">>> Finalize (recompute keywords via c-TF-IDF + OUTLIER row) ...")

    finalize_fn = None
    if hasattr(pipe, "stage07_finalize"):
        finalize_fn = pipe.stage07_finalize
    elif hasattr(pipe, "stage09_finalize"):
        finalize_fn = pipe.stage09_finalize
    else:
        raise SystemExit("ERROR: Neither stage07_finalize nor stage09_finalize found in pipeline module.")

    import pandas as pd
    dt = pd.read_csv(f_dt04)
    topics = pd.read_csv(f_topics04)

    # Normalize column names
    for df_ in (dt, topics):
        if "Topic" in df_.columns and "topic" not in df_.columns:
            df_.rename(columns={"Topic": "topic"}, inplace=True)
        if "topic_id" in df_.columns and "topic" not in df_.columns:
            df_.rename(columns={"topic_id": "topic"}, inplace=True)

    if "label" not in topics.columns:
        if "Name" in topics.columns:
            topics["label"] = topics["Name"].astype(str)
        elif "name" in topics.columns:
            topics["label"] = topics["name"].astype(str)
        else:
            topics["label"] = ""

    empty_new = pd.DataFrame(columns=["topic", "label", "keywords", "size", "source"])
    f_dt_final = run_dir / "09_final_doc_topics.csv"
    f_topics_final = run_dir / "09_final_topics.csv"

    # Robust finalize call handling extra_stopwords
    try:
        final_dt, final_topics = finalize_fn(
            dt, topics, empty_new,
            f_dt_final, f_topics_final,
            df_text=df,
            ctfidf_topn=int(args.ctfidf_topn),
            ctfidf_weighting=str(args.ctfidf_weighting),
            extra_stopwords=str(args.extra_stopwords or ""),
        )
    except TypeError:
        # Fallback for older pipeline versions lacking extra_stopwords in finalize
        final_dt, final_topics = finalize_fn(
            dt, topics, empty_new,
            f_dt_final, f_topics_final,
            df_text=df,
            ctfidf_topn=int(args.ctfidf_topn),
            ctfidf_weighting=str(args.ctfidf_weighting),
        )

    # 6. Optional: Trends
    if args.run_trends:
        if not hasattr(pipe, "stage08_trends_topics"):
            raise SystemExit("ERROR: stage08_trends_topics not found in pipeline module.")
        f_trends = run_dir / "10_topic_trends.csv"
        f_emerging = run_dir / "10_emerging_topics.csv"
        pipe.stage08_trends_topics(
            final_dt, final_topics, f_trends, f_emerging,
            float(args.trend_alpha), bool(args.emerging_if_burst), int(args.min_year_docs)
        )

    # 7. Optional: Shift
    if args.run_shift:
        if not hasattr(pipe, "stage09_semantic_shift"):
            raise SystemExit("ERROR: stage09_semantic_shift not found in pipeline module.")

        # Load embeddings (using pipeline helper if available)
        if hasattr(pipe, "load_embeddings_with_doc_ids"):
            emb, emb_doc_ids = pipe.load_embeddings_with_doc_ids(f_emb)
        else:
            import numpy as np
            emb = np.load(f_emb)
            emb_ids_path = (run_dir / "embeddings_doc_ids.json")
            if not emb_ids_path.exists():
                raise SystemExit(
                    f"ERROR: {emb_ids_path} not found. "
                    "Either ensure the pipeline saves doc_id alignment metadata, "
                    "or use pipeline.load_embeddings_with_doc_ids()."
                )
            emb_doc_ids = json.loads(emb_ids_path.read_text(encoding="utf-8"))

        if hasattr(pipe, "make_doc_id_to_emb_idx"):
            doc_id_to_emb_idx = pipe.make_doc_id_to_emb_idx(emb_doc_ids)
        else:
            doc_id_to_emb_idx = {str(d): i for i, d in enumerate(emb_doc_ids)}

        f_shift = run_dir / "11_semantic_shift.csv"
        # Signature-aware call first
        stage09_kv = [
            ("doc_topics", final_dt),
            ("embeddings", emb),
            ("doc_id_to_emb_idx", doc_id_to_emb_idx),
            ("topics_df", final_topics),
            ("out_csv", f_shift),
            ("min_docs_per_year", int(getattr(args, "min_docs_shift", 5))),
        ]
        try:
            _call_with_supported_kwargs(pipe.stage09_semantic_shift, stage09_kv, context="stage09_semantic_shift")
        except TypeError:
            # Legacy positional fallback
            try:
                pipe.stage09_semantic_shift(
                    final_dt, emb, doc_id_to_emb_idx, final_topics, f_shift,
                    min_docs_per_year=int(getattr(args, "min_docs_shift", 5))
                )
            except TypeError:
                pipe.stage09_semantic_shift(final_dt, emb, doc_id_to_emb_idx, final_topics, f_shift)

    wall_clock_sec = float(time.perf_counter() - wall_t0)
    run_metrics = {
        "run_started_at_utc": run_started_at_utc,
        "command": run_cmd,
        "run_dir": str(run_dir),
        "wall_clock_seconds": wall_clock_sec,
        "llm_calls": 0,
        "device": ("cuda" if (torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"),
        "seed": int(getattr(args, "seed", 0)),
    }
    try:
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            run_metrics["cuda_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        (run_dir / "14_run_metrics.json").write_text(json.dumps(run_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"WARNING: failed to write 14_run_metrics.json: {e}")

    print("✅ BERTopic baseline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
