#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_sensitivity_sweep.py

Fast sensitivity sweep for Stage 4a (similarity+margin outlier reassignment),
replayed on the pre-reassignment artifacts written by the pipeline:

  - 04_bertopic_doc_topics.csv   (must include: doc_id, topic)
  - embeddings.npy               (document embeddings aligned to doc_id order)

This script DOES NOT re-run BERTopic or later stages. It estimates how changing
(tau_sim, tau_mar) would change:
  - coverage (assigned fraction)
  - number of reassigned outliers
  - quality proxies for reassignment (best similarity / margin stats)

Output:
  - 05_sensitivity_results_stage4a.csv
"""

import argparse
import sys
from pathlib import Path
from itertools import product
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


def _parse_float_list(s: str) -> List[float]:
    if s is None:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [float(p) for p in parts]


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def compute_centroids_cosine(
    df_dt: pd.DataFrame,
    emb: np.ndarray,
    topic_col: str = "topic",
    min_docs_per_topic: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      topic_ids: shape (T,)
      centroids: shape (T, D) - L2 normalized
    """
    if topic_col not in df_dt.columns:
        raise ValueError(f"doc_topics missing column '{topic_col}'")

    topics = df_dt[topic_col].to_numpy()
    if emb.shape[0] != len(df_dt):
        raise ValueError(f"embeddings rows ({emb.shape[0]}) != doc_topics rows ({len(df_dt)}).")

    # We assume emb rows correspond to df_dt row order (pipeline writes embeddings with dt order).
    emb_n = _l2_normalize_rows(emb.astype(np.float32))

    topic_ids = np.array(sorted([t for t in np.unique(topics) if int(t) != -1]), dtype=int)
    centroids = []
    kept_ids = []

    for t in topic_ids:
        idx = np.where(topics == t)[0]
        if idx.size < min_docs_per_topic:
            continue
        c = emb_n[idx].mean(axis=0)
        c = c / max(1e-12, float(np.linalg.norm(c)))
        centroids.append(c)
        kept_ids.append(int(t))

    if not centroids:
        raise RuntimeError("No non-outlier topics found to build centroids.")

    return np.array(kept_ids, dtype=int), np.vstack(centroids).astype(np.float32)


def best_and_second_best(sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given sim: (N, T) similarities, return:
      best_sim (N,)
      second_sim (N,)
      best_idx (N,) index into columns (0..T-1)
    """
    if sim.shape[1] < 2:
        best_idx = np.zeros(sim.shape[0], dtype=int)
        best_sim = sim[:, 0].copy()
        second_sim = np.full(sim.shape[0], -np.inf, dtype=sim.dtype)
        return best_sim, second_sim, best_idx

    # argpartition to get top2 indices per row
    top2 = np.argpartition(sim, kth=-2, axis=1)[:, -2:]
    top2_sim = np.take_along_axis(sim, top2, axis=1)
    # order them
    order = np.argsort(top2_sim, axis=1)
    second_idx = top2[np.arange(sim.shape[0]), order[:, 0]]
    best_idx = top2[np.arange(sim.shape[0]), order[:, 1]]
    best_sim = sim[np.arange(sim.shape[0]), best_idx]
    second_sim = sim[np.arange(sim.shape[0]), second_idx]
    return best_sim, second_sim, best_idx


def run_sweep_stage4a(
    df_dt: pd.DataFrame,
    emb: np.ndarray,
    grid_sim: List[float],
    grid_mar: List[float],
    max_outliers: int = 0,
    outlier_topic: int = -1,
    min_docs_per_topic: int = 2,  # <--- Added parameter
) -> pd.DataFrame:
    """
    Replay Stage 4a-style reassignment using cosine similarity to topic centroids.
    """
    if "topic" not in df_dt.columns:
        raise ValueError("Expected column 'topic' in 04_bertopic_doc_topics.csv")
    if "doc_id" not in df_dt.columns:
        raise ValueError("Expected column 'doc_id' in 04_bertopic_doc_topics.csv")

    N = len(df_dt)
    topics = df_dt["topic"].to_numpy()
    out_idx = np.where(topics == outlier_topic)[0]
    if max_outliers and max_outliers > 0:
        out_idx = out_idx[: int(max_outliers)]

    n_out = int(out_idx.size)
    base_cov = float((N - n_out) / N) if N > 0 else 0.0

    if n_out == 0:
        # still emit a table for completeness
        rows = []
        for sim_thr, mar_thr in product(grid_sim, grid_mar):
            rows.append({
                "tau_sim": sim_thr,
                "tau_mar": mar_thr,
                "N": N,
                "n_outliers_in": 0,
                "n_reassigned": 0,
                "n_outliers_out": 0,
                "coverage_base": base_cov,
                "coverage_after": base_cov,
                "delta_coverage": 0.0,
                "mean_best_sim": np.nan,
                "median_best_sim": np.nan,
                "p90_best_sim": np.nan,
                "mean_margin": np.nan,
                "median_margin": np.nan,
                "p90_margin": np.nan,
            })
        return pd.DataFrame(rows)

    # Use the passed parameter instead of 'args'
    topic_ids, centroids = compute_centroids_cosine(
        df_dt, emb, topic_col="topic", min_docs_per_topic=min_docs_per_topic
    )

    # normalize embeddings once
    emb_n = _l2_normalize_rows(emb.astype(np.float32))
    E = emb_n[out_idx]  # (n_out, d)

    # cosine similarity = dot since both are normalized
    sim_mat = E @ centroids.T  # (n_out, T)

    best_sim, second_sim, best_col = best_and_second_best(sim_mat)
    margin = best_sim - second_sim
    best_topic_id = topic_ids[best_col]

    rows = []
    for sim_thr, mar_thr in product(grid_sim, grid_mar):
        sim_thr = float(sim_thr)
        mar_thr = float(mar_thr)

        accept = (best_sim >= sim_thr) & (margin >= mar_thr)
        n_re = int(np.sum(accept))
        n_out_after = int(n_out - n_re)
        cov_after = float((N - n_out_after) / N) if N > 0 else 0.0

        if n_re > 0:
            bs = best_sim[accept]
            mg = margin[accept]
            row = {
                "tau_sim": sim_thr,
                "tau_mar": mar_thr,
                "N": N,
                "n_outliers_in": n_out,
                "n_reassigned": n_re,
                "n_outliers_out": n_out_after,
                "coverage_base": base_cov,
                "coverage_after": cov_after,
                "delta_coverage": cov_after - base_cov,
                "mean_best_sim": float(np.mean(bs)),
                "median_best_sim": float(np.median(bs)),
                "p90_best_sim": float(np.quantile(bs, 0.90)),
                "mean_margin": float(np.mean(mg)),
                "median_margin": float(np.median(mg)),
                "p90_margin": float(np.quantile(mg, 0.90)),
            }
        else:
            row = {
                "tau_sim": sim_thr,
                "tau_mar": mar_thr,
                "N": N,
                "n_outliers_in": n_out,
                "n_reassigned": 0,
                "n_outliers_out": n_out,
                "coverage_base": base_cov,
                "coverage_after": base_cov,
                "delta_coverage": 0.0,
                "mean_best_sim": np.nan,
                "median_best_sim": np.nan,
                "p90_best_sim": np.nan,
                "mean_margin": np.nan,
                "median_margin": np.nan,
                "p90_margin": np.nan,
            }

        # Optional: where do reassignments go? (top 5)
        if n_re > 0:
            top_targets = pd.Series(best_topic_id[accept]).value_counts().head(5).to_dict()
            row["top5_target_topics"] = ";".join([f"{k}:{v}" for k, v in top_targets.items()])
        else:
            row["top5_target_topics"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to a completed run (e.g., run_main_v32)")
    ap.add_argument("--grid_sim", default="0.65,0.70,0.75,0.80", help="Comma-separated tau_sim values")
    ap.add_argument("--grid_mar", default="0.0,0.001,0.002,0.005,0.01,0.02", help="Comma-separated tau_mar values")
    ap.add_argument("--max_outliers", type=int, default=0, help="If >0, limit sweep to first K outliers (for speed)")
    ap.add_argument("--min_docs_per_topic", type=int, default=2, help="Min docs to compute a centroid for a topic (default 2)")
    ap.add_argument("--out_csv", default="", help="Optional output CSV path; default is run_dir/05_sensitivity_results_stage4a.csv")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    f_dt = run_dir / "04_bertopic_doc_topics.csv"
    f_emb = run_dir / "embeddings.npy"

    if not f_dt.exists():
        print(f"❌ Missing: {f_dt}", file=sys.stderr)
        return 2
    if not f_emb.exists():
        print(f"❌ Missing: {f_emb}", file=sys.stderr)
        return 2

    df_dt = pd.read_csv(f_dt)
    emb = np.load(f_emb)

    grid_sim = _parse_float_list(args.grid_sim)
    grid_mar = _parse_float_list(args.grid_mar)
    if not grid_sim or not grid_mar:
        print("❌ grid_sim and grid_mar must be non-empty lists.", file=sys.stderr)
        return 2

    print(f">>> Sensitivity sweep (Stage 4a replay)")
    print(f"    run_dir: {run_dir}")
    print(f"    doc_topics: {f_dt.name} rows={len(df_dt)}")
    print(f"    embeddings: {f_emb.name} shape={emb.shape}")
    print(f"    grid_sim: {grid_sim}")
    print(f"    grid_mar: {grid_mar}")
    if args.max_outliers and args.max_outliers > 0:
        print(f"    max_outliers: {args.max_outliers}")

    out_df = run_sweep_stage4a(
        df_dt=df_dt,
        emb=emb,
        grid_sim=grid_sim,
        grid_mar=grid_mar,
        max_outliers=int(args.max_outliers),
        min_docs_per_topic=int(args.min_docs_per_topic),  # <--- Passed parameter
    )

    out_csv = Path(args.out_csv) if args.out_csv else (run_dir / "05_sensitivity_results_stage4a.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Wrote: {out_csv}")

    # quick best rows by delta coverage and quality proxy
    if "delta_coverage" in out_df.columns:
        best = out_df.sort_values(["delta_coverage", "mean_best_sim"], ascending=[False, False]).head(10)
        print("\nTop-10 settings by delta_coverage (ties broken by mean_best_sim):")
        with pd.option_context("display.max_columns", 50, "display.width", 160):
            print(best[["tau_sim","tau_mar","delta_coverage","coverage_after","n_reassigned","mean_best_sim","mean_margin","top5_target_topics"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())