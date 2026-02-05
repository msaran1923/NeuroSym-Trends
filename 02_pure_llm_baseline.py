#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_pure_llm_baseline.py

TopicGPT-style pure-LLM topic discovery + refinement + assignment baseline.

Key upgrades:
  1) Incremental Generation (TopicGPT "Start Approach").
  2) Embedding-based refinement (Sentence-Transformers).
  3) Quote-based Assignment Validation (Chain-of-Thought).
  4) Robust JSON parsing (handles single-quotes).
"""

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Ollama Handler
# ----------------------------------------------------------------------------
class OllamaHandler:
    def __init__(self, model_name: str = "qwen3:14b", base_url: str = "http://localhost:11434",
                 temperature: float = 0.0, num_ctx: int = 4096, timeout_s: int = 180):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.num_ctx = int(num_ctx)
        self.timeout_s = int(timeout_s)
        self.call_count = 0

    def query(self, prompt: str, system_prompt: str | None = None) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            self.call_count += 1
            resp = requests.post(url, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            return resp.json().get("response", "") or ""
        except Exception as e:
            print(f"[ollama] Error: {e}", file=sys.stderr)
            return ""


# ----------------------------------------------------------------------------
# Helpers: JSON extraction / text normalization
# ----------------------------------------------------------------------------
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_JSON_LIST_RE = re.compile(r"(\[.*\])", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _strip_code_fences(s: str) -> str:
    if not s:
        return ""
    m = _JSON_BLOCK_RE.search(s)
    return (m.group(1) if m else s).strip()


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _clean_label(lbl: str) -> str:
    if not lbl or not isinstance(lbl, str):
        return "Outlier"
    lbl = _strip_code_fences(lbl)
    lbl = re.sub(r"^\d+[\.:\)]\s*", "", lbl)  # "1. "
    lbl = re.sub(r"^(Topic|Label)\s*:\s*", "", lbl, flags=re.IGNORECASE)
    lbl = lbl.strip("'\" \t\r\n")
    return _normalize_ws(lbl)


def _safe_json_loads(s: str):
    if not s:
        return None
    s = _strip_code_fences(s)

    # Try direct JSON first
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try extract list/object substrings
    for rgx in (_JSON_LIST_RE, _JSON_OBJ_RE):
        m = rgx.search(s)
        if m:
            cand = m.group(1)
            try:
                return json.loads(cand)
            except Exception:
                # Try python literal eval for single quotes etc.
                try:
                    import ast
                    return ast.literal_eval(cand)
                except Exception:
                    pass

    # Last resort: literal eval whole string
    try:
        import ast
        return ast.literal_eval(s)
    except Exception:
        return None


def _ensure_topic_obj(x) -> dict | None:
    """Normalize LLM topic item into {'label','desc','keywords'} or None."""
    if isinstance(x, str):
        lbl = _clean_label(x)
        if not lbl:
            return None
        return {"label": lbl, "desc": "", "keywords": ""}
    if isinstance(x, dict):
        if "label" not in x:
            return None
        lbl = _clean_label(x.get("label", ""))
        if not lbl:
            return None
        desc = _normalize_ws(x.get("desc", ""))[:500]
        kw = x.get("keywords", "")
        if isinstance(kw, list):
            kw = ", ".join([_normalize_ws(str(k)) for k in kw if str(k).strip()])
        kw = _normalize_ws(str(kw))[:400]
        return {"label": lbl, "desc": desc, "keywords": kw}
    return None


# ----------------------------------------------------------------------------
# Sentence-Transformer embeddings + clustering
# ----------------------------------------------------------------------------
def _try_load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer(model_name)
    except Exception as e:
        return None


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def cluster_topics_by_similarity(topic_texts: list[str], st_model, sim_threshold: float,
                                max_full_matrix: int = 2000) -> list[list[int]]:
    """
    Build clusters of topic indices whose ST cosine similarity >= sim_threshold.
    """
    if st_model is None or not topic_texts:
        return [[i] for i in range(len(topic_texts))]

    emb = st_model.encode(topic_texts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    n = emb.shape[0]
    uf = UnionFind(n)

    if n <= max_full_matrix:
        sim = emb @ emb.T
        for i in range(n):
            row = sim[i]
            js = np.where(row >= sim_threshold)[0]
            for j in js:
                if j <= i:
                    continue
                uf.union(i, int(j))
    else:
        chunk = 512
        for i0 in range(0, n, chunk):
            i1 = min(n, i0 + chunk)
            sim_chunk = emb[i0:i1] @ emb.T
            for ii in range(i0, i1):
                row = sim_chunk[ii - i0]
                js = np.where(row >= sim_threshold)[0]
                for j in js:
                    if j <= ii:
                        continue
                    uf.union(ii, int(j))

    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    return sorted([sorted(v) for v in groups.values()], key=lambda x: (len(x), x[0]), reverse=True)


# ----------------------------------------------------------------------------
# Phase 1: Incremental topic discovery + frequency tracking
# ----------------------------------------------------------------------------
@dataclass
class GenConfig:
    max_docs: int
    batch_size: int
    stop_no_new_docs: int
    max_raw_topics: int
    doc_chars: int


def generate_topics_incremental(df: pd.DataFrame, llm: OllamaHandler, seed: int, cfg: GenConfig,
                               seed_topics: list[dict] | None = None,
                               max_prompt_topics: int = 80):
    texts = df["text"].tolist()
    n_docs = len(texts)
    order = list(range(n_docs))
    rnd = random.Random(seed)
    rnd.shuffle(order)

    def topic_key(label: str) -> str:
        return _normalize_ws(label).lower()

    if seed_topics is None:
        seed_topics = [
            {
                "label": "Machine Learning",
                "desc": "Methods that learn patterns from data.",
                "keywords": "learning, neural networks, classification",
            },
            {
                "label": "Computer Systems",
                "desc": "Hardware, networks, and operating systems.",
                "keywords": "hardware, networks, os, distributed",
            },
        ]

    S: dict[str, dict] = {}
    freq = Counter()

    for t in seed_topics:
        obj = _ensure_topic_obj(t)
        if obj:
            k = topic_key(obj["label"])
            S[k] = {"label": obj["label"], "desc": obj.get("desc", ""), "keywords": obj.get("keywords", ""), "gen_freq": 0}

    processed = 0
    docs_since_new_topic = 0
    raw_candidates: list[dict] = []

    pbar = tqdm(total=min(cfg.max_docs, n_docs), desc="Discovering (TopicGPT)", unit="doc")

    while processed < min(cfg.max_docs, n_docs) and docs_since_new_topic < cfg.stop_no_new_docs:
        batch_idxs = order[processed:processed + cfg.batch_size]
        if not batch_idxs:
            break

        batch_texts = []
        for j, di in enumerate(batch_idxs, start=1):
            t = _normalize_ws(texts[di])[:cfg.doc_chars]
            batch_texts.append(f"Doc {j}: {t}")

        docs_block = "\n\n".join(batch_texts)

        cur = list(S.values())
        cur.sort(key=lambda x: (-int(x.get("gen_freq", 0)), x["label"].lower()))
        cur = cur[:max_prompt_topics]

        topics_block = "\n".join(
            [f"- {t['label']}: {t.get('desc','')}" for t in cur if t.get("label")]
        )

        system_msg = "You are a precise topic modeler."
        prompt = (
            f"You will be given (1) an existing topic set S and (2) a batch of documents.\n"
            f"For EACH document, choose exactly ONE topic label from S that best fits, OR create a NEW topic.\n\n"
            f"Existing topics S:\n{topics_block}\n\n"
            f"Documents:\n{docs_block}\n\n"
            "Output MUST be a single JSON object with fields:\n"
            '{"assignments":[{"doc":1,"label":"..."},...],"new_topics":[{"label":"...","desc":"...","keywords":"..."}]}\n\n'
            f"Rules:\n"
            f"1) assignments must include one entry per document (1..{len(batch_idxs)}).\n"
            f"2) assignment labels must be from S, OR in new_topics, OR 'Outlier'.\n"
            f"3) Create new_topics ONLY if needed.\n"
            f"4) Output JSON only.\n"
        )

        resp = llm.query(prompt, system_prompt=system_msg)
        parsed = _safe_json_loads(resp)

        assignments = []
        new_topics = []

        if isinstance(parsed, dict):
            assignments = parsed.get("assignments", [])
            new_topics = parsed.get("new_topics", [])
        elif isinstance(parsed, list):
            new_topics = parsed

        created_this_batch = 0
        if isinstance(new_topics, list):
            for item in new_topics:
                obj = _ensure_topic_obj(item)
                if not obj: continue
                k = topic_key(obj["label"])
                if k in S: continue
                S[k] = {"label": obj["label"], "desc": obj.get("desc", ""), "keywords": obj.get("keywords", ""), "gen_freq": 0}
                raw_candidates.append(obj)
                created_this_batch += 1

        if isinstance(assignments, list) and len(assignments) == len(batch_idxs):
            for a in assignments:
                if not isinstance(a, dict): continue
                lbl = _clean_label(str(a.get("label", "")))
                if not lbl or lbl == "Outlier": continue
                k = topic_key(lbl)
                if k in S:
                    freq[k] += 1
                    S[k]["gen_freq"] = int(S[k].get("gen_freq", 0)) + 1

        processed += len(batch_idxs)
        pbar.update(len(batch_idxs))

        if created_this_batch > 0:
            docs_since_new_topic = 0
        else:
            docs_since_new_topic += len(batch_idxs)

        if len(S) >= cfg.max_raw_topics:
            print(f"\n[warn] Reached max_raw_topics={cfg.max_raw_topics}.")
            break

    pbar.close()
    canon_topics = list(S.values())
    canon_topics.sort(key=lambda x: (-int(x.get("gen_freq", 0)), x["label"].lower()))
    return {
        "raw_candidates": raw_candidates,
        "canon_topics": canon_topics,
        "gen_freq": freq,
        "docs_processed": processed,
    }


# ----------------------------------------------------------------------------
# Phase 1b: Embedding-based near-duplicate detection + LLM cluster merge
# ----------------------------------------------------------------------------
def merge_cluster_with_llm(llm: OllamaHandler, cluster_topics: list[dict]) -> dict | None:
    if not cluster_topics: return None
    lines = []
    for t in cluster_topics:
        lines.append(f"- {t.get('label','')}: {t.get('desc','')} ({t.get('keywords','')})")

    system_msg = "Consolidate similar topics into ONE canonical topic."
    user_msg = (
        f"Merge these topics:\n{chr(10).join(lines)}\n\n"
        "Output JSON: {\"label\":..., \"desc\":..., \"keywords\":...}"
    )
    resp = llm.query(user_msg, system_prompt=system_msg)
    parsed = _safe_json_loads(resp)
    if isinstance(parsed, dict):
        return _ensure_topic_obj(parsed)
    return None


def refine_topics_with_embeddings(canon_topics: list[dict], llm: OllamaHandler,
                                 st_model_name: str, sim_threshold: float,
                                 topic_min_freq: int, max_topics: int,
                                 max_full_matrix: int = 2000):
    kept = [t for t in canon_topics if int(t.get("gen_freq", 0)) >= int(topic_min_freq)]
    if not kept:
        kept = canon_topics[:max(1, min(len(canon_topics), max_topics))]

    st_model = _try_load_sentence_transformer(st_model_name)
    if st_model is None:
        print("[warn] sentence-transformers not installed. Skipping embed refinement.")
        kept.sort(key=lambda x: (-int(x.get("gen_freq", 0)), x["label"].lower()))
        return kept[:max_topics], {"used_embeddings": False}

    topic_texts = [_normalize_ws(f"{t['label']} {t.get('keywords','')} {t.get('desc','')}") for t in kept]
    clusters = cluster_topics_by_similarity(topic_texts, st_model, float(sim_threshold), max_full_matrix)

    refined = []
    for cl in tqdm(clusters, desc="Refining (merge)", unit="cluster"):
        members = [kept[i] for i in cl]
        if len(members) == 1:
            refined.append(members[0])
            continue

        merged = merge_cluster_with_llm(llm, members)
        if merged:
            merged["gen_freq"] = sum(int(m.get("gen_freq",0)) for m in members)
            refined.append(merged)
        else:
            members.sort(key=lambda x: -int(x.get("gen_freq", 0)))
            refined.append(members[0])

    seen = {}
    for t in refined:
        k = t["label"].strip().lower()
        if k not in seen or int(t.get("gen_freq",0)) > int(seen[k].get("gen_freq",0)):
            seen[k] = t

    out = list(seen.values())
    out.sort(key=lambda x: (-int(x.get("gen_freq", 0)), x["label"].lower()))
    return out[:max_topics], {"used_embeddings": True, "clusters": len(clusters)}


# ----------------------------------------------------------------------------
# Phase 2: Assignment
# ----------------------------------------------------------------------------
def _build_taxonomy_string(taxonomy: list[dict], max_chars: int = 12000) -> str:
    lines = []
    for t in taxonomy:
        line = f"- {t.get('label','')}: {t.get('desc','')}"
        lines.append(line)
        if sum(len(x) for x in lines) > max_chars: break
    return "\n".join(lines)


def assign_one_with_retries(llm: OllamaHandler, doc_text: str, taxonomy: list[dict],
                            valid_labels: set[str], max_retries: int,
                            doc_chars: int, require_quote: bool, quote_case_insensitive: bool):
    tax_str = _build_taxonomy_string(taxonomy)
    doc = _normalize_ws(doc_text)[:doc_chars]

    system_msg = "You are an expert annotator. Output JSON only."
    base_prompt = (
        f"Assign this document to ONE topic from the list, or 'Outlier'.\n\n"
        f"Taxonomy:\n{tax_str}\n\n"
        f"Document:\n{doc}\n\n"
        "Requirements:\n"
        "1) label: Must be exact from taxonomy or 'Outlier'.\n"
        "2) quote: Verbatim substring from text supporting the label.\n"
        "3) description: Short reasoning.\n"
        "Output JSON: {\"label\":..., \"quote\":..., \"description\":...}"
    )

    last_raw = ""
    for attempt in range(max_retries + 1):
        prompt = base_prompt if attempt == 0 else \
            f"Fix previous invalid output.\nError with: {last_raw}\n\n{base_prompt}"

        raw = llm.query(prompt, system_prompt=system_msg)
        last_raw = (raw or "").strip()
        parsed = _safe_json_loads(last_raw)

        if not isinstance(parsed, dict): continue
        lbl = _clean_label(parsed.get("label",""))
        qt = parsed.get("quote","")

        if lbl != "Outlier" and lbl not in valid_labels: continue
        if require_quote and qt:
            q_norm = _normalize_ws(qt)
            d_norm = _normalize_ws(doc)
            if quote_case_insensitive:
                if q_norm.lower() not in d_norm.lower(): continue
            else:
                if q_norm not in d_norm: continue

        return {
            "label": lbl, "quote": qt, "description": parsed.get("description",""),
            "status": "ok", "retries": attempt, "raw_output": last_raw
        }

    return {"label": "Outlier", "status": "failed", "retries": max_retries, "raw_output": last_raw}


def assign_topics_to_all(df: pd.DataFrame, llm: OllamaHandler, taxonomy: list[dict],
                         assign_max_retries: int, doc_chars: int,
                         require_quote: bool, quote_case_insensitive: bool):
    print(f"\n--- Phase 2: Assignment (Docs: {len(df)}) ---")
    valid = set([t["label"] for t in taxonomy])
    results = []
    counts = Counter()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Assigning"):
        out = assign_one_with_retries(
            llm, row["text"], taxonomy, valid, assign_max_retries,
            doc_chars, require_quote, quote_case_insensitive
        )
        lbl = out["label"]
        counts[lbl] += 1
        results.append({
            "doc_id": str(row.get("doc_id", idx)),
            "topic_label": lbl,
            "description": out.get("description",""),
            "quote": out.get("quote",""),
            "status": out.get("status",""),
            "retries": out.get("retries",0),
            "raw_output": out.get("raw_output","")
        })
    return results, counts


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    start_time = time.time()  # <--- TIMER START

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=Path)
    ap.add_argument("--preproc_csv", required=True, type=Path)
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--seed", type=int, default=42)

    # Tuning params
    ap.add_argument("--max_topics", type=int, default=200)
    ap.add_argument("--gen_max_docs", type=int, default=2000)
    ap.add_argument("--topic_min_freq", type=int, default=4)
    ap.add_argument("--st_model", default="local_bge-large-en-v1.5") #all-MiniLM-L6-v2
    ap.add_argument("--st_sim_threshold", type=float, default=0.5)
    ap.add_argument("--no_assign_require_quote", action="store_true")

    args = ap.parse_args()
    random.seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.preproc_csv)
    if "text" not in df.columns:
        df["text"] = (df["english_title"].fillna("") + "\n" + df["abstract_en"].fillna("")).str.strip()
    df = df[df["text"].astype(str).str.len() > 10].copy()

    llm = OllamaHandler(model_name=args.model, base_url=args.url)

    # 1. Generate (Incremental)
    gen_out = generate_topics_incremental(df, llm, args.seed,
        GenConfig(args.gen_max_docs, 5, 200, 2000, 400))

    # 2. Refine (Embeddings + LLM)
    taxonomy, meta = refine_topics_with_embeddings(
        gen_out["canon_topics"], llm, args.st_model, args.st_sim_threshold,
        args.topic_min_freq, args.max_topics
    )
    (args.run_dir / "02_refined_taxonomy.json").write_text(json.dumps(taxonomy, indent=2))

    # 3. Assign
    results, counts = assign_topics_to_all(df, llm, taxonomy, 5, 1500, not args.no_assign_require_quote, False)

    # Output
    res_df = pd.DataFrame(results)
    sorted_topics = sorted([t["label"] for t in taxonomy])
    label_map = {t: i for i, t in enumerate(sorted_topics)}
    label_map["Outlier"] = -1
    res_df["topic"] = res_df["topic_label"].map(label_map).fillna(-1).astype(int)
    res_df[["doc_id", "topic", "topic_label"]].to_csv(args.run_dir / "09_final_doc_topics.csv", index=False)
    res_df.to_csv(args.run_dir / "09_final_doc_topics_rich.csv", index=False)

    meta_rows = []
    for t in taxonomy:
        meta_rows.append({
            "topic": label_map.get(t["label"], -1),
            "count": counts[t["label"]],
            "label": t["label"],
            "name": t["label"],
            "keywords": t.get("keywords",""),
            "gen_freq": t.get("gen_freq",0)
        })
    pd.DataFrame(meta_rows).to_csv(args.run_dir / "09_final_topics.csv", index=False)

    # Metrics
    elapsed = time.time() - start_time  # <--- TIMER STOP
    metrics = {
        "run_dir": str(args.run_dir),
        "wall_clock_seconds": elapsed,
        "llm_calls": llm.call_count,
        "topics_found": len(taxonomy),
        "outliers": counts["Outlier"],
        "avg_semantic_drift": float("nan") # Placeholder
    }
    (args.run_dir / "14_run_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Done in {elapsed:.1f}s. Topics: {len(taxonomy)}")

if __name__ == "__main__":
    main()
