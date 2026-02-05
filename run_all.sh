#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# EXPERIMENT RUNNER (High Coverage + High Coherence Version)
# =============================================================================

# -----------------------------
# Inputs & Configuration
# -----------------------------
DATA_CSV="${1:-${DATA_CSV:-data.csv}}"
CSO_TTL="${2:-${CSO_TTL:-CSO.3.5.ttl}}"

# 5 seeds is a solid reproducibility baseline
SEEDS=(${SEEDS_OVERRIDE:-32 33 34 35 36})

# -----------------------------
# Scripts
# -----------------------------
PIPELINE_BARE_PY="00_bare_bertopic_baseline.py"
PIPELINE_PY="01_thesis_topic_trends_pipeline.py"
# UPGRADED: Switch to TopicGPTv3 for better coherence in baseline
PIPELINE_PURE_LLM_PY="02_pure_llm_baseline.py"
EVAL_PY="03_evaluate_topic_model.py"
TRENDS_PY="04_trends_only.py"
SENS_SWEEP_PY="05_sensitivity_sweep.py"

# -----------------------------
# Main Pipeline Params (TUNED)
# -----------------------------
EMBED_MODEL_MAIN="${EMBED_MODEL_MAIN:-local_bge-large-en-v1.5}"
BATCH_SIZE_MAIN="${BATCH_SIZE_MAIN:-512}"

# Dimensionality Reduction & Clustering
UMAP_NEI="${UMAP_NEI:-15}"
UMAP_COMP="${UMAP_COMP:-5}"
UMAP_MIN_DIST="${UMAP_MIN_DIST:-0.0}"
# FIXED: Size 25 ensures valid NPMI stats (Coherence)
MIN_CLUSTER="${MIN_CLUSTER:-30}"
MIN_SAMPLES="${MIN_SAMPLES:-5}"
REDUCE_TOPICS="${REDUCE_TOPICS:-none}"
EXTRA_STOPWORDS="${EXTRA_STOPWORDS:-aim,aims,purpose,purposes,objective,objectives,study,studies,research,paper,thesis,dissertation,propose,proposes,proposed,present,presents,presented,approach,approaches,method,methods,methodology,result,results,conclusion,conclusions,experiment,experiments,evaluate,evaluation,performance,based,using,use,used,novel,new,implementation,application,applications,problem,problems,challenge,challenges,existing}"

# Niche Sensitivity
# FIXED: Increased to 5 to filter rare noise words (Coherence)
VECTORIZER_MIN_DF="${VECTORIZER_MIN_DF:-10}"

# Outlier Management
MAX_OUTLIERS="${MAX_OUTLIERS:-0}"
REASSIGN_SIM="${REASSIGN_SIM:-0.62}"
# FIXED: Stricter margin forces ambiguous docs to LLM (Coherence)
REASSIGN_MARGIN="${REASSIGN_MARGIN:-0.004}"
# FIXED: Disable auto-margin to ensure fixed margin takes effect
REASSIGN_MARGIN_AUTO="${REASSIGN_MARGIN_AUTO:-0}"
REASSIGN_MARGIN_Q="${REASSIGN_MARGIN_Q:-0.0}"
ENTROPY_THR="${ENTROPY_THR:-1.0}"
ENTROPY_AUTO="${ENTROPY_AUTO:-1}"
ENTROPY_Q="${ENTROPY_Q:-0.85}"
ENTROPY_GATE="${ENTROPY_GATE:-1}"
CSO_GATE_THR="${CSO_GATE_THR:-0.55}"

# Augmentation (Stage 6)
AUG_MIN_CLUSTER="${AUG_MIN_CLUSTER:-4}"
AUG_MIN_SAMPLES="${AUG_MIN_SAMPLES:-4}"
AUG_MIN_TOPWORD="${AUG_MIN_TOPWORD:-0.0015}"
# FIXED: Higher purity rejects garbage symbolic topics (Coherence)
AUG_CSO_PURITY_MIN="${AUG_CSO_PURITY_MIN:-0.70}"
AUG_UMAP_NEI="${AUG_UMAP_NEI:-15}"
AUG_UMAP_COMP="${AUG_UMAP_COMP:-5}"
AUG_UMAP_MIN_DIST="${AUG_UMAP_MIN_DIST:-0.0}"
AUG_CLUSTER_SELECTION="${AUG_CLUSTER_SELECTION:-leaf}"

# Topic Splitting (Stage 7)
SPLIT_ENABLE="${SPLIT_ENABLE:-1}"
SPLIT_MIN_TOPIC_SIZE="${SPLIT_MIN_TOPIC_SIZE:-60}"
SPLIT_TRIGGER_MEAN_SIM="${SPLIT_TRIGGER_MEAN_SIM:-0.55}"
# FIXED: Aggressive split trigger for super-clusters (Coherence)
SPLIT_TRIGGER_STD_SIM="${SPLIT_TRIGGER_STD_SIM:-0.10}"
SPLIT_UMAP_NEI="${SPLIT_UMAP_NEI:-15}"
SPLIT_UMAP_COMP="${SPLIT_UMAP_COMP:-5}"
SPLIT_UMAP_MIN_DIST="${SPLIT_UMAP_MIN_DIST:-0.0}"
SPLIT_MIN_CLUSTER="${SPLIT_MIN_CLUSTER:-4}"
SPLIT_MIN_SAMPLES="${SPLIT_MIN_SAMPLES:-2}"
SPLIT_MIN_TOPWORD="${SPLIT_MIN_TOPWORD:-0.0035}"
SPLIT_DUP_SIM="${SPLIT_DUP_SIM:-0.90}"
SPLIT_CSO_PURITY_MIN="${SPLIT_CSO_PURITY_MIN:-0.0}"
SPLIT_CLUSTER_SELECTION="${SPLIT_CLUSTER_SELECTION:-leaf}"

# Symbolic (CSO) Params
SYMBOLIC_MIN_CLUSTER="${SYMBOLIC_MIN_CLUSTER:-4}"
SYMBOLIC_TOP1_SIM="${SYMBOLIC_TOP1_SIM:-0.65}"
SYMBOLIC_MAX_TOPICS="${SYMBOLIC_MAX_TOPICS:-250}"
CSO_TOPK="${CSO_TOPK:-10}"
CSO_THR="${CSO_THR:-0.55}"
CSO_DOC_BS="${CSO_DOC_BS:-256}"

# Weighting & Trends
CTFIDF_TOPN="${CTFIDF_TOPN:-15}"
CTFIDF_WEIGHTING="${CTFIDF_WEIGHTING:-confidence}"

# Keyword refinement (post-finalization)
KW_REFINE="${KW_REFINE:-1}"
KW_REFINE_CORE_K="${KW_REFINE_CORE_K:-12}"
KW_REFINE_DISPLAY_K="${KW_REFINE_DISPLAY_K:-12}"
KW_REFINE_CANDIDATE_POOL="${KW_REFINE_CANDIDATE_POOL:-60}"
KW_REFINE_MIN_DF_GLOBAL="${KW_REFINE_MIN_DF_GLOBAL:-10}"
KW_REFINE_WEIGHTING="${KW_REFINE_WEIGHTING:-none}"
MERGE_MICRO_TOPICS="${MERGE_MICRO_TOPICS:-1}"
MICRO_MERGE_MIN_SIZE="${MICRO_MERGE_MIN_SIZE:-10}"
MICRO_MERGE_SIM_THRESHOLD="${MICRO_MERGE_SIM_THRESHOLD:-0.80}"
TREND_ALPHA="${TREND_ALPHA:-0.05}"
MIN_YEAR_DOCS="${MIN_YEAR_DOCS:-10}"
STAB_RUNS_MAIN="${STAB_RUNS_MAIN:-5}"
TRENDS_STABILITY_RUNS="${TRENDS_STABILITY_RUNS:-5}"
MIN_DOCS_SHIFT="${MIN_DOCS_SHIFT:-15}"

# Sensitivity sweep (Stage 4a replay)
SENS_SWEEP="${SENS_SWEEP:-0}"
SENS_SWEEP_SEED="${SENS_SWEEP_SEED:-${SEEDS[0]}}"
SENS_GRID_SIM="${SENS_GRID_SIM:-0.65,0.70,0.75,0.80}"
SENS_GRID_MAR="${SENS_GRID_MAR:-0.0,0.001,0.002,0.005,0.01,0.02}"
SENS_MAX_OUTLIERS="${SENS_MAX_OUTLIERS:-0}"

# LLM Gate
#LLM_MODE="${LLM_MODE:-transformers}"
LLM_MODE="${LLM_MODE:-ollama}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:14b}"
LLM_TOPK="${LLM_TOPK:-8}"
# FIXED: Relaxed slightly to catch overflow from stricter Stage 4
LLM_SIM_FLOOR="${LLM_SIM_FLOOR:-0.53}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_WORKERS="${OLLAMA_WORKERS:-4}"

# Report Config
RUN_REPORT="${RUN_REPORT:-1}"
REPORT_TOP_N="${REPORT_TOP_N:-20}"
REPORT_INCLUDE_EXAMPLES="${REPORT_INCLUDE_EXAMPLES:-0}"
REPORT_EXAMPLES_PER_TOPIC="${REPORT_EXAMPLES_PER_TOPIC:-3}"
REPORT_MIN_TOPIC_SIZE_FOR_EXAMPLES="${REPORT_MIN_TOPIC_SIZE_FOR_EXAMPLES:-25}"
REPORT_PREFER_RECENT="${REPORT_PREFER_RECENT:-1}"
REPORT_MAX_EXAMPLE_CHARS="${REPORT_MAX_EXAMPLE_CHARS:-300}"

# -----------------------------
# Helper Functions
# -----------------------------
function Require-File() {
  local p="$1"
  if [ ! -f "$p" ]; then
    echo "ERROR: Missing file: $p" >&2
    exit 1
  fi
}

function Exec-Py() {
  echo "> python3 $*" >&2
  python3 "$@"
}

run_eval() {
  local run_dir="$1"
  local seed="$2"
  local tag="$3"
  if [ ! -f "${EVAL_PY}" ]; then
    echo "(eval) Skipping: ${EVAL_PY} not found."
    return
  fi
  echo
  echo ">>> Evaluate (${tag}): ${run_dir}"
  Exec-Py "${EVAL_PY}" --run_dir "${run_dir}" --seed "${seed}" --top_n 20
}

run_trends_if_possible() {
  local run_dir="$1"
  local stability_runs="${2:-${TRENDS_STABILITY_RUNS}}"
  if [ ! -f "${TRENDS_PY}" ]; then
    echo "(trends) Skipping: ${TRENDS_PY} not found."
    return
  fi
  echo
  echo ">>> Trends-only: ${run_dir} (stability_runs=${stability_runs})"

  Exec-Py "${TRENDS_PY}" \
    --run_dir "${run_dir}" \
    --pipeline_py "${PIPELINE_PY}" \
    --preproc_csv "${run_dir}/01_preprocessed_documents.csv" \
    --embed_model "${EMBED_MODEL_MAIN}" \
    --batch_size "${BATCH_SIZE_MAIN}" \
    --trend_alpha "${TREND_ALPHA}" \
    --min_year_docs "${MIN_YEAR_DOCS}" \
    --stability_runs "${stability_runs}" \
    --min_docs_shift "${MIN_DOCS_SHIFT}" \
    --vectorizer_min_df "${VECTORIZER_MIN_DF}" \
    --trend_min_topic_size 15
}

run_report_if_possible() {
  local run_dir="$1"
  local sample_seed="${2:-32}"
  if [ "${RUN_REPORT}" = "0" ]; then return; fi
  if python3 "${PIPELINE_PY}" report --help >/dev/null 2>&1; then
    echo
    echo ">>> Stage 13 Report: ${run_dir}"
    args=(report --run_dir "${run_dir}" --top_n "${REPORT_TOP_N}")
    if [ "${REPORT_PREFER_RECENT}" = "1" ]; then args+=(--prefer_recent); fi
    if [ "${REPORT_INCLUDE_EXAMPLES}" = "1" ]; then
      args+=(--include_examples --examples_per_topic "${REPORT_EXAMPLES_PER_TOPIC}" --min_topic_size_for_examples "${REPORT_MIN_TOPIC_SIZE_FOR_EXAMPLES}" --sample_seed "${sample_seed}" --max_example_chars "${REPORT_MAX_EXAMPLE_CHARS}")
    fi
    Exec-Py "${PIPELINE_PY}" "${args[@]}"
  else
    echo "(report) Skipping: ${PIPELINE_PY} does not expose a report subcommand."
  fi
}

Require-File "${DATA_CSV}"
Require-File "${CSO_TTL}"

# =============================================================================
# EXPERIMENT 1: MAIN PIPELINE (Neuro-Symbolic)
# =============================================================================
for SEED in "${SEEDS[@]}"; do
  RUN_DIR="run_main_v${SEED}"
  echo
  echo "==============================="
  echo "MAIN PIPELINE | SEED=${SEED}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "==============================="

  Exec-Py "${PIPELINE_PY}" \
    --data "${DATA_CSV}" \
    --run_dir "${RUN_DIR}" \
    --seed "${SEED}" \
    --cso_ttl "${CSO_TTL}" \
    --cso_topk "${CSO_TOPK}" \
    --cso_sim_threshold "${CSO_THR}" \
    --cso_doc_batch_size "${CSO_DOC_BS}" \
    --embed_model "${EMBED_MODEL_MAIN}" \
    --batch_size "${BATCH_SIZE_MAIN}" \
    --umap_neighbors "${UMAP_NEI}" \
    --umap_components "${UMAP_COMP}" \
    --umap_min_dist "${UMAP_MIN_DIST}" \
    --min_cluster_size "${MIN_CLUSTER}" \
    --min_samples "${MIN_SAMPLES}" \
    --reduce_topics "${REDUCE_TOPICS}" \
    --max_outliers "${MAX_OUTLIERS}" \
    --reassign_sim_threshold "${REASSIGN_SIM}" \
    --llm_mode "${LLM_MODE}" \
    --llm_model "${LLM_MODEL}" \
    --llm_topk "${LLM_TOPK}" \
    --llm_sim_floor "${LLM_SIM_FLOOR}" \
    --ollama_url "${OLLAMA_URL}" \
    --augment_min_cluster_size "${AUG_MIN_CLUSTER}" \
    --augment_min_samples "${AUG_MIN_SAMPLES}" \
    --augment_quality_min_topword_weight "${AUG_MIN_TOPWORD}" \
    --augment_cso_purity_min "${AUG_CSO_PURITY_MIN}" \
    --augment_umap_neighbors "${AUG_UMAP_NEI}" \
    --augment_umap_components "${AUG_UMAP_COMP}" \
    --augment_umap_min_dist "${AUG_UMAP_MIN_DIST}" \
    --augment_cluster_selection_method "${AUG_CLUSTER_SELECTION}" \
    --symbolic_min_cluster_size "${SYMBOLIC_MIN_CLUSTER}" \
    --symbolic_top1_sim_min "${SYMBOLIC_TOP1_SIM}" \
    --symbolic_max_topics "${SYMBOLIC_MAX_TOPICS}" \
    $( [ "${SPLIT_ENABLE}" = "1" ] && echo "--split_topics" || echo "--no_split_topics" ) \
    --split_min_topic_size "${SPLIT_MIN_TOPIC_SIZE}" \
    --split_trigger_mean_sim "${SPLIT_TRIGGER_MEAN_SIM}" \
    --split_trigger_std_sim "${SPLIT_TRIGGER_STD_SIM}" \
    --split_umap_neighbors "${SPLIT_UMAP_NEI}" \
    --split_umap_components "${SPLIT_UMAP_COMP}" \
    --split_umap_min_dist "${SPLIT_UMAP_MIN_DIST}" \
    --split_min_cluster_size "${SPLIT_MIN_CLUSTER}" \
    --split_min_samples "${SPLIT_MIN_SAMPLES}" \
    --split_cluster_selection_method "${SPLIT_CLUSTER_SELECTION}" \
    --split_quality_min_topword_weight "${SPLIT_MIN_TOPWORD}" \
    --split_duplicate_sim_threshold "${SPLIT_DUP_SIM}" \
    --split_cso_purity_min "${SPLIT_CSO_PURITY_MIN}" \
    --ctfidf_topn "${CTFIDF_TOPN}" \
    --ctfidf_weighting "${CTFIDF_WEIGHTING}" \
    --vectorizer_min_df "${VECTORIZER_MIN_DF}" \
    --entropy_threshold "${ENTROPY_THR}" \
    --trend_alpha "${TREND_ALPHA}" \
    --min_year_docs "${MIN_YEAR_DOCS}" \
    --stability_runs "${STAB_RUNS_MAIN}" \
    --reassign_margin_threshold "${REASSIGN_MARGIN}" \
    --reassign_margin_quantile "${REASSIGN_MARGIN_Q}" \
    $( [ "${REASSIGN_MARGIN_AUTO}" = "1" ] && echo "--reassign_margin_auto" ) \
    --entropy_quantile "${ENTROPY_Q}" \
    $( [ "${ENTROPY_AUTO}" = "1" ] && echo "--entropy_auto" ) \
    $( [ "${ENTROPY_GATE}" = "1" ] && echo "--entropy_gate --cso_sim_threshold_gate ${CSO_GATE_THR}" ) \
    $( [ "${KW_REFINE}" = "1" ] && echo "--kw_refine" || echo "--no_kw_refine" ) \
    --kw_refine_core_k "${KW_REFINE_CORE_K}" \
    --kw_refine_display_k "${KW_REFINE_DISPLAY_K}" \
    --kw_refine_candidate_pool "${KW_REFINE_CANDIDATE_POOL}" \
    --kw_refine_min_df_global "${KW_REFINE_MIN_DF_GLOBAL}" \
    --kw_refine_weighting "${KW_REFINE_WEIGHTING}" \
    $( [ "${MERGE_MICRO_TOPICS}" = "1" ] && echo "--merge_micro_topics" || echo "--no_merge_micro_topics" ) \
    --micro_merge_min_topic_size "${MICRO_MERGE_MIN_SIZE}" \
    --micro_merge_sim_threshold "${MICRO_MERGE_SIM_THRESHOLD}" \
    --extra_stopwords "${EXTRA_STOPWORDS}"

  run_eval "${RUN_DIR}" "${SEED}" "main"
  run_trends_if_possible "${RUN_DIR}" "${TRENDS_STABILITY_RUNS}"
  run_report_if_possible "${RUN_DIR}" "${SEED}"
  if [ "${SENS_SWEEP}" = "1" ] && [ "${SEED}" = "${SENS_SWEEP_SEED}" ]; then
    if [ -f "${SENS_SWEEP_PY}" ]; then
      echo
      echo ">>> Sensitivity sweep (Stage 4a): ${RUN_DIR}"
      Exec-Py "${SENS_SWEEP_PY}" --run_dir "${RUN_DIR}" --grid_sim "${SENS_GRID_SIM}" --grid_mar "${SENS_GRID_MAR}" --max_outliers "${SENS_MAX_OUTLIERS}"
    else
      echo "(sweep) Skipping: ${SENS_SWEEP_PY} not found."
    fi
  fi
done

# =============================================================================
# EXPERIMENT 2: BARE BERTOPIC BASELINE
# =============================================================================
for SEED in "${SEEDS[@]}"; do
  RUN_DIR="run_bare_v${SEED}"
  echo
  echo "==============================="
  echo "BARE BERTopic | SEED=${SEED}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "==============================="

  Exec-Py "${PIPELINE_BARE_PY}" \
    --data "${DATA_CSV}" \
    --run_dir "${RUN_DIR}" \
    --seed "${SEED}" \
    --pipeline_py "${PIPELINE_PY}" \
    --embed_model "${EMBED_MODEL_MAIN}" \
    --batch_size "${BATCH_SIZE_MAIN}" \
    --umap_neighbors "${UMAP_NEI}" \
    --umap_components "${UMAP_COMP}" \
    --umap_min_dist "${UMAP_MIN_DIST}" \
    --min_cluster_size "${MIN_CLUSTER}" \
    --min_samples "${MIN_SAMPLES}" \
    --reduce_topics "${REDUCE_TOPICS}" \
    --vectorizer_min_df "${VECTORIZER_MIN_DF}" \
    --extra_stopwords "${EXTRA_STOPWORDS}"

  run_eval "${RUN_DIR}" "${SEED}" "bare"
  run_trends_if_possible "${RUN_DIR}" "${TRENDS_STABILITY_RUNS}"
  run_report_if_possible "${RUN_DIR}" "${SEED}"
  if [ "${SENS_SWEEP}" = "1" ] && [ "${SEED}" = "${SENS_SWEEP_SEED}" ]; then
    if [ -f "${SENS_SWEEP_PY}" ]; then
      echo
      echo ">>> Sensitivity sweep (Stage 4a): ${RUN_DIR}"
      Exec-Py "${SENS_SWEEP_PY}" --run_dir "${RUN_DIR}" --grid_sim "${SENS_GRID_SIM}" --grid_mar "${SENS_GRID_MAR}" --max_outliers "${SENS_MAX_OUTLIERS}"
    else
      echo "(sweep) Skipping: ${SENS_SWEEP_PY} not found."
    fi
  fi
done

# =============================================================================
# EXPERIMENT 3: PURE-LLM BASELINE (TopicGPTv3)
# =============================================================================

# Configuration for TopicGPTv3
# Using the same embedding model as the main pipeline for fair comparison
LLM_BASELINE_ST_MODEL="${EMBED_MODEL_MAIN:-local_bge-large-en-v1.5}"
LLM_BASELINE_MAX_DOCS="${LLM_BASELINE_MAX_DOCS:-1000}"  # Limit discovery to 1k docs
LLM_BASELINE_MIN_FREQ="${MIN_CLUSTER_DOCS:-3}"
LLM_BASELINE_SIM_THR="${LLM_LABEL_MERGE_THR:-0.85}"

# Explicitly set the script name (override the top of file if needed)
PIPELINE_PURE_LLM_PY="02_pure_llm_baseline.py"

for SEED in "${SEEDS[@]}"; do
  RUN_DIR="run_llm_v${SEED}"
  echo
  echo "==============================="
  echo "PURE-LLM baseline (TopicGPT) | SEED=${SEED}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "==============================="

  # Ensure preprocessed file exists (TopicGPT needs it)
  PREPROC="${RUN_DIR}/01_preprocessed_documents.csv"
  if [ ! -f "$PREPROC" ]; then
      echo "Generating preprocessed data for LLM run..."
      # Use main pipeline just to preprocess if missing
      Exec-Py "${PIPELINE_PY}" --data "${DATA_CSV}" --run_dir "${RUN_DIR}" --max_outliers 0 --reassign_sim_threshold 1.0 --seed "${SEED}" --skip_cso > /dev/null 2>&1 || true
  fi

  Exec-Py "${PIPELINE_PURE_LLM_PY}" \
    --preproc_csv "${PREPROC}" \
    --run_dir "${RUN_DIR}" \
    --seed "${SEED}" \
    --model "${LLM_MODEL}" \
    --url "${OLLAMA_URL}" \
    --st_model "${LLM_BASELINE_ST_MODEL}" \
    --st_sim_threshold "${LLM_BASELINE_SIM_THR}" \
    --topic_min_freq "${LLM_BASELINE_MIN_FREQ}" \
    --gen_max_docs "${LLM_BASELINE_MAX_DOCS}" \
    --max_topics "150"

  # Run standard evaluation
  run_eval "${RUN_DIR}" "${SEED}" "llm"

  # Run trends analysis (filtering tiny topics)
  run_trends_if_possible "${RUN_DIR}" "${TRENDS_STABILITY_RUNS}"

  # Generate report
  run_report_if_possible "${RUN_DIR}" "${SEED}"
done

# =============================================================================
# FINAL: GENERATE COMPARISON REPORT (ALL SEEDS AGGREGATED)
# =============================================================================
echo
echo ">>> Generating Master Comparison Report..."

# FIX: Aggregate ALL seeds for robust comparison
RUNS_ARGS=()

# Accumulate MAIN runs
for SEED in "${SEEDS[@]}"; do
  RUNS_ARGS+=( "main=run_main_v${SEED}" )
done

# Accumulate BARE runs
for SEED in "${SEEDS[@]}"; do
  RUNS_ARGS+=( "bare=run_bare_v${SEED}" )
done

# Accumulate LLM runs
for SEED in "${SEEDS[@]}"; do
  RUNS_ARGS+=( "llm=run_llm_v${SEED}" )
done

Exec-Py "03_evaluate_topic_model.py" \
  --out_dir "final_thesis_comparison" \
  --runs "${RUNS_ARGS[@]}" \
  --top_n 20 \
  --seed "${SEEDS[0]}"

echo
echo "✅ All runs complete."
