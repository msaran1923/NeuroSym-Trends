# NeuroSym-Trends: Neuro-Symbolic, Uncertainty-Gated Topic Discovery & Trend Analysis

Reference implementation for the paper **“An Uncertainty-Gated Neuro-Symbolic Framework for High-Coverage Topic Modeling and Trend Analysis in Scholarly Corpora with LLM Assistance.”**  
The pipeline combines **BERTopic-style neural clustering**, **Computer Science Ontology (CSO)** signals, and **selective LLM gating** to recover low-density documents while keeping expensive generative steps bounded.

---

## What this repo contains

This repository is **script-first**: you run one bash runner (recommended) or invoke the Python scripts directly.

### Core scripts

- `00_bare_bertopic_baseline.py` — Bare BERTopic baseline (embeddings → UMAP → HDBSCAN → c-TF-IDF).
- `01_thesis_topic_trends_pipeline.py` — Main neuro-symbolic pipeline (CSO retrieval + entropy signal + outlier governance + optional LLM gating + augmentation + splitting + trend/shift analysis).
- `02_pure_llm_baseline.py` — Pure-LLM baseline (TopicGPT-style label proposal/merging + constrained assignment with **label + doc-specific description + supporting quote**; robust JSON parsing with retry).
- `03_evaluate_topic_model.py` — Computes intrinsic metrics & exports figures/tables from a completed run directory.
- `04_trends_only.py` — Re-runs the trend/shift stage for an existing run (useful when you adjust trend settings only).
- `05_sensitivity_sweep.py` — Replays Stage 4a (centroid reassignment) on saved artifacts over a grid of thresholds.

### Runner (recommended)

- `run_all.sh` — End-to-end runner that executes **Main / Bare / Pure-LLM** across **5 fixed seeds**, then runs evaluation, trend-only diagnostics, and (optionally) the sensitivity sweep.

---

## Data format

Your input dataset **must** be a CSV with these columns:

- `thesis_type` (e.g., `Master`, `PhD`)
- `english_title`
- `abstract_en`
- `year` (integer)

If `doc_id` is not present, the pipeline will create a stable hash-based ID deterministically from the fields above.

---

## CSO v3.5 (ontology file)

The pipeline expects a Turtle ontology file (default name: `CSO.3.5.ttl`).

**Download:** CSO provides v3.5 on its downloads page:
- https://cso.kmi.open.ac.uk/downloads  (select **v3.5** → **Download .ttl**)

After downloading, place it at repo root (or anywhere) and pass the path via `--cso_ttl` (or as the 2nd arg to `run_all.sh`).  
If the downloaded filename differs, rename it:

```bash
mv <downloaded_file>.ttl CSO.3.5.ttl
```

---

## Installation

### Python + core deps

- Python **3.11+** recommended
- Key libraries used by the scripts:
  - `pandas`, `numpy`
  - `sentence-transformers` (+ `torch`)
  - `umap-learn`, `hdbscan`
  - `bertopic`
  - `scikit-learn`
  - `gensim` (coherence)
  - `rdflib`, `faiss-*` (CSO indexing / retrieval)

> Tip: FAISS can be installed as `faiss-cpu` (CPU) or `faiss-gpu` (CUDA). Use the one matching your environment.

### Ollama (for LLM gating / Pure-LLM)

If you use `--llm_mode ollama` (Main) or run the Pure-LLM baseline, install Ollama and pull the model:

```bash
ollama pull qwen2.5:14b
# ensure the server is reachable at the URL you will pass (default below)
```

---

## Quick start (recommended): run the full experiment suite

```bash
# 1) Put your dataset CSV in place
# 2) Download CSO v3.5 (.ttl) and save as CSO.3.5.ttl

bash run_all.sh data.csv CSO.3.5.ttl
```

This will create run folders such as:
- `run_bare_v32`, `run_main_v32`, `run_pure_llm_v32` (and likewise for seeds 33–36)

Each run directory stores intermediate artifacts and final exports (topics, doc assignments, trend outputs, figures, JSON reports).

### Common runner overrides (environment variables)

`run_all.sh` is fully configurable via env vars. Examples:

```bash
# change embedding model
EMBED_MODEL_MAIN="local_bge-large-en-v1.5" bash run_all.sh data.csv CSO.3.5.ttl

# enable sensitivity sweep for Stage 4a (centroid reassignment replay)
SENS_SWEEP=1 bash run_all.sh data.csv CSO.3.5.ttl

# disable LLM gating in Main (still runs the rest)
LLM_MODE="none" bash run_all.sh data.csv CSO.3.5.ttl

# point Ollama to a remote server
OLLAMA_URL="http://127.0.0.1:11434" bash run_all.sh data.csv CSO.3.5.ttl
```

---

## Running scripts individually

### 1) Bare BERTopic baseline

```bash
python3 00_bare_bertopic_baseline.py \
  --data data.csv \
  --run_dir run_bare_v32 \
  --seed 32 \
  --embed_model local_bge-large-en-v1.5
```

### 2) Main neuro-symbolic pipeline

```bash
python3 01_thesis_topic_trends_pipeline.py \
  --data data.csv \
  --run_dir run_main_v32 \
  --seed 32 \
  --cso_ttl CSO.3.5.ttl \
  --llm_mode ollama \
  --llm_model qwen2.5:14b
```

To disable LLM gating (fully deterministic + symbolic only):

```bash
python3 01_thesis_topic_trends_pipeline.py \
  --data data.csv \
  --run_dir run_main_v32 \
  --seed 32 \
  --cso_ttl CSO.3.5.ttl \
  --llm_mode none
```

### 3) Pure-LLM baseline

```bash
python3 02_pure_llm_baseline.py \
  --data data.csv \
  --run_dir run_pure_llm_v32 \
  --seed 32 \
  --ollama_url http://localhost:11434 \
  --llm_model qwen2.5:14b
```

### 4) Evaluation (metrics + figures)

```bash
python3 03_evaluate_topic_model.py --run_dir run_main_v32 --seed 32
```

### 5) Trends-only re-run

```bash
python3 04_trends_only.py \
  --run_dir run_main_v32 \
  --pipeline_py 01_thesis_topic_trends_pipeline.py \
  --preproc_csv run_main_v32/01_preprocessed_documents.csv \
  --embed_model local_bge-large-en-v1.5 \
  --batch_size 512
```

### 6) Stage 4a sensitivity sweep

```bash
python3 05_sensitivity_sweep.py \
  --run_dir run_main_v32 \
  --grid_sim "0.65,0.70,0.75,0.80" \
  --grid_mar "0.0,0.001,0.002,0.005,0.01,0.02"
```

---

## Outputs (what to expect in a run directory)

A completed run directory typically includes:

- `01_preprocessed_documents.csv` — cleaned text + stable `doc_id`
- `embeddings.npy` — document embeddings
- `04_bertopic_doc_topics.csv` — document topics after the neural discovery stage
- `09_final_topics.csv` / `09_final_doc_topics.csv` — final topics and assignments after recovery/augmentation/splitting
- `10_topic_trends.csv` / `10_emerging_topics.csv` — trend test outputs (relative prevalence by default)
- `11_semantic_shift.csv` — semantic drift / centroid shift summaries
- `12_cso_concept_trends.csv` — concept-level (multi-label) trend companion outputs
- `14_topic_model_evaluation.json` and figures produced by `03_evaluate_topic_model.py`

Exact filenames may vary slightly across scripts; the runner prints paths as it executes.

---

## Reproducibility notes

- The default runner uses **5 fixed seeds** (32–36).  
- Each script writes artifacts deterministically under its `run_dir`.  
- Pure-LLM uses strict JSON schema validation with automatic retry/self-correction to reduce malformed outputs.

---

## Citation

```bibtex
@article{neurosymtrends2026,
  title={An Uncertainty-Gated Neuro-Symbolic Framework for High-Coverage Topic Modeling and Trend Analysis in Scholarly Corpora with LLM Assistance},
  author={Onur Demir and Murat Saran},
  journal={},
  year={2026}
}
```

---

## License

MIT (or your chosen license). See `LICENSE`.

---

## Acknowledgements

- CSO team for the Computer Science Ontology
- BERTopic / UMAP / HDBSCAN open-source authors
- Ollama for local LLM inference

Official implementation of a Neuro-Symbolic, Uncertainty-Gated Framework for topic discovery and trend analysis in scientific corpora. Combines BERTopic, CSO, and efficient LLM gating. This repository contains the source code and experimental data for the paper **"An Uncertainty-Gated Neuro-Symbolic Framework for High-Coverage Topic Modeling and Trend Analysis in Scholarly Corpora with LLM Assistance"**.

## 📋 Overview

Our framework addresses the **"symbolic disconnect"** in neural topic models by synergizing BERTopic with the **Computer Science Ontology (CSO)**. We introduce a novel **Uncertainty-Gated Augmentation** mechanism that selectively employs Large Language Models (LLMs) to recover outlier documents based on ontological entropy. This approach significantly improves topic coverage and trend sensitivity without the prohibitive cost of fully generative pipelines.

## ✨ Key Features

- **Neuro-Symbolic Integration**: Aligns neural clusters with structured domain knowledge from the Computer Science Ontology (CSO)
- **Cost-Effective LLM Gating**: Reduces inference costs by 40-60% by targeting only high-entropy outliers for LLM processing
- **Advanced Trend Analysis**: Includes comprehensive statistical methods for longitudinal bibliometric insights:
  - Mann-Kendall trend tests
  - Sen's Slope estimators
  - Kleinberg's burst detection algorithm
- **Uncertainty-Aware Assignment**: Explicit abstention mechanism for documents with low topic confidence
- **Dynamic Topic Tracking**: Monitors topic evolution over time with temporal slicing techniques

## 📁 Repository Structure

```
├── src/                   # Source code
├── results/               # Experimental outputs and figures
├── environment.yml        # Conda environment specification
└── README.md              # This file
```

## ⚙️ Installation

### Prerequisites
- Python 3.11.9
- CUDA 12.1 (for GPU acceleration)
- conda 24.1.2 or higher

### Setup Environment

1. Clone the repository:
```bash
git clone https://github.com/[username]/neuro-symbolic-topic-analysis.git
cd neuro-symbolic-topic-analysis
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate neuro-symbolic-topic
```

3. Install spaCy language model:
```bash
python -m spacy download en_core_web_lg
```

## 🚀 Quick Start

### Basic Usage

```python
from src.modeling.neuro_symbolic_pipeline import NeuroSymbolicPipeline

# Initialize the pipeline
pipeline = NeuroSymbolicPipeline(
    corpus_path="data/processed/corpus.csv",
    cso_path="data/external/cso.csv",
    output_dir="results/experiment_1"
)

# Run complete analysis
results = pipeline.run_full_analysis(
    temporal_slices=True,
    enable_llm_gating=True,
    trend_analysis=True
)

# Generate visualizations
pipeline.visualize_trends(results)
```

### Running Experiments

Execute the main experimental pipeline:
```bash
python src/experiments/main.py --config configs/experiment_config.yaml
```

### Reproducing Paper Results

To reproduce the exact results from the paper:
```bash
python src/experiments/reproduce_paper_results.py --seed 32
```

## 📊 Dataset

The framework was evaluated on a corpus of **12,535 computer science theses** (2001-2025) obtained from Türkiye's National Thesis Center. Due to copyright restrictions, the raw dataset cannot be shared, but:

- **Processed features** and **topic assignments** are available in `results/`
- **Statistical results** and **trend analysis outputs** are included in `results/`

## 📈 Results

Key findings from our experiments:

1. **Topic Coverage Improvement**: +28% coverage compared to baseline BERTopic
2. **Cost Efficiency**: 55% reduction in LLM API calls via uncertainty gating
3. **Trend Sensitivity**: Improved detection of emerging research areas with p < 0.01
4. **Interpretability**: Human-evaluated topic labels show 89% coherence score

Detailed results are available in the `results/` directory and in the paper.

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@article{saran2026,
  title={An Uncertainty-Gated Neuro-Symbolic Framework for High-Coverage Topic Modeling and Trend Analysis in Scholarly Corpora with LLM Assistance},
  author={Onur Demir and Murat Saran},
  journal={},
  year={2026},
  publisher={}
}
```

## 🔬 Extending the Framework

### Adding New Ontologies

```python
from src.modeling.ontology_integration import OntologyIntegrator

# Integrate a custom ontology
integrator = OntologyIntegrator(
    ontology_file="path/to/your_ontology.ttl",
    ontology_format="turtle"
)
pipeline.set_ontology_integrator(integrator)
```

### Custom Trend Detection Methods

```python
from src.analysis.trend_detection import TrendAnalyzer

# Add custom trend detection method
analyzer = TrendAnalyzer()
analyzer.add_custom_method(
    method_name="my_trend_test",
    function=my_custom_trend_function
)
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- Computer Science Ontology (CSO) team for their comprehensive knowledge graph
- Hugging Face for transformer models and libraries
- Maarten Grootendorst for BERTopic
- Türkiye Council of Higher Education for thesis data access

## 📬 Contact

For questions or collaborations, please contact:
- Murat Saran [saran@cankaya.edu.tr](mailto:saran@cankaya.edu.tr)

---

*Note: This framework is research software and may require adaptation for production use.*
