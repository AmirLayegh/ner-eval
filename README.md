# NER & RE Evaluation Benchmark

Benchmarking toolkit for Named Entity Recognition (NER) and Relation Extraction (RE) models.

## Quick Start

```bash
# Install dependencies
uv sync
```

## Running Benchmarks

### With Experiment Tracking (neo4j-experiment-tracker)

```bash
uv run python experiment_tracker.py
```

Logs params, metrics, and artifacts to the tracker.

### Without Tracking

```bash
# Relation Extraction
uv run python experiments/run_re_benchmark.py

# Named Entity Recognition
uv run python experiments/run_ner_benchmark.py
```

## Supported Models

| Task | Models |
|------|--------|
| RE | GLiNER1, GLiNER2, Neo4j GraphRAG |
| NER | GLiNER1, GLiNER2, Neo4j GraphRAG, LangExtract |

## Project Structure

```
src/
├── extractors/     # Model implementations
├── evaluation/     # Metrics (precision, recall, F1)
├── runner.py       # Benchmark runners
└── models.py       # Pydantic data models
datasets/           # Benchmark data
experiments/        # Standalone benchmark scripts
```
