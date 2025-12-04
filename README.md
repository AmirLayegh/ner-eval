# NER & RE Evaluation

Benchmarking NER and Relation Extraction models.

## Models
- **GLiNER2** - Fast local model
- **LangExtract** - LLM-based extraction
- **Neo4j GraphRAG** - LLM-based graph extraction

## Structure
```
├── src/                  # Model wrappers
│   ├── gliner.py
│   ├── langextract.py
│   └── neo4j_graphrag.py
├── data/                 # Benchmark datasets
│   ├── science_ner_benchmark.json
│   └── re/scientist_re_benchmark.json
├── eval_ner.py           # NER evaluation script
├── eval_re.py            # RE evaluation script
└── result/               # Evaluation results
```

## Usage
```bash
# NER evaluation
uv run python eval_ner.py

# Relation Extraction evaluation
uv run python eval_re.py
```

## Metrics
- Precision, Recall, F1 (micro-averaged)
- Average inference time per sample
