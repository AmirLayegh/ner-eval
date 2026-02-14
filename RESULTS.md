# Benchmark Results & Analysis

## Executive Summary

Evaluated four extraction approaches across two domains:
- **GLiNER1** - Original GLiNER model (urchade/gliner_medium-v2.1)
- **GLiNER2** - Newer GLiNER model (fastino/gliner2-base-v1)
- **Neo4j GraphRAG** - LLM-based extraction (GPT-4o)
- **LangExtract** - Structured LLM extraction (GPT-4o)

### Key Findings

1. **GLiNER1 strong for NER**: F1 0.641 (science) and 0.604 (comics) - competitive with LLMs at 20x speed
2. **GLiNER2 faster but less accurate**: Better for high-throughput, lower accuracy than GLiNER1
3. **Domain-dependent winners**: LangExtract (science), Neo4j GraphRAG (comics)
4. **RE requires LLMs**: Neo4j GraphRAG (F1: 0.531) vastly outperforms GLiNER models
5. **Speed vs Accuracy**: GLiNER models 20-50x faster than LLMs

---

## Named Entity Recognition Results

### Science Dataset (543 samples, 3043 entities, 17 types)

| Model | Precision | Recall | F1 | Speed |
|-------|-----------|--------|-----|-------|
| **LangExtract** | 0.712 | 0.615 | 0.660 | 1.99s |
| **GLiNER1** | 0.666 | 0.617 | 0.641 | 0.06s |
| **Neo4j GraphRAG** | 0.592 | 0.560 | 0.575 | 2.83s |
| **GLiNER2** | 0.475 | 0.562 | 0.515 | 0.06s |

**Analysis:**
- GLiNER1 nearly matches LangExtract (Δ 0.019) at 33x speed
- GLiNER1 significantly better than GLiNER2 (+0.126 F1)
- LangExtract has best precision (0.712), GLiNER1 balanced

### Comics Dataset (66 samples, 188 entities, 10 types)

| Model | Precision | Recall | F1 | Speed |
|-------|-----------|--------|-----|-------|
| **Neo4j GraphRAG** | 0.575 | 0.798 | 0.668 | 2.98s |
| **GLiNER1** | 0.519 | 0.723 | 0.604 | 0.05s |
| **LangExtract** | 0.502 | 0.734 | 0.596 | 1.44s |
| **GLiNER2** | 0.476 | 0.750 | 0.583 | 0.05s |

**Analysis:**
- Neo4j GraphRAG leads with highest recall (0.798)
- GLiNER1 beats LangExtract at 30x speed
- GLiNER1 vs GLiNER2: +0.021 F1, similar speed

### Cross-Dataset Comparison

| Model | Science F1 | Comics F1 | Avg F1 | Speed | Cost |
|-------|-----------|-----------|--------|-------|------|
| **GLiNER1** | 0.641 | 0.604 | 0.623 | 0.05s | Free |
| **LangExtract** | 0.660 | 0.596 | 0.628 | 1.70s | $$$ |
| **Neo4j GraphRAG** | 0.575 | 0.668 | 0.622 | 2.91s | $$$ |
| **GLiNER2** | 0.515 | 0.583 | 0.549 | 0.06s | Free |

**Key Insights:**
- **GLiNER1**: Best free option, consistent across domains (avg F1: 0.623)
- **LangExtract**: Slight edge on average but 34x slower
- **Neo4j GraphRAG**: Best on comics, struggles on science
- **GLiNER2**: Weakest NER performance

---

## Relation Extraction Results

### Scientist RE (98 samples, 218 triples, 20 types)

| Model | Precision | Recall | F1 | Speed |
|-------|-----------|--------|-----|-------|
| **Neo4j GraphRAG** | 0.530 | 0.532 | 0.531 | 2.37s |
| **GLiNER2** | 0.290 | 0.427 | 0.363 | 0.14s |
| **GLiNER1** | 0.122 | 0.041 | 0.062 | 0.16s |

**Analysis:**
- Neo4j GraphRAG is the only viable option (F1: 0.531)
- GLiNER models struggle with relation extraction
- GLiNER1 particularly poor (F1: 0.062) - not recommended for RE

---

## Model Comparison

### By Task

| Task | Best Model | F1 | Runner-up | F1 | Speed Winner |
|------|-----------|-----|-----------|-----|--------------|
| **Science NER** | LangExtract | 0.660 | GLiNER1 | 0.641 | GLiNER1/2 (0.06s) |
| **Comics NER** | Neo4j GraphRAG | 0.668 | GLiNER1 | 0.604 | GLiNER1 (0.05s) |
| **RE** | Neo4j GraphRAG | 0.531 | GLiNER2 | 0.363 | GLiNER2 (0.14s) |

### Speed vs Accuracy

| Model | Avg Speed | Avg F1 (NER) | Cost | Best For |
|-------|-----------|--------------|------|----------|
| **GLiNER1** | 0.05s | 0.623 | Free | Production NER, fast & accurate |
| **GLiNER2** | 0.06s | 0.549 | Free | High-throughput, lower accuracy OK |
| **LangExtract** | 1.70s | 0.628 | $$$ | Maximum NER accuracy |
| **Neo4j GraphRAG** | 2.91s | 0.622 | $$$ | RE, cross-domain NER |

---

## Recommendations

### Named Entity Recognition
- **Production/Real-time**: GLiNER1 (best free option, F1: 0.62+)
- **Maximum accuracy**: LangExtract (F1: 0.660) if cost acceptable
- **Cross-domain**: Neo4j GraphRAG (best generalization)

### Relation Extraction
- **Only viable option**: Neo4j GraphRAG (F1: 0.531)
- GLiNER models not recommended for RE

---

## Methodology

**Evaluation Metrics:**
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × (Precision × Recall) / (Precision + Recall)

**Matching:** Case-insensitive, whitespace-normalized, exact text match

**Datasets:**
- Science NER: 543 texts, 3043 entities, 17 types
- Comics NER: 66 texts, 188 entities, 10 types
- Scientist RE: 98 texts, 218 triples, 20 relation types

---

## Reproducibility

```bash
uv sync
uv run python experiments/run_ner_benchmark.py          # Science NER
uv run python experiments/run_comics_ner_benchmark.py   # Comics NER
uv run python experiments/run_re_benchmark.py           # RE
uv run python visualize_results.py                      # View all results
```

---

## Conclusion

**Model Selection:**
- **Need speed + accuracy?** → GLiNER1 (best free option)
- **Need maximum accuracy?** → LangExtract (marginal gain, high cost)
- **Need relations?** → Neo4j GraphRAG (only viable option)
- **Budget limited?** → GLiNER1 (97% of best accuracy, free)

**Key Takeaway**: GLiNER1 offers the best cost-performance ratio for NER tasks, achieving near-LLM accuracy at zero cost and 20-30x speed.

---

*Benchmark version: 0.1.0*
