# NER & RE Evaluation Benchmark

A comprehensive benchmarking toolkit for comparing Named Entity Recognition (NER) and Relation Extraction (RE) models on scientific text. This project evaluates both open-source and LLM-based extraction methods with standardized metrics.

## 🎯 Overview

This benchmark compares multiple extraction approaches:
- **GLiNER1** - Original GLiNER model
- **GLiNER2** - Fast, open-source zero-shot model
- **Neo4j GraphRAG** - LLM-based extraction (GPT-4o)
- **LangExtract** - Structured LLM extraction (NER only)

Evaluated on scientific and entertainment domain datasets with comprehensive entity and relation types.

## 📊 Results Summary

### Named Entity Recognition - Science Dataset (500+ samples)
| Model | Precision | Recall | F1 | Avg Time/Sample |
|-------|-----------|--------|-----|-----------------|
| **LangExtract** | 0.714 | 0.619 | 0.663 | 1.39s |
| **GLiNER1** | 0.666 | 0.617 | 0.641 | 0.06s |
| **Neo4j GraphRAG** | 0.583 | 0.542 | 0.562 | 2.79s |
| **GLiNER2** | 0.475 | 0.562 | 0.515 | 0.06s |

### Named Entity Recognition - Comics Dataset (66 samples)
| Model | Precision | Recall | F1 | Avg Time/Sample |
|-------|-----------|--------|-----|-----------------|
| **Neo4j GraphRAG** | 0.575 | 0.798 | 0.668 | 2.98s |
| **GLiNER1** | 0.519 | 0.723 | 0.604 | 0.05s |
| **LangExtract** | 0.502 | 0.734 | 0.596 | 1.44s |
| **GLiNER2** | 0.476 | 0.750 | 0.583 | 0.05s |

### Relation Extraction (RE)
| Model | Precision | Recall | F1 | Avg Time/Sample |
|-------|-----------|--------|-----|-----------------|
| **Neo4j GraphRAG** | 0.547 | 0.537 | 0.542 | 2.56s |
| **GLiNER2** | 0.290 | 0.427 | 0.363 | 0.14s |
| **GLiNER1** | 0.122 | 0.041 | 0.062 | 0.16s |

**Key Findings:**
- **LangExtract leads NER** on science (F1: 0.663), but **GLiNER1 close behind** (F1: 0.641) at 23x speed
- **Neo4j GraphRAG best on comics** (F1: 0.668) and **relation extraction** (F1: 0.542)
- **GLiNER1 vs GLiNER2**: GLiNER1 significantly better for NER (+0.13 F1), worse for RE
- **Speed**: GLiNER models 20-50x faster than LLM-based approaches

---

## 🚀 Quick Start

### Prerequisites
- Python 3.13
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key (for Neo4j GraphRAG and LangExtract)

### Installation

```bash
# Clone the repository
git clone AmirLayegh/ner-eval
cd ner-eval

# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Running Benchmarks

```bash
# Named Entity Recognition - Science Dataset (500+ samples)
uv run python experiments/run_ner_benchmark.py

# Named Entity Recognition - Comics Dataset (66 samples)
uv run python experiments/run_comics_ner_benchmark.py

# Relation Extraction
uv run python experiments/run_re_benchmark.py

# Visualize all results
uv run python visualize_results.py
```
Results saved to `result/` directory with detailed metrics and timing.

#### Test GLiNER Interactively
```bash
uv run python test_gliner_open.py
```
Quick test of GLiNER extraction on sample text.

## 📁 Project Structure

```
ner-eval/
├── src/
│   ├── extractors/          # Model implementations
│   │   ├── base.py          # Abstract base classes
│   │   ├── gliner1.py       # GLiNER1 extractor
│   │   ├── gliner2.py       # GLiNER2 extractor
│   │   ├── neo4j_graphrag.py # Neo4j GraphRAG extractor
│   │   └── langextract.py   # LangExtract extractor
│   ├── evaluation/          # Evaluation logic
│   │   └── metrics.py       # Precision, recall, F1 calculation
│   ├── data/
│   │   └── loader.py        # Benchmark data loading
│   ├── runner.py            # Benchmark execution
│   └── models.py            # Pydantic data models
├── datasets/                # Benchmark datasets
│   ├── ner/                 # NER benchmarks
│   │   ├── science_ner_benchmark.json (500+ samples, 17 types)
│   │   └── comics_ner_benchmark.json (66 samples, 10 types)
│   └── re/                  # RE benchmarks
│       └── scientist_re_benchmark.json
├── experiments/             # Standalone benchmark scripts
└── result/                  # Benchmark results
```

## 🔧 Supported Models

### Relation Extraction (RE)
- **GLiNER1** (`knowledgator/gliner-relex-large-v0.5`)
- **GLiNER2** (`fastino/gliner2-base-v1`)
- **Neo4j GraphRAG** (with GPT-4o)

### Named Entity Recognition (NER)
- **GLiNER1** (`urchade/gliner_medium-v2.1`)
- **GLiNER2** (`fastino/gliner2-base-v1`)
- **Neo4j GraphRAG** (with GPT-4o)
- **LangExtract** (with GPT-4o)

## 📝 Adding Your Own Models

Extend the base classes to add new extractors:

```python
from src.extractors.base import BaseEntityExtractor
from src.models import Entity

class MyCustomExtractor(BaseEntityExtractor):
    @property
    def model_id(self) -> str:
        return "my-custom-model"
    
    def extract(self, text: str, entity_types: list[str]) -> list[Entity]:
        # Your extraction logic here
        return entities
```

## 📊 Datasets

### NER Datasets

**Science NER Benchmark**
- **Samples**: 500+ texts, 3000+ entities
- **Entity Types**: 17 types including scientist, astronomicalobject, chemicalcompound, protein, university, etc.
- **Domain**: Scientific/academic text
- **Source**: Scientific articles and Wikipedia

**Comics NER Benchmark**
- **Samples**: 66 texts, 188 entities
- **Entity Types**: 10 types including ComicsCharacter, Person, Organisation, Location, Film, etc.
- **Domain**: Comic books and entertainment
- **Source**: Comic character database

### RE Dataset

**Scientist RE Benchmark**
- **Samples**: 98 texts, 218 triples
- **Relation Types**: 20 types including almaMater, award, birthPlace, knownFor, professionalField, etc.
- **Domain**: Biographical scientist information
- **Source**: Filtered from ont_18_scientist_train dataset

## 🧪 Evaluation Metrics

All models are evaluated using:
- **Precision**: Correctness of predictions
- **Recall**: Coverage of ground truth
- **F1 Score**: Harmonic mean of precision and recall
- **Inference Time**: Average time per sample

Normalization: Case-insensitive, whitespace-stripped comparison.

## 🛠️ Configuration

### Environment Variables
Create a `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
```

Required only for Neo4j GraphRAG and LangExtract models (LLM-based). GLiNER1 and GLiNER2 work without any API keys.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional models (spaCy, Flair, etc.)
- More diverse datasets
- Additional evaluation metrics (entity-level, span-level)
- Performance optimizations

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- GLiNER1 by [urchade](https://huggingface.co/urchade) and [knowledgator](https://huggingface.co/knowledgator)
- GLiNER2 by [fastino](https://huggingface.co/fastino)
- Neo4j GraphRAG team
- LangExtract library
