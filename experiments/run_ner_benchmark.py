"""
Named Entity Recognition Benchmark Runner

Runs GLiNER extractor on the NER benchmark dataset
and evaluates performance.

Usage:
    uv run python experiments/run_ner_benchmark.py
"""
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from src.data.loader import NERBenchmarkDataLoader
from src.extractors.gliner2 import GLiNER2EntityExtractor
from src.extractors.gliner1 import GLiNER1RelationExtractor, GLiNER1EntityExtractor
from src.extractors.neo4j_graphrag import Neo4jGraphRagEntityExtractor
from src.extractors.langextract import LangExtractEntityExtractor
from src.runner import NERBenchmarkRunner
from src.evaluation.metrics import NEREvaluator
from src.models import EvaluationMetrics, NERBenchmarkResult, Entity


def print_metrics(metrics: EvaluationMetrics) -> None:
    """Print evaluation metrics in a formatted way."""
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1:        {metrics.f1:.4f}")
    print(f"  TP: {metrics.true_positives}, FP: {metrics.false_positives}, FN: {metrics.false_negatives}")


def save_results(result: NERBenchmarkResult, metrics: EvaluationMetrics, output_path: Path) -> None:
    """Save benchmark results and metrics to JSON."""
    output_data = {
        "model_id": result.model_id,
        "metrics": metrics.model_dump(),
        "timing": {
            "total_samples": result.total_samples,
            "total_entities": result.total_entities,
            "total_time_seconds": result.total_time_seconds,
            "average_time_per_sample": result.average_time_per_sample,
        },
        "results": [r.model_dump() for r in result.results],
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Results saved to: {output_path}")


def main():
    # Configuration
    benchmark_path = "datasets/ner/science_ner_benchmark.json"
    output_dir = Path("result")
    output_dir.mkdir(exist_ok=True)
    
    # Load benchmark data
    print("=" * 60)
    print("Named Entity Recognition Benchmark")
    print("=" * 60)
    
    data_loader = NERBenchmarkDataLoader(benchmark_path)
    print(f"Loaded {len(data_loader)} samples from {benchmark_path}")
    print(f"Entity types: {data_loader.entity_types}")
    print()
    
    evaluator = NEREvaluator()
    results_summary = []
    
    # --- GLiNER ---
    print("-" * 60)
    print("Running GLiNER...")
    print("-" * 60)
    
    gliner_extractor = GLiNER2EntityExtractor(model_id="fastino/gliner2-base-v1")
    gliner_runner = NERBenchmarkRunner(data_loader, gliner_extractor)
    gliner_result = gliner_runner.run()
    gliner_metrics = evaluator.evaluate(gliner_result)
    
    print_metrics(gliner_metrics)
    print(f"  Avg time/sample: {gliner_result.average_time_per_sample:.4f}s")
    save_results(gliner_result, gliner_metrics, output_dir / "gliner_ner_results.json")
    results_summary.append(("GLiNER", gliner_metrics, gliner_result.average_time_per_sample))
    print()

    # --- GLiNER1 ---
    print("-" * 60)
    print("Running GLiNER1...")
    print("-" * 60)
    
    gliner1_extractor = GLiNER1EntityExtractor(model_id="urchade/gliner_medium-v2.1")
    gliner1_runner = NERBenchmarkRunner(data_loader, gliner1_extractor)
    gliner1_result = gliner1_runner.run()
    gliner1_metrics = evaluator.evaluate(gliner1_result)
    
    print_metrics(gliner1_metrics)
    print(f"  Avg time/sample: {gliner1_result.average_time_per_sample:.4f}s")
    save_results(gliner1_result, gliner1_metrics, output_dir / "gliner1_ner_results.json")
    results_summary.append(("GLiNER1", gliner1_metrics, gliner1_result.average_time_per_sample))
    print()

    # # --- Neo4j GraphRAG ---
    # print("-" * 60)
    # print("Running Neo4j GraphRAG...")
    # print("-" * 60)
    
    # neo4j_extractor = Neo4jGraphRagEntityExtractor(model_id="gpt-4o")
    # neo4j_runner = NERBenchmarkRunner(data_loader, neo4j_extractor)
    # neo4j_result = neo4j_runner.run()
    # neo4j_metrics = evaluator.evaluate(neo4j_result)
    
    # print_metrics(neo4j_metrics)
    # print(f"  Avg time/sample: {neo4j_result.average_time_per_sample:.4f}s")
    # save_results(neo4j_result, neo4j_metrics, output_dir / "neo4j_ner_results.json")
    # results_summary.append(("Neo4j GraphRAG", neo4j_metrics, neo4j_result.average_time_per_sample))
    # print()
    # --- LangExtract ---
    # print("-" * 60)
    # print("Running LangExtract...")
    # print("-" * 60)
    
    # lang_extractor = LangExtractEntityExtractor(model_id="gpt-4o")
    # lang_extractor._set_prompt(data_loader.entity_types)
    # lang_extractor._set_examples()
    # lang_runner = NERBenchmarkRunner(data_loader, lang_extractor)
    # lang_result = lang_runner.run()
    # lang_metrics = evaluator.evaluate(lang_result)
    # print_metrics(lang_metrics)
    # print(f"  Avg time/sample: {lang_result.average_time_per_sample:.4f}s")
    # save_results(lang_result, lang_metrics, output_dir / "lang_ner_results.json")
    # results_summary.append(("LangExtract", lang_metrics, lang_result.average_time_per_sample))
    # print()
    # # --- Summary ---
    # print("=" * 60)
    # print("SUMMARY")
    # print("=" * 60)
    # header = f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Avg Time':<12}"
    # print(header)
    # print("-" * 68)
    
    # for name, metrics, avg_time in results_summary:
    #     row = f"{name:<20} {metrics.precision:<12.4f} {metrics.recall:<12.4f} {metrics.f1:<12.4f} {avg_time:<12.4f}s"
    #     print(row)
    
    # print()
    # print("Benchmark complete!")


if __name__ == "__main__":
    main()