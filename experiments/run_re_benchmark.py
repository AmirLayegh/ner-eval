"""
Relation Extraction Benchmark Runner

Runs GLiNER and Neo4j GraphRAG extractors on the benchmark dataset
and compares their performance.

Usage:
    uv run python experiments/run_re_benchmark.py
"""
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from src.data.loader import BenchmarkDataLoader
from src.extractors.gliner2 import GLiNER2RelationExtractor
from src.extractors.gliner1 import GLiNER1RelationExtractor
from src.extractors.neo4j_graphrag import Neo4jGraphRAGRelationExtractor
from src.runner import REBenchmarkRunner
from src.evaluation.metrics import REEvaluator
from src.models import EvaluationMetrics, REBenchmarkResult


def print_metrics(metrics: EvaluationMetrics) -> None:
    """Print evaluation metrics in a formatted way."""
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1:        {metrics.f1:.4f}")
    print(f"  TP: {metrics.true_positives}, FP: {metrics.false_positives}, FN: {metrics.false_negatives}")


def save_results(result: REBenchmarkResult, metrics: EvaluationMetrics, output_path: Path) -> None:
    """Save benchmark results and metrics to JSON."""
    output_data = {
        "model_id": result.model_id,
        "metrics": metrics.model_dump(),
        "timing": {
            "total_samples": result.total_samples,
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
    benchmark_path = "datasets/re/scientist_re_benchmark.json"
    output_dir = Path("result")
    output_dir.mkdir(exist_ok=True)
    
    # Load benchmark data
    print("=" * 60)
    print("Relation Extraction Benchmark")
    print("=" * 60)
    
    data_loader = BenchmarkDataLoader(benchmark_path)
    print(f"Loaded {len(data_loader)} samples from {benchmark_path}")
    print(f"Relation types: {data_loader.relation_types}")
    print()
    
    evaluator = REEvaluator()
    results_summary = []
    
    # --- GLiNER1 ---
    print("-" * 60)
    print("Running GLiNER1...")
    print("-" * 60)
    
    gliner1_extractor = GLiNER1RelationExtractor(model_id="knowledgator/gliner-relex-large-v0.5")
    gliner1_runner = REBenchmarkRunner(data_loader, gliner1_extractor)
    gliner1_result = gliner1_runner.run()
    gliner1_metrics = evaluator.evaluate(gliner1_result)
    
    print_metrics(gliner1_metrics)
    print(f"  Avg time/sample: {gliner1_result.average_time_per_sample:.4f}s")
    save_results(gliner1_result, gliner1_metrics, output_dir / "gliner1_re_results.json")
    results_summary.append(("GLiNER1", gliner1_metrics, gliner1_result.average_time_per_sample))
    print()
    
    # --- GLiNER ---
    print("-" * 60)
    print("Running GLiNER...")
    print("-" * 60)
    
    gliner_extractor = GLiNER2RelationExtractor(model_id="fastino/gliner2-base-v1")
    gliner_runner = REBenchmarkRunner(data_loader, gliner_extractor)
    gliner_result = gliner_runner.run()
    gliner_metrics = evaluator.evaluate(gliner_result)
    
    print_metrics(gliner_metrics)
    print(f"  Avg time/sample: {gliner_result.average_time_per_sample:.4f}s")
    save_results(gliner_result, gliner_metrics, output_dir / "gliner_re_results.json")
    results_summary.append(("GLiNER", gliner_metrics, gliner_result.average_time_per_sample))
    print()
    
    # --- Neo4j GraphRAG (uncomment to run) ---
    print("-" * 60)
    print("Running Neo4j GraphRAG...")
    print("-" * 60)
    
    neo4j_extractor = Neo4jGraphRAGRelationExtractor(model_id="gpt-4o")
    neo4j_runner = REBenchmarkRunner(data_loader, neo4j_extractor)
    neo4j_result = neo4j_runner.run()
    neo4j_metrics = evaluator.evaluate(neo4j_result)
    
    print_metrics(neo4j_metrics)
    print(f"  Avg time/sample: {neo4j_result.average_time_per_sample:.4f}s")
    save_results(neo4j_result, neo4j_metrics, output_dir / "neo4j_re_results.json")
    results_summary.append(("Neo4j GraphRAG", neo4j_metrics, neo4j_result.average_time_per_sample))
    print()
    
    # --- Summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Avg Time':<12}"
    print(header)
    print("-" * 68)
    
    for name, metrics, avg_time in results_summary:
        row = f"{name:<20} {metrics.precision:<12.4f} {metrics.recall:<12.4f} {metrics.f1:<12.4f} {avg_time:<12.4f}s"
        print(row)
    
    print()
    print("Benchmark complete!")


if __name__ == "__main__":
    main()