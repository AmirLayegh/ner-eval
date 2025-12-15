"""
Experiment Tracker Integration

Uses neo4j_experiment_tracker to log benchmark results for RE and NER tasks.
Run with: uv run python experiment_tracker.py
"""

import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from neo4j_experiment_tracker import create_tracker, TrackerType

from src.data.loader import BenchmarkDataLoader, NERBenchmarkDataLoader
from src.extractors.gliner2 import GLiNER2RelationExtractor, GLiNER2EntityExtractor
from src.extractors.neo4j_graphrag import Neo4jGraphRAGRelationExtractor, Neo4jGraphRagEntityExtractor
from src.runner import REBenchmarkRunner, NERBenchmarkRunner
from src.evaluation.metrics import REEvaluator, NEREvaluator
from src.models import REBenchmarkResult, NERBenchmarkResult, EvaluationMetrics


def run_re_benchmark(
    extractor_name: str,
    extractor,
    data_loader: BenchmarkDataLoader,
    evaluator: REEvaluator,
) -> tuple[REBenchmarkResult, EvaluationMetrics]:
    """Run relation extraction benchmark and return results."""
    runner = REBenchmarkRunner(data_loader, extractor)
    result = runner.run()
    metrics = evaluator.evaluate(result)
    return result, metrics


def run_ner_benchmark(
    extractor_name: str,
    extractor,
    data_loader: BenchmarkDataLoader,
    evaluator: NEREvaluator,
) -> tuple[NERBenchmarkResult, EvaluationMetrics]:
    """Run NER benchmark and return results."""
    runner = NERBenchmarkRunner(data_loader, extractor)
    result = runner.run()
    metrics = evaluator.evaluate(result)
    return result, metrics


def track_experiment(
    tracker,
    task: str,
    extractor_name: str,
    result,
    metrics: EvaluationMetrics,
    output_dir: Path,
):
    """Log experiment results to tracker and save artifacts."""
    # Log metrics
    tracker.log_metrics({
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "avg_time_per_sample": result.average_time_per_sample,
        "total_samples": result.total_samples,
        "total_time_seconds": result.total_time_seconds,
    })
    
    # Save results to file
    results_file = output_dir / f"{extractor_name}_{task}_results.json"
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
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    tracker.log_artifact(str(results_file), artifact_type=f"{extractor_name}_{task}_results")
    return results_file


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # ========================================
    # Relation Extraction Benchmark
    # ========================================
    print("\n" + "=" * 60)
    print("Running Relation Extraction Benchmark with Tracker")
    print("=" * 60 + "\n")
    
    re_data_loader = BenchmarkDataLoader("datasets/re/scientist_re_benchmark.json")
    re_evaluator = REEvaluator()
    
    # --- GLiNER RE ---
    tracker = create_tracker(
        tracker_type=TrackerType.NONE,
        experiment_name="relation-extraction",
        run_id="gliner-re-benchmark",
        tags={"task": "relation-extraction", "model": "gliner2-base-v1"},
    )
    
    with tracker:
        tracker.log_params({
            "model_name": "gliner2-base-v1",
            "dataset": "scientist_re_benchmark",
            "task": "relation-extraction",
        })
        
        print("Running GLiNER RE...")
        gliner_extractor = GLiNER2RelationExtractor(model_id="fastino/gliner2-base-v1")
        result, metrics = run_re_benchmark("gliner", gliner_extractor, re_data_loader, re_evaluator)
        
        results_file = track_experiment(tracker, "re", "gliner", result, metrics, output_dir)
        print(f"  Precision: {metrics.precision:.4f}, Recall: {metrics.recall:.4f}, F1: {metrics.f1:.4f}")
        print(f"  Results saved to: {results_file}")
    
    ## --- Neo4j GraphRAG RE ---
    # tracker = create_tracker(
    #     tracker_type=TrackerType.NONE,
    #     experiment_name="relation-extraction",
    #     run_id="neo4j-re-benchmark",
    #     tags={"task": "relation-extraction", "model": "gpt-4o"},
    # )
    # 
    # with tracker:
    #     tracker.log_params({
    #         "model_name": "gpt-4o",
    #         "dataset": "scientist_re_benchmark",
    #         "task": "relation-extraction",
    #     })
    #     
    #     print("\nRunning Neo4j GraphRAG RE...")
    #     neo4j_extractor = Neo4jGraphRAGRelationExtractor(model_id="gpt-4o")
    #     result, metrics = run_re_benchmark("neo4j", neo4j_extractor, re_data_loader, re_evaluator)
    #     
    #     results_file = track_experiment(tracker, "re", "neo4j", result, metrics, output_dir)
    #     print(f"  Precision: {metrics.precision:.4f}, Recall: {metrics.recall:.4f}, F1: {metrics.f1:.4f}")
    #     print(f"  Results saved to: {results_file}")
    
    # ========================================
    # NER Benchmark
    # ========================================
    print("\n" + "=" * 60)
    print("Running NER Benchmark with Tracker")
    print("=" * 60 + "\n")
    
    ner_data_loader = NERBenchmarkDataLoader("datasets/ner/science_ner_benchmark.json")
    ner_evaluator = NEREvaluator()
    
    # --- GLiNER NER ---
    tracker = create_tracker(
        tracker_type=TrackerType.NONE,
        experiment_name="named-entity-recognition",
        run_id="gliner-ner-benchmark",
        tags={"task": "ner", "model": "gliner2-base-v1"},
    )
    
    with tracker:
        tracker.log_params({
            "model_name": "gliner2-base-v1",
            "dataset": "science_ner_benchmark",
            "task": "named-entity-recognition",
        })
        
        print("Running GLiNER NER...")
        gliner_ner_extractor = GLiNER2EntityExtractor(model_id="fastino/gliner2-base-v1")
        result, metrics = run_ner_benchmark("gliner", gliner_ner_extractor, ner_data_loader, ner_evaluator)
        
        results_file = track_experiment(tracker, "ner", "gliner", result, metrics, output_dir)
        print(f"  Precision: {metrics.precision:.4f}, Recall: {metrics.recall:.4f}, F1: {metrics.f1:.4f}")
        print(f"  Results saved to: {results_file}")
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
