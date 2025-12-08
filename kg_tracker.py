"""Example: Using neo4j_experiment_tracker for NER evaluation experiments.

This script demonstrates how to use the experiment tracker for local development.
Run with: python experiment_example.py
"""

import json
import logging
import time
from pathlib import Path
from src.gliner import GliNER2RE
from evaluation import GliNER2REEvaluator

from dotenv import load_dotenv
load_dotenv()


# Configure logging to see NoOpTracker output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from neo4j_experiment_tracker import create_tracker, TrackerType, RunStatus


def simulate_ner_evaluation(model_name: str, dataset: str, output_file: str, input_path: str, ) -> dict:
    gliner = GliNER2RE(file_path=input_path, output_file=output_file)
    gliner_results, gliner_time = gliner.run()
    relation_types = gliner.relation_types
    evaluator = GliNER2REEvaluator(output_file=output_file)
    results = evaluator.evaluate()

    return {
        "true_positive": results["true_positive"],
        "false_positive": results["false_positive"],
        "false_negative": results["false_negative"],
        "relation_types": relation_types,
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"],
        "avg_time_per_sample": gliner_time,
        # "total_samples": len(gliner_results["results"]), # TODO: add total samples
    }
    

def main():
    # Create output directory for artifacts
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # ==========================================================================
    # OPTION 1: Using tracker_type="none" for local development (logs to console)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Running NER Evaluation with NoOp Tracker (Local Development)")
    print("=" * 60 + "\n")
    
    tracker = create_tracker(
        tracker_type=TrackerType.NONE,  # or simply "none"
        experiment_name="ner-evaluation",
        run_id=f"gliner2-base-v1-scientist-re-benchmark",
        tags={
            "task": "named-entity-recognition",
            "environment": "local",
        },
    )
    
    # Use context manager for automatic start/end handling
    with tracker:
        # 1. Log experiment parameters (hyperparameters, config, etc.)
        tracker.log_params({
            "model_name": "gliner2-base-v1",
            "dataset": "scientist_re_benchmark",
        })
        
        # 2. Run your evaluation
        print("Running NER evaluation...")
        results = simulate_ner_evaluation(
            model_name="gliner2-base-v1",
            dataset="scientist_re_benchmark",
            output_file="result/gliner_re_results.json",
            input_path="data/re/scientist_re_benchmark.json"
        )
        
        # 3. Log metrics
        tracker.log_metrics({
            "true_positive": results["true_positive"],
            "false_positive": results["false_positive"],
            "false_negative": results["false_negative"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
            "avg_time_per_sample": results["avg_time_per_sample"],
        })
        
        # 5. Save and log artifacts
        results_file = output_dir / "gliner_re_logs.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        tracker.log_artifact(str(results_file), artifact_type="gliner_re_logs")
    
    print(f"\nResults saved to: {results_file}\n")
    print("Experiment completed successfully!\n")
    


if __name__ == "__main__":
    main()

