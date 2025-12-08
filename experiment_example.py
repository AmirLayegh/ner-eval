"""Example: Using neo4j_experiment_tracker for NER evaluation experiments.

This script demonstrates how to use the experiment tracker for local development.
Run with: python experiment_example.py
"""

import json
import logging
import time
from pathlib import Path

# Configure logging to see NoOpTracker output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from neo4j_experiment_tracker import create_tracker, TrackerType, RunStatus


def simulate_ner_evaluation(model_name: str, dataset: str) -> dict:
    """Simulate NER evaluation - replace with your actual evaluation logic."""
    time.sleep(0.5)  # Simulate processing time
    
    # These would be your actual evaluation results
    return {
        "precision": 0.87,
        "recall": 0.82,
        "f1_score": 0.845,
        "accuracy": 0.91,
        "entity_types_evaluated": 5,
        "total_samples": 1000,
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
        run_id=f"ner-eval-{time.strftime('%Y%m%d-%H%M%S')}",
        tags={
            "task": "named-entity-recognition",
            "environment": "local",
        },
    )
    
    # Use context manager for automatic start/end handling
    with tracker:
        # 1. Log experiment parameters (hyperparameters, config, etc.)
        tracker.log_params({
            "model_name": "bert-base-cased",
            "dataset": "conll2003",
            "max_seq_length": 512,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "entity_types": ["PER", "ORG", "LOC", "MISC"],
            "preprocessing": "lowercase_normalize",
        })
        
        # 2. Run your evaluation
        print("Running NER evaluation...")
        results = simulate_ner_evaluation(
            model_name="bert-base-cased",
            dataset="conll2003"
        )
        
        # 3. Log metrics
        tracker.log_metrics({
            "precision": results["precision"],
            "recall": results["recall"],
            "f1_score": results["f1_score"],
            "accuracy": results["accuracy"],
        })
        
        # 4. Log step-wise metrics (useful for multi-epoch training or batch processing)
        for step in range(1, 4):
            tracker.log_metrics(
                {"batch_f1": 0.80 + step * 0.02},
                step=step
            )
        
        # 5. Save and log artifacts
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        tracker.log_artifact(str(results_file), artifact_type="evaluation_results")
        
        # 6. Set additional tags during the run
        tracker.set_tags({
            "status": "completed",
            "run_type": "evaluation",
        })
    
    print(f"\nResults saved to: {results_file}")
    print("Experiment completed successfully!\n")
    
    # ==========================================================================
    # OPTION 2: Manual start/end without context manager
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Alternative: Manual Start/End (with error handling)")
    print("=" * 60 + "\n")
    
    tracker2 = create_tracker(
        tracker_type="none",
        experiment_name="ner-evaluation-manual",
        run_id=f"ner-manual-{time.strftime('%Y%m%d-%H%M%S')}",
    )
    
    try:
        tracker2.start_run()
        
        tracker2.log_params({"model": "custom-ner-model"})
        tracker2.log_metrics({"f1_score": 0.88})
        
        # Simulate success
        tracker2.end_run(status=RunStatus.COMPLETED)
        
    except Exception as e:
        # Mark run as failed if something goes wrong
        tracker2.end_run(status=RunStatus.FAILED)
        raise
    
    print("\nManual run completed!\n")


if __name__ == "__main__":
    main()

