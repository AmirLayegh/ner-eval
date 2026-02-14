"""
Visualize benchmark results from JSON files.

Usage:
    uv run python visualize_results.py
"""
import json
from pathlib import Path
from typing import Dict, List


def load_results(result_dir: Path) -> Dict[str, Dict]:
    """Load all result JSON files from directory."""
    results = {}
    
    for json_file in result_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            stem = json_file.stem
            
            # Determine task and dataset
            if "ner" in stem:
                if "comics" in stem:
                    task = "NER-Comics"
                    dataset = "Comics"
                else:
                    task = "NER-Science"
                    dataset = "Science"
            else:
                task = "RE"
                dataset = "Scientist"
            
            # Extract model name
            model_name = stem.replace("_ner_results", "").replace("_re_results", "").replace("_comics", "")
            
            key = f"{model_name}_{task}"
            results[key] = {
                "model_id": data.get("model_id", "unknown"),
                "metrics": data.get("metrics", {}),
                "timing": data.get("timing", {}),
                "task": task,
                "dataset": dataset,
                "model_name": model_name
            }
    
    return results


def print_comparison_table(results: Dict[str, Dict], task: str, title: str = None):
    """Print comparison table for a specific task."""
    task_results = {k: v for k, v in results.items() if v["task"] == task}
    
    if not task_results:
        print(f"No results found for {task}")
        return
    
    print(f"\n{'=' * 80}")
    print(title or f"{task} Benchmark Results Comparison")
    print(f"{'=' * 80}")
    
    # Header
    header = f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Avg Time':<12} {'Samples':<10}"
    print(header)
    print("-" * 80)
    
    # Sort by F1 score descending
    sorted_results = sorted(
        task_results.items(),
        key=lambda x: x[1]["metrics"].get("f1", 0),
        reverse=True
    )
    
    for key, result in sorted_results:
        metrics = result["metrics"]
        timing = result["timing"]
        
        model_name = result["model_name"].replace("_", " ").title()
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        f1 = metrics.get("f1", 0)
        avg_time = timing.get("average_time_per_sample", 0)
        samples = timing.get("total_samples", 0)
        
        row = f"{model_name:<20} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {avg_time:<12.4f}s {samples:<10}"
        print(row)
    
    print()


def print_detailed_metrics(results: Dict[str, Dict]):
    """Print detailed metrics for each model."""
    print(f"\n{'=' * 80}")
    print("Detailed Metrics")
    print(f"{'=' * 80}\n")
    
    for key, result in sorted(results.items()):
        model_name = result["model_name"].replace("_", " ").title()
        task = result["task"]
        metrics = result["metrics"]
        timing = result["timing"]
        
        # Calculate total ground truth items
        tp = metrics.get('true_positives', 0)
        fn = metrics.get('false_negatives', 0)
        total_gt = tp + fn
        
        print(f"{model_name} ({task})")
        print("-" * 40)
        print(f"  Model ID: {result['model_id']}")
        print(f"  Dataset: {result.get('dataset', 'Unknown')}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0):.4f}")
        print(f"  F1 Score: {metrics.get('f1', 0):.4f}")
        print(f"  True Positives: {tp}")
        print(f"  False Positives: {metrics.get('false_positives', 0)}")
        print(f"  False Negatives: {fn}")
        print(f"  Total Ground Truth: {total_gt} {'entities' if 'NER' in task else 'triples'}")
        print(f"  Total Samples: {timing.get('total_samples', 0)} texts")
        print(f"  Total Time: {timing.get('total_time_seconds', 0):.2f}s")
        print(f"  Avg Time/Sample: {timing.get('average_time_per_sample', 0):.4f}s")
        print()


def generate_markdown_table(results: Dict[str, Dict], task: str) -> str:
    """Generate markdown table for README."""
    task_results = {k: v for k, v in results.items() if v["task"] == task}
    
    if not task_results:
        return f"No results found for {task}"
    
    # Sort by F1 score descending
    sorted_results = sorted(
        task_results.items(),
        key=lambda x: x[1]["metrics"].get("f1", 0),
        reverse=True
    )
    
    # Generate markdown
    lines = [
        f"### {task} Results",
        "",
        "| Model | Precision | Recall | F1 | Avg Time/Sample |",
        "|-------|-----------|--------|-----|-----------------|",
    ]
    
    for key, result in sorted_results:
        metrics = result["metrics"]
        timing = result["timing"]
        
        model_name = result["model_name"].replace("_", " ").title()
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        f1 = metrics.get("f1", 0)
        avg_time = timing.get("average_time_per_sample", 0)
        
        row = f"| **{model_name}** | {precision:.3f} | {recall:.3f} | {f1:.3f} | {avg_time:.2f}s |"
        lines.append(row)
    
    return "\n".join(lines)


def main():
    result_dir = Path("result")
    
    if not result_dir.exists():
        print(f"Error: {result_dir} directory not found")
        print("Run benchmarks first to generate results")
        return
    
    # Load all results
    results = load_results(result_dir)
    
    if not results:
        print("No result files found in result/ directory")
        return
    
    # Print comparison tables
    print_comparison_table(results, "NER-Science", "NER - Science Dataset (500+ samples)")
    print_comparison_table(results, "NER-Comics", "NER - Comics Dataset (66 samples)")
    print_comparison_table(results, "RE", "Relation Extraction - Scientist Dataset")
    
    # Print detailed metrics
    print_detailed_metrics(results)
    
    # Generate markdown for README
    print(f"\n{'=' * 80}")
    print("Markdown Tables for README")
    print(f"{'=' * 80}\n")
    print("### Science Dataset")
    print(generate_markdown_table(results, "NER-Science"))
    print()
    print("### Comics Dataset")
    print(generate_markdown_table(results, "NER-Comics"))
    print()
    print("### Relation Extraction")
    print(generate_markdown_table(results, "RE"))
    print()


if __name__ == "__main__":
    main()
