"""
Simple Relation Extraction evaluation script that calculates F1 scores for GLiNER2 and Neo4j GraphRAG.
Compares extracted triples (subject, relation, object) against ground truth.
"""
import json
from src.gliner import GliNER2RE
from src.neo4j_graphrag import Neo4jGraphRagRE


def normalize_triple(triple: dict) -> tuple:
    """Normalize a triple to (subject, relation, object) tuple for comparison."""
    subj = triple.get("subject", "").lower().strip()
    rel = triple.get("relation", "").lower().strip()
    obj = triple.get("object", "").lower().strip()
    return (subj, rel, obj)


def normalize_results(results: list) -> list:
    """Convert list of list of triples to list of sets of normalized tuples."""
    normalized = []
    for sample_triples in results:
        triples_set = set()
        for triple in sample_triples:
            triples_set.add(normalize_triple(triple))
        normalized.append(triples_set)
    return normalized


def normalize_ground_truth(samples: list) -> list:
    """Convert ground truth samples to list of sets of normalized tuples."""
    normalized = []
    for sample in samples:
        triples_set = set()
        for triple in sample['triples']:
            triples_set.add(normalize_triple(triple))
        normalized.append(triples_set)
    return normalized


def calculate_f1(predictions: list, ground_truth: list):
    """Calculate micro-averaged precision, recall, and F1 score."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred, gt in zip(predictions, ground_truth):
        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }


def main():
    # Load ground truth
    with open("data/re/scientist_re_benchmark.json", "r") as f:
        data = json.load(f)
    samples = data['samples']
    
    print("Running Relation Extraction evaluation...")
    print("=" * 60)
    print(f"Total samples: {len(samples)}")
    print(f"Total ground truth triples: {sum(len(s['triples']) for s in samples)}")
    print("=" * 60)
    
    # GLiNER2
    print("\nRunning GLiNER2...")
    gliner = GliNER2RE()
    gliner_results, gliner_time = gliner.run()
    gliner_normalized = normalize_results(gliner_results)
    ground_truth = normalize_ground_truth(samples)
    gliner_scores = calculate_f1(gliner_normalized, ground_truth)
    print(f"GLiNER2 - P: {gliner_scores['precision']:.4f}, R: {gliner_scores['recall']:.4f}, F1: {gliner_scores['f1']:.4f}")
    print(f"GLiNER2 - TP: {gliner_scores['tp']}, FP: {gliner_scores['fp']}, FN: {gliner_scores['fn']}")
    print(f"GLiNER2 - Avg time per sample: {gliner_time:.4f}s")
    
    # Neo4j GraphRAG
    print("\nRunning Neo4j GraphRAG...")
    neo4j = Neo4jGraphRagRE()
    neo4j_results, neo4j_time = neo4j.run()
    neo4j_normalized = normalize_results(neo4j_results)
    # Re-normalize ground truth to match sample count (in case Neo4j runs on subset)
    ground_truth_neo4j = normalize_ground_truth(samples[:len(neo4j_results)])
    neo4j_scores = calculate_f1(neo4j_normalized, ground_truth_neo4j)
    print(f"Neo4j GraphRAG - P: {neo4j_scores['precision']:.4f}, R: {neo4j_scores['recall']:.4f}, F1: {neo4j_scores['f1']:.4f}")
    print(f"Neo4j GraphRAG - TP: {neo4j_scores['tp']}, FP: {neo4j_scores['fp']}, FN: {neo4j_scores['fn']}")
    print(f"Neo4j GraphRAG - Avg time per sample: {neo4j_time:.4f}s")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Relation Extraction")
    print("=" * 60)
    header = f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Avg Time':<12}"
    separator = "-" * 70
    row1 = f"{'GLiNER2':<20} {gliner_scores['precision']:<12.4f} {gliner_scores['recall']:<12.4f} {gliner_scores['f1']:<12.4f} {gliner_time:<12.4f}s"
    row2 = f"{'Neo4j GraphRAG':<20} {neo4j_scores['precision']:<12.4f} {neo4j_scores['recall']:<12.4f} {neo4j_scores['f1']:<12.4f} {neo4j_time:<12.4f}s"

    print(header)
    print(separator)
    print(row1)
    print(row2)

    # Save summary to a file
    with open("re_eval_summary.txt", "w") as summary_file:
        summary_file.write("=" * 60 + "\n")
        summary_file.write("SUMMARY - Relation Extraction\n")
        summary_file.write("=" * 60 + "\n")
        summary_file.write(header + "\n")
        summary_file.write(separator + "\n")
        summary_file.write(row1 + "\n")
        summary_file.write(row2 + "\n")
        summary_file.write("\n")
        summary_file.write(f"GLiNER2 - TP: {gliner_scores['tp']}, FP: {gliner_scores['fp']}, FN: {gliner_scores['fn']}\n")
        summary_file.write(f"Neo4j GraphRAG - TP: {neo4j_scores['tp']}, FP: {neo4j_scores['fp']}, FN: {neo4j_scores['fn']}\n")
    
    print(f"\nSummary saved to re_eval_summary.txt")


if __name__ == "__main__":
    main()

