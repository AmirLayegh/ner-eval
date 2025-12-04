"""
Simple NER evaluation script that calculates F1 scores for GLiNER2, LangExtract, and Neo4j GraphRAG.
"""
import json
from src.gliner import GliNER2Extractor
from src.langextract import LangExtractor
from src.neo4j_graphrag import Neo4jGraphRagExtractor


def normalize_gliner2(results):
    """Convert GLiNER2 output to list of sets of (text, type) tuples."""
    normalized = []
    for result in results:
        entities = set()
        for entity_type, texts in result['entities'].items():
            for text in texts:
                entities.add((text.lower().strip(), entity_type.lower()))
        normalized.append(entities)
    return normalized


def normalize_langextract(results):
    """Convert LangExtract output to list of sets of (text, type) tuples."""
    normalized = []
    for doc in results:
        entities = set()
        for extraction in doc.extractions:
            entities.add((extraction.extraction_text.lower().strip(), extraction.extraction_class.lower()))
        normalized.append(entities)
    return normalized


def normalize_neo4j(results):
    """Convert Neo4j GraphRAG output to list of sets of (text, type) tuples."""
    normalized = []
    for result in results:
        entities = set()
        for node in result:
            name = node.get('properties', {}).get('name', '')
            if name:
                entities.add((name.lower().strip(), node['label'].lower()))
        normalized.append(entities)
    return normalized


def normalize_ground_truth(samples):
    """Convert ground truth to list of sets of (text, type) tuples."""
    normalized = []
    for sample in samples:
        entities = set()
        for entity in sample['entities']:
            entities.add((entity['text'].lower().strip(), entity['type'].lower()))
        normalized.append(entities)
    return normalized


def calculate_f1(predictions, ground_truth):
    """Calculate precision, recall, and F1 score."""
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
    
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    # Load ground truth
    with open("data/science_ner_benchmark.json", "r") as f:
        data = json.load(f)
    samples = data['samples']
    ground_truth = normalize_ground_truth(samples)
    
    print("Running NER evaluation...")
    print("=" * 50)
    
    # GLiNER2
    print("\nRunning GLiNER2...")
    gliner = GliNER2Extractor()
    gliner_results, gliner_time = gliner.run()
    gliner_normalized = normalize_gliner2(gliner_results)
    gliner_scores = calculate_f1(gliner_normalized, ground_truth)
    print(f"GLiNER2 - Precision: {gliner_scores['precision']:.4f}, Recall: {gliner_scores['recall']:.4f}, F1: {gliner_scores['f1']:.4f}")
    print(f"GLiNER2 - Avg time per sample: {gliner_time:.4f}s")
    
    # LangExtract
    print("\nRunning LangExtract...")
    langextract = LangExtractor()
    langextract_results, langextract_time = langextract.run()
    langextract_normalized = normalize_langextract(langextract_results)
    langextract_scores = calculate_f1(langextract_normalized, ground_truth)
    print(f"LangExtract - Precision: {langextract_scores['precision']:.4f}, Recall: {langextract_scores['recall']:.4f}, F1: {langextract_scores['f1']:.4f}")
    print(f"LangExtract - Avg time per sample: {langextract_time:.4f}s")
    
    # Neo4j GraphRAG
    print("\nRunning Neo4j GraphRAG...")
    neo4j = Neo4jGraphRagExtractor()
    neo4j_results, neo4j_time = neo4j.run()
    neo4j_normalized = normalize_neo4j(neo4j_results)
    neo4j_scores = calculate_f1(neo4j_normalized, ground_truth)
    print(f"Neo4j GraphRAG - Precision: {neo4j_scores['precision']:.4f}, Recall: {neo4j_scores['recall']:.4f}, F1: {neo4j_scores['f1']:.4f}")
    print(f"Neo4j GraphRAG - Avg time per sample: {neo4j_time:.4f}s")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    header = f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Avg Time':<12}"
    separator = "-" * 60
    row1 = f"{'GLiNER2':<20} {gliner_scores['precision']:<12.4f} {gliner_scores['recall']:<12.4f} {gliner_scores['f1']:<12.4f} {gliner_time:<12.4f}s"
    row2 = f"{'LangExtract':<20} {langextract_scores['precision']:<12.4f} {langextract_scores['recall']:<12.4f} {langextract_scores['f1']:<12.4f} {langextract_time:<12.4f}s"
    row3 = f"{'Neo4j GraphRAG':<20} {neo4j_scores['precision']:<12.4f} {neo4j_scores['recall']:<12.4f} {neo4j_scores['f1']:<12.4f} {neo4j_time:<12.4f}s"

    print(header)
    print(separator)
    print(row1)
    print(row2)
    print(row3)

    # Save summary to a file
    with open("ner_eval_summary.txt", "w") as summary_file:
        summary_file.write("=" * 50 + "\n")
        summary_file.write("SUMMARY\n")
        summary_file.write("=" * 50 + "\n")
        summary_file.write(header + "\n")
        summary_file.write(separator + "\n")
        summary_file.write(row1 + "\n")
        summary_file.write(row2 + "\n")
        summary_file.write(row3 + "\n")


if __name__ == "__main__":
    main()

