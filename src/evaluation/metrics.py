from src.models import REBenchmarkResult, NERBenchmarkResult, EvaluationMetrics, Triple, Entity


def normalize_triple(triple: Triple) -> tuple[str, str, str]:
    """Normalize a triple for comparison (lowercase, stripped)."""
    return (
        triple.subject.strip().lower(),
        triple.relation.strip().lower(),
        triple.object.strip().lower(),
    )


class REEvaluator:
    """Evaluator for relation extraction results.
    
    Calculates micro-averaged precision, recall, and F1 score.
    """
    
    def evaluate(self, benchmark_result: REBenchmarkResult) -> EvaluationMetrics:
        """Evaluate benchmark results against ground truth.
        
        Args:
            benchmark_result: The benchmark result containing predictions and ground truth.
            
        Returns:
            EvaluationMetrics with precision, recall, F1, and counts.
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for sample_result in benchmark_result.results:
            gt_set = {normalize_triple(t) for t in sample_result.ground_truth}
            pred_set = {normalize_triple(t) for t in sample_result.predicted}
            
            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EvaluationMetrics(
            model_id=benchmark_result.model_id,
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
        )

def normalize_entity(entity: Entity) -> tuple[str, str]:
    """Normalize an entity for comparison (lowercase, stripped)."""
    return (
        entity.text.strip().lower(),
        entity.type.strip().lower(),
    )


class NEREvaluator:
    """Evaluator for named entity recognition results.
    
    Calculates micro-averaged precision, recall, and F1 score.
    """
    
    def evaluate(self, benchmark_result: NERBenchmarkResult) -> EvaluationMetrics:
        """Evaluate benchmark results against ground truth.
        
        Args:
            benchmark_result: The benchmark result containing predictions and ground truth.

        Returns:
            EvaluationMetrics with precision, recall, F1, and counts.
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for sample_result in benchmark_result.results:
            gt_set = {normalize_entity(e) for e in sample_result.ground_truth}
            pred_set = {normalize_entity(e) for e in sample_result.predicted}

            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EvaluationMetrics(
            model_id=benchmark_result.model_id,
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
        )