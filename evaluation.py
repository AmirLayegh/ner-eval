from dotenv import load_dotenv
load_dotenv()


class GliNER2REEvaluator:
    def __init__(self, output_file: str = "result/gliner_re_results.json"):
        self.output_file = output_file

    def evaluate(self):
        import json

        def normalize_triple(triple):
            return (
                triple["subject"].strip().lower(),
                triple["relation"].strip().lower(),
                triple["object"].strip().lower(),
            )

        tp = 0
        fp = 0
        fn = 0

        with open(self.output_file, 'r') as f:
            data = json.load(f)

        for sample in data["results"]:
            gt_triples = {normalize_triple(tr) for tr in sample.get("ground_truth", [])}
            pred_triples = {normalize_triple(tr) for tr in sample.get("predicted", [])}

            tp += len(pred_triples & gt_triples)
            fp += len(pred_triples - gt_triples)
            fn += len(gt_triples - pred_triples)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        return results


if __name__ == "__main__":
    evaluator = GliNER2REEvaluator()
    results = evaluator.evaluate()
    print(results)