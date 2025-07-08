from collections import defaultdict

def compute_metrics_per_label(preds, targets):
    # Initialize count dictionaries
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    label_set = set()

    for pred, target in zip(preds, targets):
        pred_set = set(pred)
        target_set = set(target)
        
        # Update label set
        label_set.update(pred_set)
        label_set.update(target_set)

        for label in pred_set:
            if label in target_set:
                true_positives[label] += 1
            else:
                false_positives[label] += 1
        for label in target_set:
            if label not in pred_set:
                false_negatives[label] += 1

    # Calculate precision, recall, f1 for each label
    results = {}
    for label in label_set:
        tp = true_positives[label]
        fp = false_positives[label]
        fn = false_negatives[label]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return results

if __name__ == "__main__":
    preds = [[1, 2, 3], [1, 2, 4], [1, 2, 3], [1, 2, 4]]
    targets = [[1, 2, 3], [1, 2, 4], [1, 2, 3], [1, 2, 4]]
    print(compute_metrics_per_label(preds, targets))