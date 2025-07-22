from collections import defaultdict
from config import CAT_TO_LABELS

def compute_metrics_per_label(preds, targets):
    """
    Compute per-class metrics for multi-label multi-class classification
    where each label can be one of: 1 (class A), -1 (class B), 0 (class C)

    Args:
        preds: List of prediction dictionaries {label_index: value (1, -1, 0)}
        targets: List of target dictionaries {label_index: value (1, -1, 0)}

    Returns:
        Dictionary with per-class, per-label metrics and macro-averaged metrics.
    """
    NUM_LABELS = len(CAT_TO_LABELS)
    CLASSES = [1, -1, 0]

    # Initialize counts
    metrics = {label: {c: defaultdict(int) for c in CLASSES} for label in range(NUM_LABELS)}

    for pred, target in zip(preds, targets):
        for label in range(NUM_LABELS):
            p = pred[label]
            t = target[label]

            for c in CLASSES:
                if t == c and p == c:
                    metrics[label][c]['tp'] += 1
                elif t != c and p == c:
                    metrics[label][c]['fp'] += 1
                elif t == c and p != c:
                    metrics[label][c]['fn'] += 1
                else:
                    metrics[label][c]['tn'] += 1  # You can collect it, but it's rarely used

    # Compute metrics
    results = {}
    global_macro = {'precision': [], 'recall': [], 'f1': []}

    for label in range(NUM_LABELS):
        class_metrics = {}
        label_macro = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        for c in CLASSES:
            tp = metrics[label][c]['tp']
            fp = metrics[label][c]['fp']
            fn = metrics[label][c]['fn']
            tn = metrics[label][c]['tn']

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            class_metrics[c] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
            }

            # Accumulate for macro
            label_macro['precision'] += precision
            label_macro['recall'] += recall
            label_macro['f1'] += f1

            global_macro['precision'].append(precision)
            global_macro['recall'].append(recall)
            global_macro['f1'].append(f1)

        # Average over the 3 classes for this label
        for k in label_macro:
            label_macro[k] /= len(CLASSES)

        class_metrics['macro'] = label_macro
        results[label] = class_metrics

    # Global macro across all labels and all classes
    final_macro = {
        'macro_precision': sum(global_macro['precision']) / len(global_macro['precision']),
        'macro_recall': sum(global_macro['recall']) / len(global_macro['recall']),
        'macro_f1': sum(global_macro['f1']) / len(global_macro['f1']),
    }

    return {
        'per_label': results,
        'global_macro': final_macro,
    }
