from config import CAT_TO_LABELS

def compute_metrics(preds, targets):
    """
    Computes:
    1. Presence-vs-absence binary metrics (present = 1 or -1, absent = 0)
    2. Stance (1 vs -1) metrics only for cases where target is present (non-zero)

    Args:
        preds: List[Dict[int, int]] — each is a label->value mapping (1, -1, or 0)
        targets: List[Dict[int, int]] — same structure

    Returns:
        Dict containing:
            - presence metrics per label
            - stance metrics per label (only on present targets)
            - global macro averages
    """
    NUM_LABELS = len(CAT_TO_LABELS)
    
    presence_results = {}
    stance_results = {}
    
    presence_macro = {'precision': [], 'recall': [], 'f1': []}
    stance_macro = {'precision': [], 'recall': [], 'f1': []}
    
    for label in range(NUM_LABELS):
        # Counters for presence (binary)
        tp_p, fp_p, fn_p = 0, 0, 0
        
        # Counters for stance (multi-class 1 vs -1)
        tp_1, fp_1, fn_1 = 0, 0, 0
        tp_neg1, fp_neg1, fn_neg1 = 0, 0, 0
        
        for pred, target in zip(preds, targets):
            p = pred[label]
            t = target[label]

            # --- Presence metrics (present = 1 or -1)
            target_present = t != 0
            pred_present = p != 0

            if target_present and pred_present:
                tp_p += 1
            elif not target_present and pred_present:
                fp_p += 1
            elif target_present and not pred_present:
                fn_p += 1
            # TN not needed for F1/precision/recall

            # --- Stance metrics (only on present targets)
            if t != 0:
                if t == 1:
                    if p == 1:
                        tp_1 += 1
                    else:
                        fn_1 += 1
                elif t == -1:
                    if p == -1:
                        tp_neg1 += 1
                    else:
                        fn_neg1 += 1

                if p == 1 and t != 1:
                    fp_1 += 1
                elif p == -1 and t != -1:
                    fp_neg1 += 1

        # --- Presence metrics
        precision_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) else 0.0
        recall_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) else 0.0
        f1_p = 2 * precision_p * recall_p / (precision_p + recall_p) if (precision_p + recall_p) else 0.0
        
        presence_results[label] = {
            'precision': precision_p,
            'recall': recall_p,
            'f1': f1_p,
            'tp': tp_p,
            'fp': fp_p,
            'fn': fn_p,
        }
        presence_macro['precision'].append(precision_p)
        presence_macro['recall'].append(recall_p)
        presence_macro['f1'].append(f1_p)

        # --- Stance metrics
        precision_1 = tp_1 / (tp_1 + fp_1) if (tp_1 + fp_1) else 0.0
        recall_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) else 0.0
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) else 0.0

        precision_neg1 = tp_neg1 / (tp_neg1 + fp_neg1) if (tp_neg1 + fp_neg1) else 0.0
        recall_neg1 = tp_neg1 / (tp_neg1 + fn_neg1) if (tp_neg1 + fn_neg1) else 0.0
        f1_neg1 = 2 * precision_neg1 * recall_neg1 / (precision_neg1 + recall_neg1) if (precision_neg1 + recall_neg1) else 0.0

        macro_prec = (precision_1 + precision_neg1) / 2
        macro_rec = (recall_1 + recall_neg1) / 2
        macro_f1 = (f1_1 + f1_neg1) / 2

        stance_results[label] = {
            'class_1': {'precision': precision_1, 'recall': recall_1, 'f1': f1_1},
            'class_-1': {'precision': precision_neg1, 'recall': recall_neg1, 'f1': f1_neg1},
            'macro': {'precision': macro_prec, 'recall': macro_rec, 'f1': macro_f1},
        }

        stance_macro['precision'].append(macro_prec)
        stance_macro['recall'].append(macro_rec)
        stance_macro['f1'].append(macro_f1)

    return {
        'presence_per_label': presence_results,
        'stance_per_label': stance_results,
        'macro_averages': {
            'presence': {
                'precision': sum(presence_macro['precision']) / NUM_LABELS,
                'recall': sum(presence_macro['recall']) / NUM_LABELS,
                'f1': sum(presence_macro['f1']) / NUM_LABELS,
            },
            'stance': {
                'precision': sum(stance_macro['precision']) / NUM_LABELS,
                'recall': sum(stance_macro['recall']) / NUM_LABELS,
                'f1': sum(stance_macro['f1']) / NUM_LABELS,
            }
        }
    }
