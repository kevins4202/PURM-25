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
        tp_p, fp_p, fn_p, tn_p = 0, 0, 0, 0
        
        # Counters for stance (binary: 1 vs -1)
        tp_stance, fp_stance, fn_stance, tn_stance = 0, 0, 0, 0
        
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
            elif not target_present and not pred_present:
                tn_p += 1

            # --- Stance metrics (only on present targets)
            if t != 0:
                # Treat stance as binary: 1 (social need) vs -1 (no social need)
                if t == 1:  # Target is social need present
                    if p == 1:  # Correctly predicted social need
                        tp_stance += 1
                    else:  # Predicted no social need or 0
                        fn_stance += 1
                elif t == -1:  # Target is no social need
                    if p == -1:  # Correctly predicted no social need
                        tn_stance += 1
                    else:  # Predicted social need or 0
                        fp_stance += 1

        # --- Presence metrics
        precision_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) else 0.0
        recall_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) else 0.0
        f1_p = 2 * precision_p * recall_p / (precision_p + recall_p) if (precision_p + recall_p) else 0.0

        presence_total_instances = tp_p + fp_p + fn_p + tn_p
        presence_correct = tp_p + tn_p
        presence_accuracy = presence_correct / presence_total_instances if presence_total_instances > 0 else 0.0
        
        presence_results[label] = {
            'precision': precision_p,
            'recall': recall_p,
            'f1': f1_p,
            'tp': tp_p,
            'fp': fp_p,
            'fn': fn_p,
            'tn': tn_p,
            'accuracy': presence_accuracy,
            'correct': presence_correct,
            'total_instances': presence_total_instances,
        }
        presence_macro['precision'].append(precision_p)
        presence_macro['recall'].append(recall_p)
        presence_macro['f1'].append(f1_p)

        # --- Stance metrics (binary: 1 vs -1)
        stance_precision = tp_stance / (tp_stance + fp_stance) if (tp_stance + fp_stance) else 0.0
        stance_recall = tp_stance / (tp_stance + fn_stance) if (tp_stance + fn_stance) else 0.0
        stance_f1 = 2 * stance_precision * stance_recall / (stance_precision + stance_recall) if (stance_precision + stance_recall) else 0.0
        
        # Calculate accuracy for stance (only on present targets)
        stance_total_instances = tp_stance + fp_stance + fn_stance + tn_stance
        stance_correct = tp_stance + tn_stance
        stance_accuracy = stance_correct / stance_total_instances if stance_total_instances > 0 else 0.0

        stance_results[label] = {
            'precision': stance_precision,
            'recall': stance_recall,
            'f1': stance_f1,
            'tp': tp_stance,
            'fp': fp_stance,
            'fn': fn_stance,
            'tn': tn_stance,
            'accuracy': stance_accuracy,
            'correct': stance_correct,
            'total_instances': stance_total_instances,
        }

        stance_macro['precision'].append(stance_precision)
        stance_macro['recall'].append(stance_recall)
        stance_macro['f1'].append(stance_f1)

    # Calculate macro totals for presence (sum across all categories)
    macro_presence_tp = sum(presence_results[cat]['tp'] for cat in presence_results.keys())
    macro_presence_fp = sum(presence_results[cat]['fp'] for cat in presence_results.keys())
    macro_presence_tn = sum(presence_results[cat]['tn'] for cat in presence_results.keys())
    macro_presence_fn = sum(presence_results[cat]['fn'] for cat in presence_results.keys())
    macro_presence_instances = macro_presence_tp + macro_presence_fp + macro_presence_tn + macro_presence_fn
    
    # Calculate macro accuracy for presence
    macro_presence_correct = macro_presence_tp + macro_presence_tn
    macro_presence_accuracy = macro_presence_correct / macro_presence_instances if macro_presence_instances > 0 else 0.0
    
    # Calculate macro totals for stance (sum across all categories)
    macro_stance_tp = sum(stance_results[cat]['tp'] for cat in stance_results.keys())
    macro_stance_fp = sum(stance_results[cat]['fp'] for cat in stance_results.keys())
    macro_stance_fn = sum(stance_results[cat]['fn'] for cat in stance_results.keys())
    macro_stance_tn = sum(stance_results[cat]['tn'] for cat in stance_results.keys())
    
    macro_stance_instances = macro_stance_tp + macro_stance_fp + macro_stance_fn + macro_stance_tn
    
    # Calculate macro accuracy for stance
    macro_stance_correct = macro_stance_tp + macro_stance_tn
    macro_stance_accuracy = macro_stance_correct / macro_stance_instances if macro_stance_instances > 0 else 0.0

    social_needs_tp = 0
    social_needs_fp = 0
    social_needs_fn = 0
    social_needs_tn = 0

    for pred, target in zip(preds, targets):
        no_social_needs = all(t == 0 for t in target)
        pred_no_social_needs = all(p == 0 for p in pred)

        if no_social_needs and pred_no_social_needs:
            social_needs_tn += 1
        elif no_social_needs and not pred_no_social_needs:
            social_needs_fn += 1
        elif not no_social_needs and pred_no_social_needs:
            social_needs_fp += 1
        elif not no_social_needs and not pred_no_social_needs:
            social_needs_tp += 1

    social_needs_instances = social_needs_tp + social_needs_fp + social_needs_fn + social_needs_tn
    social_needs_correct = social_needs_tp + social_needs_tn
    social_needs_accuracy = social_needs_correct / social_needs_instances if social_needs_instances > 0 else 0.0
    social_needs_precision = social_needs_tp / (social_needs_tp + social_needs_fp) if (social_needs_tp + social_needs_fp) else 0.0
    social_needs_recall = social_needs_tp / (social_needs_tp + social_needs_fn) if (social_needs_tp + social_needs_fn) else 0.0
    social_needs_f1 = 2 * social_needs_precision * social_needs_recall / (social_needs_precision + social_needs_recall) if (social_needs_precision + social_needs_recall) else 0.0
    
    return {
        'presence_per_label': presence_results,
        'stance_per_label': stance_results,
        'macro_averages': {
            'presence': {
                'precision': sum(presence_macro['precision']) / NUM_LABELS,
                'recall': sum(presence_macro['recall']) / NUM_LABELS,
                'f1': sum(presence_macro['f1']) / NUM_LABELS,
                'tp': macro_presence_tp,
                'fp': macro_presence_fp,
                'tn': macro_presence_tn,
                'fn': macro_presence_fn,
                'correct': macro_presence_correct,
                'total_instances': macro_presence_instances,
                'accuracy': macro_presence_accuracy,
            },
            'stance': {
                'precision': sum(stance_macro['precision']) / NUM_LABELS,
                'recall': sum(stance_macro['recall']) / NUM_LABELS,
                'f1': sum(stance_macro['f1']) / NUM_LABELS,
                'tp': macro_stance_tp,
                'fp': macro_stance_fp,
                'fn': macro_stance_fn,
                'tn': macro_stance_tn,
                'correct': macro_stance_correct,
                'total_instances': macro_stance_instances,
                'accuracy': macro_stance_accuracy,
            }
        },
        'social_needs': {
            'precision': social_needs_precision,
            'recall': social_needs_recall,
            'f1': social_needs_f1,
            'tp': social_needs_tp,
            'fp': social_needs_fp,
            'fn': social_needs_fn,
            'tn': social_needs_tn,
            'correct': social_needs_correct,
            'total_instances': social_needs_instances,
            'accuracy': social_needs_accuracy,
        }
    }
