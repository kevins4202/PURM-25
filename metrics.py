from collections import defaultdict

cat_to_labels = {
    "PatientCaregiver_Employment": [
        "PatientCaregiver_Unemployment"
    ],
    "HousingInstability": [
        "Homelessness",
        "GeneralHousingInstability",
        "NeedTemporaryLodging",
        "HouseInstability_Other"
    ],
    "FoodInsecurity": [
        "LackofFundsforFood",
        "FoodInsecurity_Other"
    ],
    "FinancialStrain": [
        "Poverty",
        "LackofInsurance",
        "UnabletoPay",
        "FinancialStrain_Other"
    ],
    "Transportation": [
        "DistancefromHospital",
        "LackofTransportation",
        "Transportation_Other"
    ],
    "Childcare": [
        "ChildcareBarrierfromHospitalization",
        "ChildcareBarrierfromNonHospitalization",
        "NeedofChildcare",
        "Childcare_Other"
    ],
    "SubstanceAbuse": [
        "DrugUse",
        "Alcoholism",
        "SubstanceAbuse_Other"
    ],
    "Safety": [
        # Home environment
            "ChildAbuse",
            "HomeSafety",
            "HomeAccessibility",
            "IntimatePartnerViolence",
            "HomeEnvironment_Other",
        # Community environment
            "CommunitySafety",
            "CommunityAccessibility",
            "CommunityViolence",
            "CommunityEnvironment_Other"
        ],
    "Permanency": [
        "NonPermanentPlacement",
        "PermanentPlacementPending",
        "Permanency_Other"
    ]
}

def compute_metrics_per_label(preds, targets):
    """
    Compute metrics for multi-label multi-class classification
    where each label can be: 1 (positive), -1 (negative), 0 (absent/not applicable)
    
    Args:
        preds: List of prediction dictionaries {label: value} where value is 1, -1, or 0
        targets: List of target dictionaries {label: value} where value is 1, -1, or 0
    """
    # Initialize count dictionaries
    true_positives = [0 for _ in range(len(cat_to_labels))]
    false_positives = [0 for _ in range(len(cat_to_labels))]
    false_negatives = [0 for _ in range(len(cat_to_labels))]
    true_negatives = [0 for _ in range(len(cat_to_labels))]
    absent_correct = [0 for _ in range(len(cat_to_labels))]
    absent_incorrect = [0 for _ in range(len(cat_to_labels))]

    for pred, target in zip(preds, targets):
        # For each label, determine the classification
        for label in range(len(cat_to_labels.keys())): 
            pred_val = pred[label]  # Default to absent if not in prediction
            target_val = target[label]  # Default to absent if not in target
            
            # Handle different scenarios
            if target_val == 1:  # Target is positive
                if pred_val == 1:
                    true_positives[label] += 1
                elif pred_val == -1:
                    false_negatives[label] += 1
                elif pred_val == 0:
                    false_negatives[label] += 1  # Absent treated as wrong for positive target
            
            elif target_val == -1:  # Target is negative
                if pred_val == -1:
                    true_negatives[label] += 1
                elif pred_val == 1:
                    false_positives[label] += 1
                elif pred_val == 0:
                    false_positives[label] += 1  # Absent treated as wrong for negative target
            
            elif target_val == 0:  # Target is absent
                if pred_val == 0:
                    absent_correct[label] += 1
                elif pred_val == 1:
                    false_positives[label] += 1
                elif pred_val == -1:
                    false_negatives[label] += 1

    # Calculate metrics for each label
    results = [0 for _ in range(len(cat_to_labels))]
    for label in range(len(cat_to_labels.keys())):
        tp = true_positives[label]
        fp = false_positives[label]
        fn = false_negatives[label]
        tn = true_negatives[label]
        ac = absent_correct[label]
        ai = absent_incorrect[label]

        # Standard binary metrics (treating 0 as "not applicable")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy considering all three classes
        total_samples = tp + fp + fn + tn + ac + ai
        accuracy = (tp + tn + ac) / total_samples if total_samples > 0 else 0.0
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Absent accuracy (how well we predict absent cases)
        absent_accuracy = ac / (ac + ai) if (ac + ai) > 0 else 0.0

        results[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'absent_accuracy': absent_accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'absent_correct': ac,
            'absent_incorrect': ai,
            'total_samples': total_samples
        }

    return results

def compute_macro_metrics(results):
    """Compute macro-averaged metrics across all labels"""
    macro_precision = sum(r['precision'] for r in results.values()) / len(results)
    macro_recall = sum(r['recall'] for r in results.values()) / len(results)
    macro_f1 = sum(r['f1'] for r in results.values()) / len(results)
    macro_accuracy = sum(r['accuracy'] for r in results.values()) / len(results)
    macro_absent_accuracy = sum(r['absent_accuracy'] for r in results.values()) / len(results)
    
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_accuracy': macro_accuracy,
        'macro_absent_accuracy': macro_absent_accuracy
    }

if __name__ == "__main__":
    # Example with multi-label multi-class data
    preds = [
        {1: 1, 2: -1, 3: 0},  # Label 1: positive, Label 2: negative, Label 3: absent
        {1: -1, 2: 1, 3: 1},  # Label 1: negative, Label 2: positive, Label 3: positive
        {1: 0, 2: 0, 3: -1},  # Label 1: absent, Label 2: absent, Label 3: negative
    ]
    targets = [
        {1: 1, 2: -1, 3: 0},  # All correct
        {1: 1, 2: 1, 3: 1},   # Label 1: wrong (predicted negative, should be positive)
        {1: 0, 2: 0, 3: -1},  # All correct
    ]
    
    results = compute_metrics_per_label(preds, targets)
    print("Per-label metrics:")
    for label, metrics in results.items():
        print(f"Label {label}: {metrics}")
    
    macro_metrics = compute_macro_metrics(results)
    print(f"\nMacro-averaged metrics: {macro_metrics}")
