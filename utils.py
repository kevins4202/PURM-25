"""
Utility functions for the PURM evaluation system.
Contains helper functions for prompt management, string formatting, and data processing.
"""

import json
import os
from typing import Dict, List, Any, Tuple, Literal
from outlines import BaseModel


def get_prompt(broad: bool, zero_shot: bool):
    if broad:
        if zero_shot:
            return "prompts/broad_0_shot.txt"
        else:
            return "prompts/broad_1_shot.txt"
    else:
        if zero_shot:
            return "prompts/granular_0_shot.txt"
        else:
            return "prompts/granular_1_shot.txt"


def load_prompt(broad: bool, zero_shot: bool, note: str) -> str:
    """Load prompt from file"""
    prompt_path = get_prompt(broad, zero_shot)
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r") as f:
        prompt = f.read()

    return prompt.format(note=note)

sentiment = Literal["positive", "negative"]
category_sentiment_broad = Tuple[str, sentiment]


class BroadOutputSchema(BaseModel):
    Employment: List[category_sentiment_broad]
    Housing: List[category_sentiment_broad]
    Food: List[category_sentiment_broad]
    Financial: List[category_sentiment_broad]
    Transportation: List[category_sentiment_broad]
    Childcare: List[category_sentiment_broad]
    Permanency: List[category_sentiment_broad]
    SubstanceAbuse: List[category_sentiment_broad]
    Safety: List[category_sentiment_broad]


employment_labels = Literal["PatientCaregiver_Unemployment"]
housing_labels = Literal[
    "Homelessness",
    "GeneralHousingInstability",
    "NeedTemporaryLodging",
    "HouseInstability_Other",
]
food_labels = Literal["LackofFundsforFood", "FoodInsecurity_Other"]
financial_labels = Literal[
    "Poverty", "LackofInsurance", "UnabletoPay", "FinancialStrain_Other"
]
transportation_labels = Literal[
    "DistancefromHospital", "LackofTransportation", "Transportation_Other"
]
childcare_labels = Literal[
    "ChildcareBarrierfromHospitalization",
    "ChildcareBarrierfromNonHospitalization",
    "NeedofChildcare",
    "Childcare_Other",
]
substance_labels = Literal["DrugUse", "Alcoholism", "SubstanceAbuse_Other"]
safety_labels = Literal[
    "ChildAbuse",
    "HomeSafety",
    "HomeAccessibility",
    "IntimatePartnerViolence",
    "HomeEnvironment_Other",
    "CommunitySafety",
    "CommunityAccessibility",
    "CommunityViolence",
    "CommunityEnvironment_Other",
]
permanency_labels = Literal[
    "NonPermanentPlacement", "PermanentPlacementPending", "Permanency_Other"
]


class GranularOutputSchema(BaseModel):
    Employment: List[Tuple[employment_labels, str, sentiment]]
    Housing: List[Tuple[housing_labels, str, sentiment]]
    Food: List[Tuple[food_labels, str, sentiment]]
    Financial: List[Tuple[financial_labels, str, sentiment]]
    Transportation: List[Tuple[transportation_labels, str, sentiment]]
    Childcare: List[Tuple[childcare_labels, str, sentiment]]
    Permanency: List[Tuple[permanency_labels, str, sentiment]]
    SubstanceAbuse: List[Tuple[substance_labels, str, sentiment]]
    Safety: List[Tuple[safety_labels, str, sentiment]]


def create_evaluation_summary(
    preds: List[List[int]], targets: List[List[int]], broken_indices: List[int]
) -> Dict[str, Any]:
    """
    Create a summary of evaluation results

    Args:
        preds: List of predictions
        targets: List of targets
        broken_indices: List of indices where parsing failed

    Returns:
        Summary dictionary
    """
    total_samples = len(broken_indices) + len(preds)
    successful_samples = len(preds)
    failed_samples = len(broken_indices)

    return {
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "failed_samples": failed_samples,
        "success_rate": (
            successful_samples / total_samples if total_samples > 0 else 0.0
        ),
        "broken_indices": broken_indices,
    }


def save_results(metrics: Dict, broad_metrics: Dict) -> None:
    """
    Save evaluation results to files

    Args:
        metrics: Per-label metrics
        broad_metrics: Macro metrics
        summary: Evaluation summary
        output_dir: Directory to save results
    """
    # Save per-label metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save macro metrics
    with open("broad_metrics.json", "w") as f:
        json.dump(broad_metrics, f, indent=2)

    print(f"Results saved")
