"""
Configuration file for the PURM evaluation system.
Contains category definitions, model configurations, and other constants.
"""

from typing import List, Tuple, Literal
from pydantic import BaseModel

# Model configuration
MODEL_CONFIG = {
<<<<<<< HEAD
    "model_id": "meta-llama/Llama-3.3-70B-Instruct",
=======
    "model_id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
>>>>>>> d3a6a8f22a9f3f533c4b0ed1f28bf4d1d81eba6e
    "quantization": True,
}

# Evaluation settings
EVALUATION_CONFIG = {
    "batch_size": 1,
    "max_batches": None,  # For testing, set to None for full evaluation
    "broad": False,
    "zero_shot": True,
}

OUTPUT_TO_CAT = {
    "Employment": "PatientCaregiver_Employment",
    "Housing": "HousingInstability",
    "Food": "FoodInsecurity",
    "Financial": "FinancialStrain",
    "Transportation": "Transportation",
    "Childcare": "Childcare",
    "Permanency": "Permanency",
    "SubstanceAbuse": "SubstanceAbuse",
    "Safety": "Safety",
}

# Detailed category definitions with subcategories
CAT_TO_LABELS = {
    "PatientCaregiver_Employment": ["PatientCaregiver_Unemployment"],
    "HousingInstability": [
        "Homelessness",
        "GeneralHousingInstability",
        "NeedTemporaryLodging",
        "HouseInstability_Other",
    ],
    "FoodInsecurity": ["LackofFundsforFood", "FoodInsecurity_Other"],
    "FinancialStrain": [
        "Poverty",
        "LackofInsurance",
        "UnabletoPay",
        "FinancialStrain_Other",
    ],
    "Transportation": [
        "DistancefromHospital",
        "LackofTransportation",
        "Transportation_Other",
    ],
    "Childcare": [
        "ChildcareBarrierfromHospitalization",
        "ChildcareBarrierfromNonHospitalization",
        "NeedofChildcare",
        "Childcare_Other",
    ],
    "SubstanceAbuse": ["DrugUse", "Alcoholism", "SubstanceAbuse_Other"],
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
        "CommunityEnvironment_Other",
    ],
    "Permanency": [
        "NonPermanentPlacement",
        "PermanentPlacementPending",
        "Permanency_Other",
    ],
}

# Category to index mapping for evaluation
CAT_TO_I = {cat: i for i, cat in enumerate(CAT_TO_LABELS.keys())}

sentiment = Literal["social need", "no social need"]
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

employment_labels = Literal[*CAT_TO_LABELS["PatientCaregiver_Employment"]]
housing_labels = Literal[*CAT_TO_LABELS["HousingInstability"]]
food_labels = Literal[*CAT_TO_LABELS["FoodInsecurity"]]
financial_labels = Literal[*CAT_TO_LABELS["FinancialStrain"]]
transportation_labels = Literal[*CAT_TO_LABELS["Transportation"]]
childcare_labels = Literal[*CAT_TO_LABELS["Childcare"]]
substance_labels = Literal[*CAT_TO_LABELS["SubstanceAbuse"]]
safety_labels = Literal[*CAT_TO_LABELS["Safety"]]
permanency_labels = Literal[*CAT_TO_LABELS["Permanency"]]


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
