"""
Configuration file for the PURM evaluation system.
Contains category definitions, model configurations, and other constants.
"""

from typing import List, Tuple, Literal
from pydantic import BaseModel

BATCH_SIZE = 1

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

employment_labels = Literal["PatientCaregiver_Unemployment"]
housing_labels = Literal["Homelessness", "GeneralHousingInstability", "NeedTemporaryLodging", "HouseInstability_Other"]
food_labels = Literal["LackofFundsforFood", "FoodInsecurity_Other"]
financial_labels = Literal["Poverty", "LackofInsurance", "UnabletoPay", "FinancialStrain_Other"]
transportation_labels = Literal["DistancefromHospital", "LackofTransportation", "Transportation_Other"]
childcare_labels = Literal["ChildcareBarrierfromHospitalization", "ChildcareBarrierfromNonHospitalization", "NeedofChildcare", "Childcare_Other"]
substance_labels = Literal["DrugUse", "Alcoholism", "SubstanceAbuse_Other"]
safety_labels = Literal["ChildAbuse", "HomeSafety", "HomeAccessibility", "IntimatePartnerViolence", "HomeEnvironment_Other", "CommunitySafety", "CommunityAccessibility", "CommunityViolence", "CommunityEnvironment_Other"]
permanency_labels = Literal["NonPermanentPlacement", "PermanentPlacementPending", "Permanency_Other"]


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
