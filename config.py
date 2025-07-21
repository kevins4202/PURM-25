"""
Configuration file for the PURM evaluation system.
Contains category definitions, model configurations, and other constants.
"""

# Model configuration
MODEL_CONFIG = {
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "quantization": True,
    "max_new_tokens": 256,
}

# Detailed category definitions with subcategories
CAT_TO_LABELS = {
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

# Category to index mapping for evaluation
CAT_TO_I = {cat: i for i, cat in enumerate(CAT_TO_LABELS.keys())}

# Evaluation settings
EVALUATION_CONFIG = {
    "batch_size": 1,
    "max_batches": 5,  # For testing, set to None for full evaluation
    "broad": True,
    "zero_shot": True
}
