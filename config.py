"""
Configuration file for the PURM evaluation system.
Contains category definitions, model configurations, and other constants.
"""

# Model configuration
MODEL_CONFIG = {
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "quantization": True,
    "max_new_tokens": 96
}

# Category mapping for broad classification (output categories to internal categories)
CATEGORY_MAPPING = {
    "Employment": "PatientCaregiver_Employment",
    "Housing": "HousingInstability", 
    "Food": "FoodInsecurity",
    "Financial": "FinancialStrain",
    "Transportation": "Transportation",
    "Childcare": "Childcare",
    "Permanency": "Permanency",
    "Substance": "SubstanceAbuse",
    "Safety": "Safety"
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
    "max_batches": 6,  # For testing, set to None for full evaluation
    "prompt_path": "prompts/broad_0_shot.txt",
    "output_files": {
        "metrics": "metrics.json",
        "broad_metrics": "broad_metrics.json"
    }
}

# Prompt templates
PROMPT_TEMPLATES = {
    "broad_0_shot": {
        "path": "prompts/broad_0_shot.txt",
        "description": "Broad classification with 0-shot learning"
    },
    "broad_1_shot": {
        "path": "prompts/broad_1_shot.txt", 
        "description": "Broad classification with 1-shot learning"
    },
    "granular_0_shot": {
        "path": "prompts/granular_0_shot.txt",
        "description": "Granular classification with 0-shot learning"
    },
    "granular_1_shot": {
        "path": "prompts/granular_1_shot.txt",
        "description": "Granular classification with 1-shot learning"
    }
} 