import json
from dataloader import get_dataloaders
from load_model import load_model_pipeline
from metrics import compute_metrics_per_label
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
torch.set_float32_matmul_precision('high')

# pipeline = load_model_pipeline("meta-llama/Llama-3.3-70B-Instruct")
model_id = "meta-llama/Llama-3.1-8B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config, device_map="auto"
)
#quantized_model.compile()


prompt_path = "prompts/broad_0_shot.txt"

with open(prompt_path, "r") as f:
    system_prompt = f.read()

tokenizer = AutoTokenizer.from_pretrained(model_id)

def generate_output(user_message, print_output=False):
    input_text = system_prompt + "\n\n" + user_message
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    output = quantized_model.generate(**input_ids, max_new_tokens=96)

    return tokenizer.decode(output[0], skip_special_tokens=True)

def parse_output(pred):
    try:
        dct = json.loads(pred)

        annotations = [0] * len(cat_to_labels)

        for cat in cat_to_labels.keys():
            if cat in dct:
                assert isinstance(dct[cat], list)
                assert all(isinstance(item, list) and len(item) == 3 and item[0] in cat_to_labels[cat] and item[2] in ["positive", "negative"] for item in dct[cat])

                if any(item[2] == "positive" for item in dct[cat]):
                    annotations[cat_to_i[cat]] = 1
                elif any(item[2] == "negative" for item in dct[cat]):
                    annotations[cat_to_i[cat]] = -1

        return annotations
    except:
        print(f"error converting to JSON {pred}")
        return None

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

cat_to_i = {cat: i for i, cat in enumerate(list(cat_to_labels.keys()))}

model_labels = []

dataloader = get_dataloaders(batch_size=1, split=False)

preds = []
targets = []

for idx, batch in enumerate(dataloader):
    print(f"batch {idx} of {len(dataloader)}")
    notes, labels = batch["note"], batch["labels"]

    for i in range(len(notes)):
        print("NOTE: ", notes[i])
        output = generate_output(notes[i], print_output=False)
        
        print("OUTPUT AND LABELS: ", output, "\n\n", labels)

        pred = parse_output(output["content"])
        if pred is not None:
            preds.append(pred)
            targets.append(labels[i])
            print(f"prediction: {pred} target: {[t.item() for t in labels]}")
        else:
            print("invalid output")
        # i -= 1
    if idx > 0:
        break

if len(preds) != len(targets):
    print("Lengths of preds and targets do not match:", len(preds), len(targets))
    exit()

metrics = compute_metrics_per_label(preds, targets)
print(metrics)
with open("metrics.json", "w") as fp:
    json.dump(metrics, fp)
