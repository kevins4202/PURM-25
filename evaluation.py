import json
from dataloader import get_dataloaders
from load_model import load_model_pipeline
from metrics import compute_metrics_per_label, compute_macro_metrics
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
    input_text = system_prompt + "\n\n" + user_message + "\n\nAnnotation in specified JSON output format with NO ADDITIONAL NOTES OR TEXT, JUST THE JSON:"
    
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    input_length = input_ids["input_ids"].shape[1]

    output = quantized_model.generate(**input_ids, max_new_tokens=96)
    
    generated_tokens = output[0][input_length:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def parse_output(pred, granular=False):
    try:
        assert isinstance(pred, str)
        pred = pred.strip()
        
        i1 = pred.index('{')
        i2 = pred.index('}')
        
        pred = pred[i1:i2+1]
        print("\n\nJSON: --------------\n", pred.replace('\n', ''), "\n-----------")
        
        dct = json.loads(pred)

        annotations = [0] * len(cat_to_labels)

        for cat in cat_to_labels.keys():
            if cat in dct:
                assert isinstance(dct[cat], list)
                dct[cat] = [x for x in dct[cat] if len(x) > 0]
                assert all(isinstance(item, list) and len(item) == 2 + int(granular) and (item[0] in cat_to_labels[cat] if granular else True) and item[-1] in ["positive", "negative"] for item in dct[cat])

                if any(item[2] == "positive" for item in dct[cat]):
                    annotations[cat_to_i[cat]] = 1
                elif any(item[2] == "negative" for item in dct[cat]):
                    annotations[cat_to_i[cat]] = -1
        
        print("successfull parsing")
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
broke = []

for idx, batch in enumerate(dataloader):
    if idx > 5:
        break
        
    print(f"batch {idx} of {len(dataloader)}")
    notes, labels = batch["note"], batch["labels"]

    for i in range(len(notes)):
        print("\n\nNOTE: ", notes[i][:20])
        output = generate_output(notes[i], print_output=False)

        pred = parse_output(output)
        if pred is not None:
            preds.append(pred)
            targets.append(labels[i])
            print(f"prediction: {pred} target: {labels[i]}")
        else:
            print("invalid output")
            broke.append(idx)

if len(preds) != len(targets):
    print("Lengths of preds and targets do not match:", len(preds), len(targets))
    exit()

print("\n\nBroken inputs: ", len(broke), broke) 

metrics = compute_metrics_per_label(preds, targets)
print(metrics)

with open("metrics.json", "w") as fp:
    json.dump(metrics, fp)
    
broad_metrics = compute_macro_metrics(preds, targets)
print(metrics)

with open("broad_metrics.json", "w") as fp:
    json.dump(broad_metrics, fp)
