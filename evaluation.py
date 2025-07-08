import json
from dataloader import get_dataloaders
from load_model import load_model_pipeline
from metrics import compute_metrics_per_label


pipeline = load_model_pipeline("meta-llama/Llama-3.1-8B-Instruct")
prompt_path = "prompts/prompt2.txt"

with open(prompt_path, "r") as f:
    system_prompt = f.read()

def generate_output(user_message, print_output=False):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )

    if print_output:
        print(outputs[0]["generated_text"][-1])

    return outputs[0]["generated_text"][-1]

def parse_output(pred):
    if pred == "NoSocialNeedsFoundLabel": return 10

    if isinstance(pred, dict):
        for cat in cats:
            if cat in pred:
                if pred[cat] not in ["NOT PRESENT", "PRESENT"]:
                    return None
        try:
            dct =  json.loads(pred)

            labels = []
            for cat in cats:
                if cat in dct and dct[cat] == "PRESENT":
                    labels.append(cattoi[cat])
            return labels
        except:
            return None
    else:
        return None
    
cats = ["Employment", "Childcare", "Transportation", "Housing", "Food", "Financial", "Permanency", "Substance", "HomeEnvironment", "CommunityEnvironment"]
cattoi = {cat: i for i, cat in enumerate(cats)}

model_labels = []

if __name__ == "__main__":
    pipeline = load_model_pipeline("meta-llama/Llama-3.1-8B-Instruct")

    dataloader = get_dataloaders(batch_size=1, split=False)

    preds = []
    targets = []

    for batch in dataloader:
        notes, labels = batch['note'], batch['labels']
        targets.extend(labels)

        for i in range(len(notes)):
            output = generate_output(notes[i], print_output=False)

            pred = parse_output(output)
            if pred is not None:
                preds.append(pred)
            else:
                print(f"Invalid output: {output}")
                i -= 1

    print(len(preds))
    print(len(targets))

    print(preds)
    print(targets)

    if len(preds) != len(targets):
        print("Lengths of preds and targets do not match:", len(preds), len(targets))
        exit()

    metrics = compute_metrics_per_label(preds, targets)
    print(metrics)