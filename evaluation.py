import json
from dataloader import get_dataloaders
from load_model import load_model_pipeline
from metrics import compute_metrics_per_label


pipeline = load_model_pipeline("meta-llama/Llama-3.3-70B-Instruct")
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
    if pred.strip() == "NoSocialNeedsFoundLabel":
        print("No labels found")
        return [10]
    
    try:
        dct = json.loads(pred)
        
       # print(f"Output: {dct}")
        
        labels = []

        for cat in cats:
            if cat in dct and dct[cat] == "PRESENT":
                labels.append(cattoi[cat])
            elif dct[cat] != "NOT PRESENT":
                print(f"bad label: {cat} {dct}")
                return None
        if len(labels) == 0: return [10]
        return labels
    except:
        print(f"error converting to JSON {dct}")
        return None
    
cats = ["Employment", "Childcare", "Transportation", "Housing", "Food", "Financial", "Permanency", "Substance", "HomeEnvironment", "CommunityEnvironment"]
cattoi = {cat: i for i, cat in enumerate(cats)}

model_labels = []


dataloader = get_dataloaders(batch_size=1, split=False)

preds = []
targets = []

for idx, batch in enumerate(dataloader):
    print(f"batch {idx} of {len(dataloader)}")
    notes, labels = batch['note'], batch['labels']
    
    for i in range(len(notes)):
        output = generate_output(notes[i], print_output=False)

        pred = parse_output(output['content'])
        if pred is not None:
            preds.append(pred)
            targets.append(labels)
            print(f"prediction: {pred} target: {[t.item() for t in labels]}")
        else:
            print("invalid output")
           # i -= 1
   # if idx > 1:
   #     break

if len(preds) != len(targets):
    print("Lengths of preds and targets do not match:", len(preds), len(targets))
    exit()

targets = [[t.item() for t in tar] for tar in targets]

#print(preds)
#print(targets)

metrics = compute_metrics_per_label(preds, targets)
print(metrics)
with open('metrics.json', 'w') as fp:
    json.dump(metrics, fp)
