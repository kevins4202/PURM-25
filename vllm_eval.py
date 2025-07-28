from dataloader import get_dataloaders
from metrics import compute_metrics
from config import MODEL_CONFIG, EVALUATION_CONFIG, BroadOutputSchema, GranularOutputSchema
from utils import (
    load_prompt,
    get_annotations,
)
from vllm import LLM, SamplingParams
import torch
import json
import os
import re

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
model_path = MODEL_CONFIG["model_id"]
output_path = f"results/{model_path.split('/')[-1]}.json"

def load_model(model_config):
    """Load and configure the vLLM model"""
    model_id = model_config["model_id"].split("/")[-1]
    print("Loading vLLM model ", model_id)
    
    llm = LLM(
        model=model_config["model_id"],
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
    )
    return llm

def generate_output(llm, prompt, note, output_schema):
    """Generate model output for a given user message using vLLM"""
    try:
        # Format the prompt with the note
        formatted_prompt = prompt.format(note=note)
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop=None
        )
        
        # Generate output
        outputs = llm.generate([formatted_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        
        # Try to extract JSON from the generated text
        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            # Parse and validate the JSON
            pred = output_schema.model_validate_json(json_str).model_dump(mode="json")
            print("Parsed structured output:", json_str)
            return pred
        else:
            print("No JSON found in generated text:", generated_text)
            return None
            
    except Exception as e:
        print("Failed to generate and parse structured output:", e)
        return None


def evaluate(llm, dataloader, evaluation_config, prompt, output_schema):
    """Evaluate model on a batch of data"""
    preds = []
    targets = []
    broken_indices = []
    max_batches = evaluation_config["max_batches"]

    for idx, batch in enumerate(dataloader):
        if max_batches and idx >= max_batches:
            break

        print(f"\nbatch {idx + 1} of {max_batches if max_batches else len(dataloader)}")
        notes, labels = batch["note"], batch["labels"]

        for i in range(len(notes)):
            print(f"NOTE: {notes[i][:50].replace(chr(10), '')}...")
            pred = generate_output(llm, prompt, notes[i], output_schema)
            try:
                preds.append(get_annotations(pred))
                targets.append(labels[i])
                print(f"prediction: {preds[-1]} \ntarget: {labels[i].tolist()}")
            except Exception as e:
                print(f"Error parsing output: {e} {pred}")
                broken_indices.append(idx)

    return preds, targets, broken_indices


def compute_and_save_metrics(preds, targets, broken_indices, model_id, prompt_path):
    """Compute metrics and save results"""
    if len(preds) != len(targets):
        print(
            f"Lengths of preds and targets do not match: {len(preds)} vs {len(targets)}"
        )
        return

    # Create evaluation summary
    total_samples = len(broken_indices) + len(preds)
    successful_samples = len(preds)
    failed_samples = len(broken_indices)

    success_rate = successful_samples / total_samples if total_samples > 0 else 0.0
    
    print(f"\n\nEvaluation Summary:")
    print(f"Total samples: {total_samples}")
    print(f"Successful samples: {successful_samples}")
    print(f"Failed samples: {failed_samples}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Broken indices: {broken_indices}")

    # Compute per-label metrics
    metrics = compute_metrics(preds, targets)
    print("\nMetrics:", metrics)

    os.makedirs("results", exist_ok=True)
    os.makedirs(f"results/{model_id}", exist_ok=True)

    i = 0
    while os.path.exists(f"results/{model_id}/{prompt_path}_{i}.json"):
        i += 1

    with open(f"results/{model_id}/{prompt_path}_{i}.json", "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    """Main evaluation function"""
    # Get model ID
    model_id = MODEL_CONFIG["model_id"].split("/")[-1]
    
    # Determine output schema
    output_schema = BroadOutputSchema if EVALUATION_CONFIG["broad"] else GranularOutputSchema
    
    # Load prompt
    prompt_path, prompt = load_prompt(
        EVALUATION_CONFIG["broad"],
        EVALUATION_CONFIG["zero_shot"]
    )
    
    # Load model
    llm = load_model(MODEL_CONFIG)
    
    # Load data
    dataloader = get_dataloaders(
        batch_size=EVALUATION_CONFIG["batch_size"], split=False
    )

    # Evaluate model
    preds, targets, broken_indices = evaluate(llm, dataloader, EVALUATION_CONFIG, prompt, output_schema)

    # Compute and save metrics
    compute_and_save_metrics(preds, targets, broken_indices, model_id, prompt_path)


if __name__ == "__main__":
    main()
