from metrics import compute_metrics
from config import BroadOutputSchema, GranularOutputSchema
from utils import (
    load_prompt,
    get_annotations,
)
from vllm import LLM, SamplingParams
import json
import os
import re
import argparse
from vllm_dataloader import get_dataloaders

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
            stop=None
        )

def load_model(model_id):
    """Load and configure the vLLM model"""
    
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
    )
    return llm

def generate_output(llm, prompt, notes):
    """Generate model output for a given user message using vLLM"""
    try:
        prompts = [prompt.format(note=note) for note in notes]
        # Generate output
        outputs = llm.generate(prompts, sampling_params)
        
        # Decode all outputs
        preds = []
        for i, output in enumerate(outputs[0].outputs):
            generated_output = output.text
            print(f"OUTPUT {i}: ", generated_output)
            
            # Find JSON content
            json_match = re.search(r'\{\s*"Employment"\s*:\s*\[.*?],\s*"HousingInstability"\s*:\s*\[.*?],\s*"FoodInsecurity"\s*:\s*\[.*?],\s*"FinancialStrain"\s*:\s*\[.*?],\s*"Transportation"\s*:\s*\[.*?],\s*"Childcare"\s*:\s*\[.*?],\s*"Permanency"\s*:\s*\[.*?],\s*"SubstanceAbuse"\s*:\s*\[.*?],\s*"Safety"\s*:\s*\[.*?]\s*\}', generated_output, re.DOTALL)
            
            if json_match:
                json_text = json_match.group(0)
                print(f"Parsing found JSON text {i}:", json_text.replace('\n', ''))
                
                # Try to parse as JSON
                parsed_json = json.loads(json_text)
                print(f"Successfully parsed JSON {i}")
                preds.append(parsed_json)
            else:
                print(f"No JSON content found for output {i}")
                preds.append(None)
            
    except Exception as e:
        print("Failed to generate and parse structured output:", e)
        return None

    return preds


def evaluate(llm, dataloader, prompt):
    """Evaluate model on a batch of data"""
    preds = []
    targets = []
    broken_indices = []
    max_batches = None

    for idx, batch in enumerate(dataloader):
        if max_batches and idx >= max_batches:
            break

        print(f"\nbatch {idx + 1} of {max_batches if max_batches else len(dataloader)}")
        notes, labels = batch["note"], batch["labels"]

        batch_preds = generate_output(llm, prompt, [notes[0]])

        for i, (pred, label) in enumerate(zip(batch_preds, labels)):
            print(f"NOTE {i}: {notes[i][:50].replace(chr(10), '')}...")
            try:
                if pred is None:
                    print(f"Failed to generate output for sample {idx}_{i}")
                    broken_indices.append(f"{idx}_{i}")
                    continue
                
                print("\n\nGetting annotations...")
                pred_annotations = get_annotations(pred)
                preds.append(pred_annotations)
                targets.append(label)
                print(f"prediction: {pred_annotations} \ntarget: {label}")
            except Exception as e:
                print(f"Error parsing output: {e} {pred}")
                broken_indices.append(f"{idx}_{i}")
    

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate model with command line arguments')
    parser.add_argument('--broad', action='store_true', help='Use broad evaluation')
    parser.add_argument('--granular', action='store_true', help='Use granular evaluation (default if neither specified)')
    parser.add_argument('--zero-shot', action='store_true', help='Use zero-shot evaluation')
    parser.add_argument('--few-shot', action='store_true', help='Use few-shot evaluation (default if neither specified)')
    parser.add_argument('--model-id', type=str, required=True, help='Model ID (overwrites config)')
    args = parser.parse_args()
    
    # Create evaluation config with command line overrides
    # Set defaults if neither option is specified
    if not args.broad and not args.granular:
        args.broad = True  # Default to broad if neither specified
    
    if not args.zero_shot and not args.few_shot:
        args.zero_shot = False  # Default to few-shot if neither specified
    
    evaluation_config = {
        "broad": args.broad,  # True if --broad is used, False otherwise
        "zero_shot": args.zero_shot,  # True if --zero_shot is used, False otherwise
    }
    
    # Create model config with command line overrides
    model_id = args.model_id
    
    print(f"Using evaluation config: broad={evaluation_config['broad']}, zero_shot={evaluation_config['zero_shot']}")
    
    # Get model ID
    output = model_id.split("/")[-1]
    
    # Load prompt
    prompt_path, prompt, examples = load_prompt(
        evaluation_config["broad"],
        evaluation_config["zero_shot"]
    )
    
    # Load model
    llm = load_model(model_id)
    
    # Load data
    dataloader = get_dataloaders(
        batch_size=1, zero_shot=evaluation_config["zero_shot"]
    )

    # Evaluate model
    preds, targets, broken_indices = evaluate(llm, dataloader, prompt)

    # Compute and save metrics
    compute_and_save_metrics(preds, targets, broken_indices, output, prompt_path)

if __name__ == "__main__":
    main()
