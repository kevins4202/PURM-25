from dataloader import get_dataloaders
from metrics import compute_metrics
from config import BroadOutputSchema, GranularOutputSchema
from utils import (
    load_prompt,
    get_annotations,
)
import outlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from vllm import LLM, SamplingParams
import json
import os
import argparse
import re   
import json

# Configuration
torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

class ModelEvaluator:
    def __init__(self, model_id, evaluation_config):
        self.model_id = model_id
        self.output_dir = model_id.split("/")[-1]
        self.evaluation_config = evaluation_config
        self.outlined_model = None
        self.generator = None
        self.output_schema = BroadOutputSchema if self.evaluation_config["broad"] else GranularOutputSchema
        self._load_model()
        self.prompt_path, self.prompt, self.examples = load_prompt(
            self.evaluation_config["broad"],
            self.evaluation_config["zero_shot"]
        )

    def _load_model(self):
        """Load and configure the model"""
        print("Loading and quantizing model ", self.output_dir)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map=device,
        )
        print("Loaded model")
        model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        print("Loaded tokenizer")
        
        if self.evaluation_config["eval_type"] == "s":
            self.outlined_model = outlines.from_transformers(model, tokenizer)
        elif self.evaluation_config["eval_type"] == "us":
            self.generator = model
            self.tokenizer = tokenizer
        else:
            raise ValueError(f"Invalid evaluation type: {self.evaluation_config['eval_type']}")

    def generate_structured_output(self, note):
        """Generate model output for a given user message"""
        try:
            structured = self.outlined_model(
                self.prompt.format(note=note.replace('\n\n', '\n'), examples=self.examples),
                self.output_schema,
                max_new_tokens=512
            )
            
            pred = self.output_schema.model_validate_json(structured).model_dump(mode="json")
            print("Parsed structured output:", structured)
        except Exception as e:
            print("Failed to generate and parse structured output:", e)
            pred = None
                
        return pred

    def generate_unstructured_output(self, notes):
        """Generate model output for a batch of notes (for unstructured output)"""
        # For unstructured output, process as batch
        try:
            # Create batch prompts
            batch_prompts = []
            for note in notes:
                prompt = self.prompt.format(note=note.replace('\n\n', '\n'), examples=self.examples)
                batch_prompts.append(prompt)
            
            # Tokenize all prompts
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=8192,
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = self.generator.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.1,
                    pad_token_id=self.generator.generation_config.eos_token_id[0]
                )
            
            # Decode all outputs
            preds = []
            for i, output in enumerate(outputs):
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Extract the generated part (after the prompt)
                prompt = batch_prompts[i]
                generated_output = generated_text[len(prompt):].strip()
                # print(f"OUTPUT {i}: ", generated_output)
                
                # Find JSON content
                json_match = re.search(r'\{\s*"Employment"\s*:\s*\[.*?],\s*"HousingInstability"\s*:\s*\[.*?],\s*"FoodInsecurity"\s*:\s*\[.*?],\s*"FinancialStrain"\s*:\s*\[.*?],\s*"Transportation"\s*:\s*\[.*?],\s*"Childcare"\s*:\s*\[.*?],\s*"Permanency"\s*:\s*\[.*?],\s*"SubstanceAbuse"\s*:\s*\[.*?],\s*"Safety"\s*:\s*\[.*?]\s*\}', generated_output, re.DOTALL)
                
                if json_match:
                    json_text = json_match.group(0)
                    # print(f"Parsing found JSON text {i}:", json_text.replace('\n', ''))
                    
                    # Try to parse as JSON
                    try:
                        parsed_json = json.loads(json_text)
                        preds.append(parsed_json)
                    except Exception as e:
                        print(f"Failed to parse JSON {i}")
                        preds.append(None)

                    
                else:
                    print(f"No JSON content found for output {i}")
                    preds.append(None)
                    
        except Exception as e:
            print(f"Failed to generate batch output: {e}")
            return preds + [None] * (self.batch_size - len(preds))
                
        return preds
    
    def evaluate(self, dataloader):
        """Evaluate model on a batch of data"""
        try:
            preds = []
            targets = []
            broken_indices = []

            for idx, batch in enumerate(dataloader):
                if (
                    self.max_batches
                    and idx >= self.max_batches
                ):
                    break

                print(f"\nbatch {idx + 1} of {self.max_batches if self.max_batches else len(dataloader)}")
                notes, labels = batch["note"], batch["labels"]

                if self.evaluation_config["eval_type"] == 's':
                    # Process batch
                    batch_preds = []
                    for note in notes:
                        batch_preds.append(self.generate_structured_output(note))
                elif self.evaluation_config["eval_type"] == 'us':
                    # Process batch
                    batch_preds = self.generate_unstructured_output(notes)
                else:
                    raise ValueError("Invalid eval type")
            
                # Process each prediction in the batch
                for i, (pred, label) in enumerate(zip(batch_preds, labels)):
                    print(f"\nNOTE {i}: {notes[i][:50].replace(chr(10), '')}...\n")
                    try:
                        if pred is None:
                            print(f"Failed to generate output for sample {idx}_{i}")
                            broken_indices.append(f"{idx}_{i}")
                            continue
                   
                        print(pred)
                    
                        pred_annotations = get_annotations(pred)
                        preds.append(pred_annotations)
                        targets.append(label)
                        print(f"\nprediction: {pred_annotations} \ntarget: {label.tolist()}")
                    except Exception as e:
                        print(f"Error getting annotations: {e} {pred}")
                        broken_indices.append(f"{idx}_{i}")
        except Exception as e:
            print("method failed, returning early")
            return preds, targets, broken_indices
            
        return preds, targets, broken_indices

    def compute_and_save_metrics(self, preds, targets, broken_indices):
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
        os.makedirs(f"results/{self.output_dir}", exist_ok=True)
        
        file_name = self.prompt_path.split('_')[0] + "_" + str(1 - int(self.evaluation_config["zero_shot"]))+ "_shot"

        i = 0
        while os.path.exists(f"results/{self.output_dir}/{file_name}_{i}.json"):
            i += 1
        
        print(f"\n\nSaving metrics to: results/{self.output_dir}/{file_name}_{i}.json") 
            
        with open(f"results/{self.output_dir}/{file_name}_{i}.json", "w") as f:
            json.dump(metrics, f, indent=2)

def main():
    """Main evaluation function
    
    Usage examples:
    # Broad evaluation, zero-shot, structured output
    python evaluation.py --model-id "meta-llama/Llama-3.3-70B-Instruct" --broad --zero-shot
    
    # Granular evaluation, few-shot, unstructured output
    python evaluation.py --model-id "meta-llama/Llama-3.3-70B-Instruct" --granular --few-shot --unstructured
    
    # Defaults: granular, zero-shot, structured
    python evaluation.py --model-id "meta-llama/Llama-3.3-70B-Instruct"
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate model with command line arguments')
    parser.add_argument('--broad', action='store_true', help='Use broad evaluation')
    parser.add_argument('--granular', action='store_true', help='Use granular evaluation (default if neither specified)')
    parser.add_argument('--zero-shot', action='store_true', help='Use zero-shot evaluation')
    parser.add_argument('--few-shot', action='store_true', help='Use few-shot evaluation (default if neither specified)')
    parser.add_argument('--model-id', type=str, required=True, help='Model ID (overwrites config)')
    parser.add_argument('--eval-type', type=str, required=True, help='Evaluation type (structured or unstructured or vllm)')
    parser.add_argument('--max-batches', type=int, default=None, help='Maximum number of batches to process (default: 5)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing (default: 8)')
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
        "eval_type": args.eval_type  # Use structured unless --unstructured is specified
    }
    
    evaluator = ModelEvaluator(
        model_id=args.model_id, evaluation_config=evaluation_config
    )
    
    # Update max_batches from command line argument
    evaluator.max_batches = args.max_batches
    evaluator.batch_size = args.batch_size
    
    # Print evaluation configuration
    print(f"\nEvaluation Configuration:")
    print(f"Model: {args.model_id}")
    print(f"Broad evaluation: {evaluation_config['broad']}")
    print(f"Zero-shot: {evaluation_config['zero_shot']}")
    print(f"Evaluation type: {evaluation_config['eval_type']}")
    print(f"Max batches: {args.max_batches}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Load data
    dataloader = get_dataloaders(
        batch_size=args.batch_size, split=False, zero_shot=evaluation_config["zero_shot"]
    )

    # Evaluate model
    preds, targets, broken_indices = evaluator.evaluate(dataloader)

    # Compute and save metrics
    evaluator.compute_and_save_metrics(preds, targets, broken_indices)


if __name__ == "__main__":
    main()
