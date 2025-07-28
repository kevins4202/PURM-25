from dataloader import get_dataloaders
from metrics import compute_metrics
from config import MODEL_CONFIG, EVALUATION_CONFIG, BroadOutputSchema, GranularOutputSchema
from utils import (
    load_prompt,
    get_annotations,
)
import outlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import os
import argparse

# Configuration
torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"


class ModelEvaluator:
    def __init__(self, model_config, evaluation_config):
        self.model_config = model_config
        self.model_id = self.model_config["model_id"].split("/")[-1]
        self.evaluation_config = evaluation_config
        self.outlined_model = None
        self.generator = None
        self.output_schema = BroadOutputSchema if self.evaluation_config["broad"] else GranularOutputSchema
        self._load_model()
        self.prompt_path, self.prompt = load_prompt(
            self.evaluation_config["broad"],
            self.evaluation_config["zero_shot"]
        )
        self.max_batches = self.evaluation_config["max_batches"]

    def _load_model(self):
        """Load and configure the model"""
        print("Loading and quantizing model ", self.model_id)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config["model_id"],
            quantization_config=quantization_config,
            device_map=device,
        )
        model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]
        tokenizer = AutoTokenizer.from_pretrained(self.model_config["model_id"])
        # Wrap with outlines
        self.outlined_model = outlines.from_transformers(model, tokenizer)

    def generate_output(self, note):
        """Generate model output for a given user message"""
        # Use outlines to generate structured output
        try:
            structured = self.outlined_model(
                self.prompt.format(note=note),
                self.output_schema,
                max_new_tokens=512
            )
            
            pred = self.output_schema.model_validate_json(structured).model_dump(mode="json")
            print("Parsed structured output:", structured)
        except Exception as e:
            print("Failed to generate and parse structured output:", e)
            pred = None
        return pred

    def evaluate(self, dataloader):
        """Evaluate model on a batch of data"""
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

            for i in range(len(notes)):
                print(f"NOTE: {notes[i][:50].replace(chr(10), '')}...")
                pred = self.generate_output(notes[i])
                try:
                    preds.append(get_annotations(pred))
                    targets.append(labels[i])
                    print(f"prediction: {preds[-1]} \ntarget: {labels[i].tolist()}")
                except Exception as e:
                    print(f"Error parsing output: {e} {pred}")
                    broken_indices.append(idx)

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
        os.makedirs(f"results/{self.model_id}", exist_ok=True)

        i = 0
        while os.path.exists(f"results/{self.model_id}/{self.prompt_path}_{i}.json"):
            i += 1

        with open(f"results/{self.model_id}/{self.prompt_path}_{i}.json", "w") as f:
            json.dump(metrics, f, indent=2)

def main():
    """Main evaluation function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate model with command line arguments')
    parser.add_argument('--broad', action='store_true', help='Use broad evaluation (overwrites config)')
    parser.add_argument('--zero_shot', action='store_true', help='Use zero-shot evaluation (overwrites config)')
    args = parser.parse_args()
    
    # Create evaluation config with command line overrides
    evaluation_config = EVALUATION_CONFIG.copy()
    if args.broad is not None:
        evaluation_config["broad"] = args.broad
    if args.zero_shot is not None:
        evaluation_config["zero_shot"] = args.zero_shot
    
    evaluator = ModelEvaluator(
        model_config=MODEL_CONFIG, evaluation_config=evaluation_config
    )

    # Load data
    dataloader = get_dataloaders(
        batch_size=evaluation_config["batch_size"], split=False, zero_shot=evaluation_config["zero_shot"]
    )

    # Evaluate model
    preds, targets, broken_indices = evaluator.evaluate(dataloader)

    # Compute and save metrics
    evaluator.compute_and_save_metrics(preds, targets, broken_indices)


if __name__ == "__main__":
    main()
