from dataloader import get_dataloaders
from metrics import compute_metrics_per_label, compute_macro_metrics
from config import MODEL_CONFIG, EVALUATION_CONFIG
from utils import (
    create_evaluation_summary,
    load_prompt,
    save_results,
    BroadOutputSchema,
    GranularOutputSchema,
)
from outlines import Outline, models, generate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configuration
torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"


class ModelEvaluator:
    def __init__(self, model_config, evaluation_config):
        print("Initializing model")
        self.model_config = model_config
        self.evaluation_config = evaluation_config
        self.outlined_model = None
        self.generator = None
        self._load_model()

    def _load_model(self):
        """Load and configure the model"""
        print("Loading and quantizing model")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config["model_id"],
            quantization_config=quantization_config,
            device_map=device,
        )
        model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]
        tokenizer = AutoTokenizer.from_pretrained(self.model_config["model_id"])
        # Wrap with outlines
        self.outlined_model = models.transformers(model, tokenizer)

        self.generator = generate.json(
            self.outlined_model,
            Outline(
                BroadOutputSchema
                if self.evaluation_config["broad"]
                else GranularOutputSchema
            ),
        )

    def generate_output(self, note):
        """Generate model output for a given user message"""
        print("Generating output...")
        # Use outlines to generate structured output
        try:
            structured = self.generator(
                load_prompt(
                    self.evaluation_config["broad"],
                    self.evaluation_config["zero_shot"],
                    note,
                )
            )
            print("Structured output:", structured)
        except Exception as e:
            print("Failed to generate structured output:", e)
            structured = None
        return structured

    def evaluate(self, dataloader):
        """Evaluate model on a batch of data"""
        print("Starting evaluation")
        preds = []
        targets = []
        broken_indices = []

        for idx, batch in enumerate(dataloader):
            if (
                "max_batches" in self.model_config
                and idx >= self.model_config["max_batches"]
            ):
                break

            print(f"\nbatch {idx + 1} of {len(dataloader)}")
            notes, labels = batch["note"], batch["labels"]

            for i in range(len(notes)):
                print(f"NOTE: {notes[i][:50].replace(chr(10), '')}...")
                pred = self.generate_output(notes[i])
                try:
                    preds.append(pred.dict())
                    targets.append(labels[i])
                    print(f"prediction: {pred} target: {labels[i]}")
                except Exception as e:
                    print(f"Error parsing output: {e} {pred}")
                    broken_indices.append(idx)

        return preds, targets, broken_indices

    def compute_and_save_metrics(self, preds, targets, broken_indices, output_dir="."):
        """Compute metrics and save results"""
        if len(preds) != len(targets):
            print(
                f"Lengths of preds and targets do not match: {len(preds)} vs {len(targets)}"
            )
            return

        # Create evaluation summary
        summary = create_evaluation_summary(preds, targets, broken_indices)
        print(f"\n\nEvaluation Summary:")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Successful samples: {summary['successful_samples']}")
        print(f"Failed samples: {summary['failed_samples']}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Broken indices: {summary['broken_indices']}")

        # Compute per-label metrics
        metrics = compute_metrics_per_label(preds, targets)
        print("\nPer-label metrics:", metrics)

        # Compute macro metrics
        broad_metrics = compute_macro_metrics(metrics)
        print("Macro metrics:", broad_metrics)

        # Save results
        save_results(metrics, broad_metrics, summary, output_dir)

        return metrics, broad_metrics, summary


def main():
    """Main evaluation function"""
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_config=MODEL_CONFIG, evaluation_config=EVALUATION_CONFIG
    )

    # Load data
    dataloader = get_dataloaders(
        batch_size=EVALUATION_CONFIG["batch_size"], split=False
    )

    # Evaluate model
    preds, targets, broken_indices = evaluator.evaluate(dataloader)

    # Compute and save metrics
    evaluator.compute_and_save_metrics(preds, targets, broken_indices)


if __name__ == "__main__":
    main()
