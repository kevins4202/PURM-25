from dataloader import get_dataloaders
from metrics import compute_metrics_per_label, compute_macro_metrics
from config import MODEL_CONFIG, CATEGORY_MAPPING, EVALUATION_CONFIG
from utils import PromptManager, OutputParser, create_evaluation_summary, save_results
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configuration
torch.set_float32_matmul_precision("high")


class ModelEvaluator:
    def __init__(self, model_config=None, prompt_template_name="broad_0_shot"):
        self.model_config = model_config or MODEL_CONFIG
        self.prompt_manager = PromptManager(prompt_template_name)
        self.output_parser = OutputParser(CATEGORY_MAPPING)
        self.model = None
        self.tokenizer = None
        self._load_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Load and configure the model"""
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config["model_id"],
            quantization_config=quantization_config,
            device_map=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config["model_id"])

    def generate_output(self, user_message):
        """Generate model output for a given user message"""
        input_text = self.prompt_manager.format_prompt(user_message)

        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_length = input_ids["input_ids"].shape[1]

        output = self.model.generate(
            **input_ids, max_new_tokens=self.model_config["max_new_tokens"]
        )

        generated_tokens = output[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def evaluate(self, dataloader):
        """Evaluate model on a batch of data"""
        preds = []
        targets = []
        broken_indices = []

        for idx, batch in enumerate(dataloader):
            if self.model_config["max_batches"] and idx >= self.model_config["max_batches"]:
                break

            print(f"batch {idx + 1} of {len(dataloader)}")
            notes, labels = batch["note"], batch["labels"]

            for i in range(len(notes)):
                print(f"\n\nNOTE: {notes[i][:20]}...")
                output = self.generate_output(notes[i])
                pred = self.output_parser.parse_output(output)
                print(f"prediction: {pred} target: {labels[i]}")

                if pred is not None:
                    preds.append(pred)
                    targets.append(labels[i])
                    print(f"prediction: {pred} target: {labels[i]}")
                else:
                    print("invalid output")
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
        broad_metrics = compute_macro_metrics(preds, targets)
        print("Macro metrics:", broad_metrics)

        # Save results
        save_results(metrics, broad_metrics, summary, output_dir)

        return metrics, broad_metrics, summary


def main():
    """Main evaluation function"""
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_config=MODEL_CONFIG, prompt_template_name="broad_0_shot"
    )

    # Load data
    dataloader = get_dataloaders(
        batch_size=EVALUATION_CONFIG["batch_size"], split=False
    )

    # Evaluate model
    preds, targets, broken_indices = evaluator.evaluate(
        dataloader
    )

    # Compute and save metrics
    evaluator.compute_and_save_metrics(preds, targets, broken_indices)


if __name__ == "__main__":
    main()
