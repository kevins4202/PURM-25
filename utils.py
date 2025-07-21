"""
Utility functions for the PURM evaluation system.
Contains helper functions for prompt management, string formatting, and data processing.
"""

import json
import os
from typing import Dict, List, Optional, Any
from config import PROMPT_TEMPLATES, CATEGORY_MAPPING


class PromptManager:
    """Manages prompt loading, formatting, and validation"""
    
    def __init__(self, prompt_template_name: str):
        self.prompt_template_name = prompt_template_name
        self.prompt_path = PROMPT_TEMPLATES[prompt_template_name]["path"]
        self.system_prompt = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """Load prompt from file"""
        if not os.path.exists(self.prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")
        
        with open(self.prompt_path, "r") as f:
            return f.read()
    
    def format_prompt(self, note: str) -> str:
        """
        Format the prompt with proper string formatting
        
        Args:
            user_message: The user message to include in the prompt
            include_instruction: Whether to include the JSON instruction
            
        Returns:
            Formatted prompt string
        """
        prompt = self.system_prompt.format(note=note)
        
        return prompt
    
    def validate_output_format(self, output: str) -> bool:
        """
        Validate that the output follows the expected JSON format
        
        Args:
            output: Model output string
            
        Returns:
            True if format is valid, False otherwise
        """
        try:
            # Extract JSON from response
            start_idx = output.find('{')
            end_idx = output.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                return False
            
            json_str = output[start_idx:end_idx + 1]
            parsed = json.loads(json_str)
            
            return True
            
        except (json.JSONDecodeError, KeyError, TypeError):
            return False


class OutputParser:
    """Handles parsing and validation of model outputs"""
    
    def __init__(self, category_mapping: Dict[str, str]):
        self.category_mapping = category_mapping
    
    def parse_output(self, pred: str, granular: bool = False) -> Optional[List[int]]:
        """
        Parse model output and convert to annotation format
        
        Args:
            pred: Model prediction string
            granular: Whether this is granular classification
            
        Returns:
            List of annotations or None if parsing failed
        """
        try:
            assert isinstance(pred, str)
            pred = pred.strip()
            print(f"Parsing output: {chr(10)}{pred}{chr(10)}")
            
            # Extract JSON from response
            i1 = pred.index('{')
            i2 = pred.index('}')
            pred = pred[i1:i2+1]
            
            dct = json.loads(pred)
            annotations = [0] * len(self.category_mapping)

            for output_cat, internal_cat in self.category_mapping.items():
                if output_cat in dct:
                    assert isinstance(dct[output_cat], list)
                    dct[output_cat] = [x for x in dct[output_cat] if len(x) > 0]
                    
                    # Validate format
                    assert all(
                        isinstance(item, list) and 
                        len(item) == 2 + int(granular) and 
                        item[-1] in ["positive", "negative"] 
                        for item in dct[output_cat]
                    )

                    # Determine annotation value
                    if any(item[-1] == "positive" for item in dct[output_cat]):
                        # Find the index of the internal category
                        internal_cat_index = list(self.category_mapping.values()).index(internal_cat)
                        annotations[internal_cat_index] = 1
                    elif any(item[-1] == "negative" for item in dct[output_cat]):
                        internal_cat_index = list(self.category_mapping.values()).index(internal_cat)
                        annotations[internal_cat_index] = -1
            
            return annotations
            
        except Exception as e:
            print(f"Error parsing output: {e}")
            return None

def create_evaluation_summary(preds: List[List[int]], targets: List[List[int]], 
                            broken_indices: List[int]) -> Dict[str, Any]:
    """
    Create a summary of evaluation results
    
    Args:
        preds: List of predictions
        targets: List of targets
        broken_indices: List of indices where parsing failed
        
    Returns:
        Summary dictionary
    """
    total_samples = len(broken_indices) + len(preds)
    successful_samples = len(preds)
    failed_samples = len(broken_indices)
    
    return {
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "failed_samples": failed_samples,
        "success_rate": successful_samples / total_samples if total_samples > 0 else 0.0,
        "broken_indices": broken_indices
    }


def save_results(metrics: Dict, broad_metrics: Dict, summary: Dict, 
                output_dir: str = ".") -> None:
    """
    Save evaluation results to files
    
    Args:
        metrics: Per-label metrics
        broad_metrics: Macro metrics
        summary: Evaluation summary
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save per-label metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save macro metrics
    with open(os.path.join(output_dir, "broad_metrics.json"), "w") as f:
        json.dump(broad_metrics, f, indent=2)
    
    # Save evaluation summary
    with open(os.path.join(output_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {output_dir}/") 
