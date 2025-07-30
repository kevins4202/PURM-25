"""
Utility functions for the PURM evaluation system.
Contains helper functions for prompt management, string formatting, and data processing.
"""

import os
from typing import List, Tuple
from config import OUTPUT_TO_CAT, CAT_TO_I
import re

def load_prompt(broad: bool, zero_shot: bool) -> Tuple[str, str]:
    """Load prompt from file"""
    prompt_path = "broad" if broad else "granular" + str(int(zero_shot)) + "_shot.txt"
    print("Using prompt: ", prompt_path)
    if not os.path.exists(f"prompts/{prompt_path}"):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(f"prompts/{prompt_path}", "r") as f:
        prompt = f.read()
        
    examples = ""
    
    if not zero_shot:
        print("Adding shots...")
        examples_path = "broad_shots.txt" if broad else "granular_shots.txt"
        with open(f"prompts/{examples_path}", "r") as f:
            examples = f.read()

    return prompt_path, prompt, examples

def get_annotations(output_json) -> List:
    annotations = [0] * len(CAT_TO_I)

    for output_cat, internal_cat in OUTPUT_TO_CAT.items():
        output_json[output_cat] = [x for x in output_json[output_cat] if len(re.sub(r'[^\w\s]', '', x[-2]).strip()) > 0]
        
        # Determine annotation value
        if any(item[-1] == "social need" for item in output_json[output_cat]):
            # Find the index of the internal category
            annotations[CAT_TO_I[internal_cat]] = 1
        elif any(item[-1] == "no social need" for item in output_json[output_cat]):
            annotations[CAT_TO_I[internal_cat]] = -1

    return annotations


