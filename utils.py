"""
Utility functions for the PURM evaluation system.
Contains helper functions for prompt management, string formatting, and data processing.
"""

import os
from typing import List, Tuple
from config import OUTPUT_TO_CAT, CAT_TO_I
import re

def get_prompt_path(broad: bool, zero_shot: bool):
    if broad:
        if zero_shot:
            return "broad_0_shot.txt"
        else:
            return "broad_1_shot.txt"
    else:
        if zero_shot:
            return "granular_0_shot.txt"
        else:
            return "granular_1_shot.txt"


def load_prompt(broad: bool, zero_shot: bool) -> Tuple[str, str]:
    """Load prompt from file"""
    prompt_path = get_prompt_path(broad, zero_shot)
    print("Using prompt: ", prompt_path)
    if not os.path.exists(f"prompts/{prompt_path}"):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(f"prompts/{prompt_path}", "r") as f:
        prompt = f.read()

    return prompt_path, prompt

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


