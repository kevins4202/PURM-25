import os
import json
import glob
from utils import load_prompt
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# === SETTINGS ===
PROCESSED_DIR = "data/finetuning/granular"
NOTES_DIR = "data/notes"
OUTPUT_DIR = "data/finetuning"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_processed_data():
    """Load all processed JSON files from the granular directory"""
    processed_files = glob.glob(os.path.join(PROCESSED_DIR, "*.json"))
    data = {}
    
    for file_path in processed_files:
        filename = os.path.basename(file_path)
        file = filename.split('.')[0]  # Remove .json extension
        
        with open(file_path, 'r') as f:
            content = json.load(f)
            data[file] = content
    
    return data

def load_note_text(file):
    """Load the corresponding note text file"""
    note_path = os.path.join(NOTES_DIR, f"{file}.txt")
    
    if os.path.exists(note_path):
        with open(note_path, 'r') as f:
            return f.read().strip()
    
    return None

def create_prompt_template(note_text, social_needs_data, file, prompt_template, examples):
    """Create a prompt for social needs classification using the same format as evaluation.py"""
    
    # Format the prompt using the same pattern as evaluation.py
    formatted_prompt = prompt_template.format(
        note=note_text.replace("\n\n", "\n"), 
        examples=examples
    )
    
    return {
        "instruction": formatted_prompt,
        "input": "",
        "file": file,
        "output": json.dumps(social_needs_data)
    }

def main():
    # Load prompt using the same method as evaluation.py
    # Using granular evaluation by default
    BROAD, ZERO_SHOT = False, False
    prompt_path, prompt_template, examples = load_prompt(broad=BROAD, zero_shot=ZERO_SHOT)
    print(f"Using prompt: {prompt_path}")
    
    # Load processed data
    processed_data = load_processed_data()
    print(f"Loaded {len(processed_data)} processed files")
    
    # Create dataset
    dataset = []
    
    for note_file, social_needs in processed_data.items():
        # Load corresponding note text
        note_text = load_note_text(note_file)
        
        if note_text is None:
            print(f"Warning: Could not find note text for {note_file}")
            continue
        
        # Create prompt
        prompt_data = create_prompt_template(note_text, social_needs, note_file, prompt_template, examples)
        dataset.append(prompt_data)
    
    print(f"Created {len(dataset)} dataset entries")
    
    # Write dataset to file
    output_name = f"dataset_{'broad' if BROAD else 'granular'}_{1 - int(ZERO_SHOT)}_shot.json"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    with open(output_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    
    # Write dataset info
    dataset_info = {
        "train": {"file_name": output_name},
    }
    
    with open(os.path.join(OUTPUT_DIR, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Dataset saved to {output_path}")
    print(f"Dataset info saved to {os.path.join(OUTPUT_DIR, 'dataset_info.json')}")

if __name__ == "__main__":
    main() 

