import csv
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

model_path = "/mnt/isilon/llm_collaborative_deidentified/Llama3/Llama-3.3-70B-Instruct"
notes_directory = "./original_files/stage2/Emma_Olshin"
output_file = "./LLaMa3.3-70B_prompt1/stage2/Emma_Olshin/sdoh_classification_results.csv"

# Initialize LLM
llm = LLM(
    model=model_path,
    dtype="bfloat16",
    tensor_parallel_size=8,
    trust_remote_code=True,
    max_model_len=5000,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=200,
)

# Load the prompt template
with open("prompt_template_all.txt", "r") as f:
    prompt_template = f.read()

# Prepare to store results
results = []

# Process each `.txt` file in the directory
for filename in os.listdir(notes_directory):
    if filename.endswith(".txt"):
        note_id = filename  # Use the filename as the note ID
        file_path = os.path.join(notes_directory, filename)

        # Read the clinical note from the file
        with open(file_path, "r") as f:
            clinical_note = f.read().strip()

        # Format the prompt by inserting the clinical note
        prompt = prompt_template.replace("{{CLINICAL_NOTE}}", clinical_note)

        # Run inference
        try:
            output = llm.generate(prompt, sampling_params)
            generated_text = output[0].outputs[0].text.strip()  # Extract the generated labels
        except Exception as e:
            print(f"Error processing {note_id}: {e}")
            generated_text = "Error"

        # Append the result
        results.append({
            "note_id": note_id,
            "labels": generated_text
        })

# Save the results to a CSV file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["note_id", "note", "labels"])
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {output_file}")
