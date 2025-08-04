import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import subprocess
import json

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# === SETTINGS ===
#DATA_PATH = "data/primary-care_output_file.csv"  # CSV with 'text' and 'label' columns
DATA_PATH = "primary-care_testset/primary-care_output_file_new.csv"  # CSV with 'text' and 'label' columns
OUTPUT_DIR = "primary-care_testset"


# Load dataset
df = pd.read_csv(DATA_PATH)
df = df[["NOTE_ID", "NOTE_TEXT", "PAT_ENC_CSN_ID"]].dropna()

df["text"] = df["NOTE_TEXT"]

# Final columns needed
df = df[["NOTE_ID", "text", "PAT_ENC_CSN_ID"]]

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

fold_dir = OUTPUT_DIR

train_df = df



def apply_prompt(row):
    instruction = f"""### Role:
        You are a doctor analyzing clinical notes.

        ### Objective:
        Determine whether the note describes or mentions bruises.

        ### Instructions:
        Classify the note into one of the following labels:

        - **bruise**: If the note contains text related to bruises, including terms such as *ecchymosis*, *contusion*, *hematoma*, *lesion*, or descriptions like a *concerning mark*.
        - **none**: If no bruises or bruise-related text are mentioned.

        If you choose **bruise**, also extract the specific sentence or phrase that triggered this classification.

        ### Answer Format:
        Respond with only one of the following labels in **JSON** format:
            
        {{"Answer":(<bruise>|<none>),"trigger":<relevant sentence or phrase>}}

        ### Question:
        Does the following clinical note mention a bruise?\n\n{row['text']}"""


    output = ''

    #If you choose **bruise**, also extract the specific sentence or phrase that triggered this classification.
    #{{"Answer":"<bruise>|<none>","trigger":"<relevant sentence or phrase>"}}
    #{{'Answer':(<bruise>|<none>)}}
    return {
        "instruction": instruction,
        "input": "",
        "note_id":row['NOTE_ID'],
        "pat_enc_csn_id":row['PAT_ENC_CSN_ID'],
        "output": output
    }

train_json = train_df.apply(apply_prompt, axis=1).tolist()

train_path = os.path.join(fold_dir, "primary-care_testset_new.json")

with open(train_path, "w") as f:
    for item in train_json:
        f.write(json.dumps(item) + "\n")


# Write dataset_info.json
dataset_info = {
    "train": {"file_name": "primary-care_testset_new.json"}
}
with open(os.path.join(fold_dir, "dataset_info.json"), "w") as f:
    json.dump(dataset_info, f, indent=2) 

