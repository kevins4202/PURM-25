# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import time
import torch

hf_token = os.getenv("HF_TOKEN")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
# model_id = "mistralai/Mistral-7B-Instruct-v0.3"

begin = time.time()
start_time = time.time()
print(f"Starting...\n\n")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

cuda_available = torch.cuda.is_available()

if cuda_available:
    print("CUDA is available")
    # model.to("cuda")
    print(f"Model moved to GPU\n\n")
else:
    print("CUDA is not available")
    print("Using CPU")

print(f"Model and tokenizer loaded in {time.time() - start_time} seconds\n\n")
start_time = time.time()

prompt_path = "prompts/PURM25.txt"

with open(prompt_path, "r") as f:
    system_prompt = f.read()

user_message = '"Mom reports that she has to return to work on Monday but in need of childcare. She may have to leave child with elderly GM with 16 and 10 year old in the home. In treatment for opioid use disorder, currently on methadone. Mother shared CPS involvement due to recent childline report indicating sexual abuse from father."'

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_message},
]

print("Tokenizing...")

inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
)

if cuda_available:
    inputs = inputs.to("cuda")

print(f"Tokenized in {time.time() - start_time} seconds\n\n")
start_time = time.time()

print("Generating...")
outputs = model.generate(
    inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id
)

print(f"Generated in {time.time() - start_time} seconds\n\n")
start_time = time.time()

print("Decoding...")
input_length = inputs.shape[1]
generated_tokens = outputs[0][input_length:]

decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

output_path = f"output/test/{model_id.split('/')[-1]}_output.txt"

with open(output_path, "w") as f:
    f.write(decoded_output)

print(f"Decoded in {time.time() - start_time} seconds\n\n")

print(f"Done. Total time: {time.time() - begin} seconds")
