# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

hf_token = os.getenv("HF_TOKEN")

model_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# pipe = pipeline(
#     "text-generation", model=model_id, use_auth_token=hf_token, trust_remote_code=True, device=0
# )


prompt_path = "prompts/PURM25.txt"

with open(prompt_path, "r") as f:
    system_prompt = f.read()

user_message = "## Annotation task\nNow with this information, annotate the following clinical note, using the output format specified.\n\nThe patient's father has chosen not to work due to mental health concerns and instead focuses on his treatment plan, with adequate financial support from family members, allowing them to maintain stability in their lives."

# result = pipe(system_prompt + "\n\n" + user_message)
# print(result[0]["generated_text"])


inputs = tokenizer(system_prompt + "\n\n" + user_message, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=256)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))