from transformers import pipeline

prompt_path = "prompts/PURM25.txt"

with open(prompt_path, "r") as f:
    system_prompt = f.read()

user_message = "## Annotation task\nNow with this information, annotate the following clinical note, using the output format specified.\n\nThe patient's father has chosen not to work due to mental health concerns and instead focuses on his treatment plan, with adequate financial support from family members, allowing them to maintain stability in their lives."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_message},
]

mistral_model = "mistralai/Mistral-7B-Instruct-v0.3"

chatbot = pipeline("text-generation", model=mistral_model)
result = chatbot(messages)
print(result)
# print(result[0]["generated_text"])