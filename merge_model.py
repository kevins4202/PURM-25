from transformers import AutoModelForCausalLM
from peft import PeftModel
import os

base = "/mnt/isilon/tsui_lab/hans2/models--mistralai--Mixtral-8x7B-Instruct-v0.1"
lora_path = f"outputs_train_full/model_mixtral"

out_path = f"outputs_train_full/model_merged_mixtral"

base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
lora_model = PeftModel.from_pretrained(base_model, lora_path)
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained(out_path)
