import transformers
import torch


def load_model_pipeline(model_id, device="auto"):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
    )

    return pipeline




