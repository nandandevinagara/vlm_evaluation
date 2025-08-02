import os
import importlib.util
from datetime import datetime


def load_model_module(model_name):
    model_path = os.path.join("vision_models", model_name, f"{model_name}.py")
    spec = importlib.util.spec_from_file_location(model_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_prompt(dataset_name):
    with open(f"dataset/{dataset_name}/{dataset_name}_prompt.txt") as f:
        prompt = f.readlines()
    # NOTE: readlines returns a list, hence take the first index assuming it is having prompt in one line
    return prompt[0]


def get_timestamped_filename(prefix, model_name, dataset_name, extension):
    now = datetime.now()
    timestamp = now.strftime("%H_%M_%d_%m_%Y")
    return f"{prefix}_{model_name}_{timestamp}.{extension}"
