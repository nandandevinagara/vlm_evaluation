import torch
import torch.nn.functional as F
from torch import Tensor, argmax
from transformers import AutoTokenizer, AutoModel
import json
import sys

# eg: python embedding_generator_via_linq_embed_mistral.py ucf101_class_list.json
# Ensure the json file is in the format as expected, refer other json files

json_filename = sys.argv[1]
dataset_name = json_filename.rsplit('_class', 1)[0]
# Read the JSON file

try:
    with open(json_filename, "r") as f:
        data = json.load(f)
    # Fetch the 'class_list' field and store it in a local list
    if "class_list" in data:
        class_list_local = data["class_list"]
        print(f"The 'class_list' field successfully extracted: {class_list_local}")
    else:
        print("The 'class_list' field was not found in the JSON file.")

except FileNotFoundError:
    print(
        "Error: 'data.json' not found. Please ensure the file exists in the same directory."
    )
except json.JSONDecodeError:
    print(
        "Error: Could not decode JSON from 'data.json'. Please ensure it's a valid JSON file."
    )
except Exception as e:
    print(f"An unexpected error occurred: {e}")

    
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """This code snippet defines a function last_token_pool that extracts the embedding of the last token from a sequence."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

passages = class_list_local

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")
model = AutoModel.from_pretrained("Linq-AI-Research/Linq-Embed-Mistral")


input_texts = [*passages]
# Tokenize the input texts
batch_dict = tokenizer(
    input_texts, max_length=4096, padding=True, truncation=True, return_tensors="pt"
)
# print('batch_dict =', batch_dict)
outputs = model(
    **batch_dict
)  # dicitonary, hence **, which is used as for loop at the models end, ** is not unpacked automatically, but list is unpacked automatically
# print((outputs.last_hidden_state))
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
print(
    "embeddings shape is = ", embeddings.shape
)  # torch.Size([102, 4096]) with 1 + 101 words

# print('before normalizing ', embeddings)
# Normalize embeddings
embeddings = F.normalize(
    embeddings, p=2, dim=1
)  # normalization is better to reduce values from 1.175 etc to 0.45 etc, still i was getting 48% etc
# print('after normalizing ', embeddings)
embeddings_list = embeddings.tolist()
# Save as JSON
embeddings_dict = dict(zip(input_texts, embeddings_list))
    

output_filename = f'{dataset_name}_embeddings.json'
try:
    with open(output_filename, "w") as f:
        json.dump(embeddings_dict, f, indent=2)
    print(f"\nSuccessfully dumped data to '{output_filename}'")
except Exception as e:
    print(f"Error dumping to JSON: {e}")
