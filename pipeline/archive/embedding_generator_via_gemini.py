# embedding generator
# input : a json file with class list
# output : a json file with class embeddings

import json
import sys, time
import numpy as np
from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyCG4RITfY0CIiHWiAxCfHzgIdZNqw1WCCU")

# Create a dummy JSON file for demonstration purposes
json_filename = sys.argv[1]

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


# generate embeddings for each of the class and dump to a json file
class_embeddings = []
n = len(class_list_local)
part_size = n // 3
remainder = n % 3
# Compute the sizes of the three parts
sizes = [part_size + (1 if i < remainder else 0) for i in range(3)]
start = 0

for i in range(3):
    end = start + sizes[i]
    temp_list = [
        np.array(e.values)
        for e in client.models.embed_content(
            model="gemini-embedding-001",
            contents=class_list_local[start:end],
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        ).embeddings
    ]
    class_embeddings.extend(temp_list)
    start = end
    time.sleep(70)

print(type(class_embeddings))

# Convert each embedding to a Python list
serializable_dict = {
    class_list_local[i]: class_embeddings[i].tolist()
    for i in range(len(class_list_local))
}

# Convert each NumPy array in the list to a Python list
# You can use .tolist() method for this.
### serializable_embeddings_list = [arr.tolist() for arr in class_embeddings]
### print("\nConverted to Python list of lists for JSON serialization:")
### print(len(serializable_embeddings_list[0]))
### print(f"Type of serializable_list: {type(serializable_embeddings_list)}")
### print(f"Type of first element: {type(serializable_embeddings_list[0])}")
###
# Dump the serializable list to a JSON file
output_filename = sys.argv[2]
try:
    with open(output_filename, "w") as f:
        json.dump(serializable_dict, f, indent=4)  # indent for pretty printing
    print(f"\nSuccessfully dumped data to '{output_filename}'")
except Exception as e:
    print(f"Error dumping to JSON: {e}")
