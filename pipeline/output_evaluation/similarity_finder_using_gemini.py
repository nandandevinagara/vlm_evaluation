from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import json

#input_string = sys.argv[1]


#print('dimensions of input_string_embedding is ', input_string_embedding.shape)
#print('dimensions of class_embeddings is ', class_embedding_matrix.shape)



def get_class_embedding_matrix(dataset_json_filename):
    # Load class embeddings
    with open(dataset_json_filename, "r") as f:
        embeddings_dict = json.load(f)
    # Sort keys to get consistent ordering
    sorted_keys = sorted(embeddings_dict.keys())
    # Convert to NumPy array (list of embeddings)
    class_embedding_matrix = np.array([embeddings_dict[key] for key in sorted_keys])
    return class_embedding_matrix

def get_predicted_class(google_api_key, model_output, class_embedding_matrix):
    #client = genai.Client(api_key='AIzaSyCG4RITfY0CIiHWiAxCfHzgIdZNqw1WCCU')
    client = genai.Client(api_key=google_api_key)
    input_string_embedding = np.array(client.models.embed_content(
        model="gemini-embedding-001",
        contents=input_string, 
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings[0].values)
    input_string_embedding = input_string_embedding.reshape(1, -1) # Reshape to (1, D) for cosine_similarity
    
    # Calculate cosine similarity. Higher scores = greater semantic similarity.
    similarities = cosine_similarity(input_string_embedding, class_embedding_matrix)
    print('similarities = ', similarities )

    # --- Find the class with the highest similarity ---
    max_similarity_index = np.argmax(similarities)
    highest_similarity_score = similarities[0][max_similarity_index]
    keys_list = list(embeddings_dict.keys())
    best_match_class = keys_list[max_similarity_index]

    print("\n--- Results ---")
    print(f"Input string: \"{input_string}\"")
    print(f"Best matching class: {best_match_class}")
    print(f"Cosine Similarity Score: {highest_similarity_score:.4f}")
    return best_match_class

if __name__ == "__main__":
    get_predicted_class(sys.argv[1])
