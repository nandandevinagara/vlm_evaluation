import sys
import os
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

genai.configure(api_key='AIzaSyCG4RITfY0CIiHWiAxCfHzgIdZNqw1WCCU')

# --- 2. Define the sports classes and the embedding model ---
SPORTS_CLASSES = [
    "basketball",
    "football", # Assuming Soccer (association football) for clarity
    "badminton",
    "tabletennis",
    "cricket"
]

# Using the free-tier embedding model
EMBEDDING_MODEL = 'gemini-embedding-001'

# --- Function to get embeddings ---
def get_embedding(text_list, task_type="retrieval_document"):
    """Generates embeddings for a list of texts using the Gemini API."""
    try:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text_list,
            task_type=task_type
        )
        return np.array(response['embedding'])
    except Exception as e:
        print(f"Error generating embedding: {e}", file=sys.stderr)
        sys.exit(1)

# --- Main execution ---
if __name__ == "__main__":
    # --- Check for CLI argument ---
    if len(sys.argv) < 2:
        print("Usage: python sport_classifier.py \"Your random string here\"")
        sys.exit(1)

    input_string = sys.argv[1]
    print(f"Input string: \"{input_string}\"")

    # --- Generate embeddings for sports classes (documents) ---
    print("Generating embeddings for sport classes...")
    # Use "retrieval_document" for the classes themselves
    class_embeddings = get_embedding(SPORTS_CLASSES, task_type="retrieval_document")
    # Reshape for `cosine_similarity` if only one class is processed at a time
    # For multiple, it returns (N, D) which is already fine.

    # --- Generate embedding for the input string (query) ---
    print("Generating embedding for input string...")
    # Use "retrieval_query" for the string you're comparing against documents
    input_embedding = get_embedding([input_string], task_type="retrieval_query")[0] # Get the single embedding from the list
    input_embedding = input_embedding.reshape(1, -1) # Reshape to (1, D) for cosine_similarity

    # --- Calculate Cosine Similarity ---
    # cosine_similarity expects inputs of shape (n_samples_1, n_features) and (n_samples_2, n_features)
    # where n_features is the embedding dimension.
    # class_embeddings is (5, D), input_embedding is (1, D)
    similarities = cosine_similarity(input_embedding, class_embeddings)

    # `similarities` will be a 2D array like [[sim1, sim2, sim3, sim4, sim5]]
    # We want the 1D array of scores:
    scores = similarities[0]

    # --- Find the class with the highest similarity ---
    max_similarity_index = np.argmax(scores)
    highest_similarity_score = scores[max_similarity_index]
    best_match_class = SPORTS_CLASSES[max_similarity_index]

    print("\n--- Results ---")
    print(f"Input string: \"{input_string}\"")
    print(f"Best matching class: {best_match_class}")
    print(f"Cosine Similarity Score: {highest_similarity_score:.4f}")

    # Optional: Print all scores for debugging/insight
    # print("\nAll similarity scores:")
    # for i, score in enumerate(scores):
    #     print(f"  {SPORTS_CLASSES[i]}: {score:.4f}")