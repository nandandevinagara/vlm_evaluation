from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import json
import os

# input_string = sys.argv[1]

# print('dimensions of input_string_embedding is ', input_string_embedding.shape)
# print('dimensions of class_embeddings is ', class_embedding_matrix.shape)
class_embedding_mapping = {}


class SimilarityFinder:
    def __init__(self, dataset_json_filename, google_api_key):
        self.dataset_json_filename = dataset_json_filename
        self.google_api_key = google_api_key
        self.class_embedding_matrix, self.class_embedding_mapping = (
            self.load_class_embeddings()
        )
        self.client = genai.Client(api_key=self.google_api_key)

    def load_class_embeddings(self):
        # Load class embeddings from file
        with open(self.dataset_json_filename, "r") as f:
            class_embedding_mapping = json.load(f)
        sorted_keys = sorted(class_embedding_mapping.keys())
        class_embedding_matrix = np.array(
            [class_embedding_mapping[key] for key in sorted_keys]
        )
        return class_embedding_matrix, class_embedding_mapping

    def get_matching_class(self, model_output):
        # Get embedding for the model output
        input_string_embedding = np.array(
            self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=model_output,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )
            .embeddings[0]
            .values
        ).reshape(1, -1)

        # Compute cosine similarities
        self.similarities = cosine_similarity(
            input_string_embedding, self.class_embedding_matrix
        )
        print("similarities from cosine function = ", self.similarities)

        # Find the best matching class
        max_similarity_index = np.argmax(self.similarities)
        highest_similarity_score = self.similarities[0][max_similarity_index]
        sorted_keys = sorted(self.class_embedding_mapping.keys())
        best_match_class = sorted_keys[max_similarity_index]

        print("\n--- Results ---")
        print(f'Input string: "{model_output}"')
        print(f"Best matching class: {best_match_class}")
        print(f"Cosine Similarity Score: {highest_similarity_score:.4f}")

        return best_match_class, f"{highest_similarity_score:.4f}"

    def compute_topk_accuracy(self, ground_truth, model_output, k=3):
        top1_correct = 0
        topk_correct = 0
        similarities_list = self.similarities.tolist()[0]
        top_k_classes = []
        # --- Find the class with the highest similarity ---
        # print('finding top 1 class')
        # max_similarity_index = np.argmax(similarities)
        # highest_similarity_score = similarities[0][max_similarity_index]
        # keys_list = list(class_embedding_mapping.keys())
        # best_match_class = keys_list[max_similarity_index]

        # --- Find the class with the highest 3 similarity ---
        print(f"finding top {k} classes")
        top_k_similarities = sorted(similarities_list, reverse=True)[:k]
        print("top_k_similarities ", top_k_similarities)
        for i in top_k_similarities:
            index_of_similarity = similarities_list.index(i)
            top_k_classes.append(
                list(self.class_embedding_mapping)[index_of_similarity]
            )
            print("class is ", list(self.class_embedding_mapping)[index_of_similarity])

        if ground_truth in top_k_classes:
            return True
        else:
            return False


if __name__ == "__main__":
    # TODO : adjust the following arguments for this
    get_predicted_class(sys.argv[1])
