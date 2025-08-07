from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import json
import os

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

    def get_matching_classes(self, model_output, k=3):
        # Get embedding for the model output
        while (
            True
        ):  # Loop indefinitely until successful or a non-retryable error occurs
            try:
                input_string_embedding = np.array(
                    self.client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=model_output,
                        config=types.EmbedContentConfig(
                            task_type="SEMANTIC_SIMILARITY"
                        ),
                    )
                    .embeddings[0]
                    .values
                ).reshape(1, -1)
                break
            except ResourceExhausted as e:
                print("waiting for 3 mins")
                time.sleep(180)  # Wait for 3 minutes (180 seconds)

        # Compute cosine similarities
        self.similarities = cosine_similarity(
            input_string_embedding, self.class_embedding_matrix
        )
        print("similarities from cosine function = ", self.similarities)

        similarities_list = self.similarities.tolist()[0]
        top_k_classes = []
        print(f"finding top {k} classes")
        top_k_similarities = sorted(similarities_list, reverse=True)[:k]
        print("top_k_similarities ", top_k_similarities)
        for i in top_k_similarities:
            index_of_similarity = similarities_list.index(i)
            top_k_classes.append(
                list(self.class_embedding_mapping)[index_of_similarity]
            )
            print("class is ", list(self.class_embedding_mapping)[index_of_similarity])
        return top_k_classes, top_k_similarities

    def get_topk_result(self, ground_truth, top_k_classes, k=3):
        if ground_truth in top_k_classes:
            return True
        else:
            return False


if __name__ == "__main__":
    # TODO : adjust the following arguments for this
    get_predicted_class(sys.argv[1])
