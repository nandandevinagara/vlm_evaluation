import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json
import os
from torch import Tensor, argmax

class_embedding_mapping = {}


class SimilarityFinder:
    def __init__(self, dataset_json_filename):
        self.dataset_json_filename = dataset_json_filename
        self.class_embedding_matrix, self.class_embedding_mapping = (
            self.load_class_embeddings()
        )
        self.tokenizer = AutoTokenizer.from_pretrained('Linq-AI-Research/Linq-Embed-Mistral')
        self.model = AutoModel.from_pretrained('Linq-AI-Research/Linq-Embed-Mistral')
        

    def load_class_embeddings(self):
        # Load class embeddings from file
        with open(self.dataset_json_filename, "r") as f:
            self.class_embedding_mapping = json.load(f)
        #sorted_keys = sorted(class_embedding_mapping.keys())
        #class_embedding_matrix = np.array(
        #    [class_embedding_mapping[key] for key in sorted_keys]
        #)
        # Convert class names and their embeddings into usable tensors
        class_names = list(self.class_embedding_mapping.keys())
        embeddings_values = list(self.class_embedding_mapping.values())
        self.embeddings_tensor = torch.tensor(embeddings_values)
        # self.class_embeddings = torch.tensor([self.class_embedding_mapping[name] for name in class_names])
        # self.class_embeddings = F.normalize(self.class_embeddings, p=2, dim=1)  # Ensure they're normalized
        return self.embeddings_tensor, self.class_embedding_mapping

    def last_token_pool(self, last_hidden_states: Tensor,
             attention_mask: Tensor) -> Tensor:
        """This code snippet defines a function last_token_pool that extracts the embedding of the last token from a sequence."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_matching_classes(self, model_output, k=3):
        # Get embedding for the model output
        input_word= [model_output]
        #passages = 
        max_length = 4096
        # Tokenize the input text
        batch_dict = self.tokenizer(input_word, max_length=4096, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**batch_dict)
        embeddings_of_a_word = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings_of_a_word = F.normalize(embeddings_of_a_word, p=2, dim=1)

        # Compute cosine similarity between query and all class embeddings
        #self.similarities = torch.matmul(query_embedding, self.class_embeddings.T).squeeze(0)
        self.similarities = (embeddings_of_a_word @ self.embeddings_tensor.T) * 100
        print(self.similarities) 
      

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
    sf = SimilarityFinder('ucf101_embeddings.json')
    sf.get_matching_classes('Soldier')
