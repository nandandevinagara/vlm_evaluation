# the code is fine, but it gives high similarity even for garbage values


import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import sys
from transformers import BertTokenizer, BertModel
import torch

# Load class embeddings
with open("class_embeddings.json", "r") as f:
    class_embeddings = json.load(f)

# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Convert to usable format
label_names = list(class_embeddings.keys())
embedding_matrix = np.array([class_embeddings[label] for label in label_names])


# Function to get embedding of a prediction
def get_text_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state[0][0].numpy()


# Example predicted label (from model)
predicted_label = sys.argv[1]

# Get embedding for prediction
pred_embedding = get_text_embedding(predicted_label, tokenizer, model).reshape(1, -1)

print("shape is ", pred_embedding.shape, " ", embedding_matrix.shape)
# Normalize the prediction embedding (shape: (1, D))
pred_embedding = normalize(pred_embedding)

# Normalize each class embedding in the matrix (shape: (N, D))
embedding_matrix = normalize(embedding_matrix)

# Compute similarity with class labels
similarities = cosine_similarity(pred_embedding, embedding_matrix)[0]
print(similarities.shape)
top_idx = int(np.argmax(similarities))
top_label = label_names[top_idx]
top_score = similarities[top_idx]

print(f"Best match: {top_label} (score: {top_score:.4f})")
