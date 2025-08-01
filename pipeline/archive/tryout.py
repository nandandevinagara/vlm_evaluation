from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys

client = genai.Client(api_key='AIzaSyCG4RITfY0CIiHWiAxCfHzgIdZNqw1WCCU')

SPORTS_CLASSES = [
    "basketball",
    "football", # Assuming Soccer (association football) for clarity
    "badminton",
    "tabletennis",
    "cricket"
]

#print("embedding of purple is ", type(client.models.embed_content(
#        model="gemini-embedding-001",
#        contents="purple", 
#        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings[0].values))
#embedding of purple is  [ContentEmbedding(
#  values=[
#    -0.011449482,
#    0.009784373,
#    <... 3067 more items ...>,
#  ]
#)]
#ContentEmbedding is a tuple probably, hence we can iterate [0] using 'embeddings' word and then in that 0th index, fetch values, which is just an attribute
#the above print prints the whole embedding of 'purple' string which is just a list, it is converted into an numpy array of size 3072, 1 , a vector
#the following is the numpy array and note that the last digit is removed
#print("embedding of purple is ", np.array(client.models.embed_content(
#        model="gemini-embedding-001",
#        contents="purple", 
#        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings[0].values))


class_embeddings = [
    np.array(e.values) for e in client.models.embed_content(
        model="gemini-embedding-001",
        contents=SPORTS_CLASSES, 
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings
]

print(class_embeddings)
print(len(class_embeddings))
#I VERIFIED THAT ABOVE PASSING LIST WORKS PERFECTLY FINE AND THERE IS ALREADY A FOR LOOP THAT DOES THE JOB, IT IS CORRECT

#for each_class_member in SPORTS_CLASSES:
#    print(client.models.embed_content(
#    model="gemini-embedding-001",
#    contents=each_class_member, 
#    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings)

input_string = sys.argv[1]
input_string_embedding = np.array(client.models.embed_content(
        model="gemini-embedding-001",
        contents=input_string, 
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings[0].values)
input_string_embedding = input_string_embedding.reshape(1, -1) # Reshape to (1, D) for cosine_similarity, basically something invert thing is done, thats all or else, actually it should be (D,1) when we calculate with hand

print('dimensions of input_string_embedding is ', input_string_embedding.shape)
print('dimensions of class_embeddings is ', class_embeddings[0].shape)
#class_embeddings is list of arrays where each array is (3072,1), cosine_similarity has inbuilt feature where that list of arrrays is converted into a matrix

# Parameters
#    ----------
#    X : {array-like, sparse matrix} of shape (n_samples_X, n_features)
#        Input data.
#
#    Y : {array-like, sparse matrix} of shape (n_samples_Y, n_features),             default=None
#        Input data. If ``None``, the output will be the pairwise
#        similarities between all samples in ``X``.
#            
# Calculate cosine similarity. Higher scores = greater semantic similarity.
similarities = cosine_similarity(input_string_embedding, class_embeddings)
#NOTE HERE THAT THE similarities IS A LIST OF one single LIST; DIRECTLY WE CANNOT DO [4], but we have to do [0][actual_index]
print('similarities = ', similarities )

# --- Find the class with the highest similarity ---
max_similarity_index = np.argmax(similarities)
highest_similarity_score = similarities[0][max_similarity_index]
best_match_class = SPORTS_CLASSES[max_similarity_index]

print("\n--- Results ---")
print(f"Input string: \"{input_string}\"")
print(f"Best matching class: {best_match_class}")
print(f"Cosine Similarity Score: {highest_similarity_score:.4f}")

exit()

