from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#genai.configure()
client = genai.Client(api_key='AIzaSyCG4RITfY0CIiHWiAxCfHzgIdZNqw1WCCU')

SPORTS_CLASSES = [
    "basketball",
    "football", # Assuming Soccer (association football) for clarity
    "badminton",
    "tabletennis",
    "cricket"
]

result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=SPORTS_CLASSES,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings)

class_embeddings = np.array(result.embeddings)
print(result['embedding'])
exit()
#print(class_embeddings)
#print(np.array(result.embeddings).shape)

# Step 4: Embed the query string
query = "sdsdjaksjd"
query_response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query])
query_embedding = np.array(query_response.embeddings)[0].reshape(1, -1)  # shape: (1, 768)



