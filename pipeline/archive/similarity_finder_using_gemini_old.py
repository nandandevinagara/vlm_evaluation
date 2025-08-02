import google.generativeai as genai
import os  # For securely loading your API key

# --- IMPORTANT: Configure your API Key ---
# It's highly recommended to load your API key from an environment variable
# to keep it out of your code.
# Alternatively, you can directly assign it: genai.configure(api_key="YOUR_API_KEY_HERE")
# But for security, prefer environment variables.


genai.configure(api_key="AIzaSyCG4RITfY0CIiHWiAxCfHzgIdZNqw1WCCU")

# --- Define the text you want to embed ---
text_to_embed = [
    "This is a test sentence for embedding.",
    "Another example sentence for generating a vector.",
    "Machine learning models are powerful.",
]

# --- Define the embedding model (free tier) ---
# 'models/embedding-001' is the commonly used, free-tier embedding model.
# As of current knowledge, 'gemini-embedding-001' maps to this.
# Always check the latest Google AI documentation for the exact model name.
EMBEDDING_MODEL = "models/embedding-001"

print(f"Generating embeddings using model: {EMBEDDING_MODEL}...")

# --- Generate the embeddings ---
try:
    # For a single string:
    # response = genai.embed_content(model=EMBEDDING_MODEL, content=text_to_embed[0])

    # For a list of strings (batch embedding):
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text_to_embed,
        task_type="retrieval_document",  # Or "retrieval_query", "semantic_similarity", "classification", "clustering"
    )

    # The embeddings are in the 'embedding' field of the response
    embeddings = response["embedding"]

    print("\nEmbeddings generated successfully!")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Dimension of each embedding: {len(embeddings[0])}")

    # Print the first few values of the first embedding to see the structure
    print("\nFirst embedding (partial):")
    print(embeddings[0][:5], "...")  # Print first 5 values

except Exception as e:
    print(f"An error occurred: {e}")
