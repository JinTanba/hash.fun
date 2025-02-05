import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Set a fixed random seed for reproducibility (for any random projections, etc.)
np.random.seed(42)

# Load a pre-trained Sentence-BERT model (all nodes should load the same model)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example sentences to index
sentences = [
   "Analyze the socioeconomic implications of remote work adoption in Fortune 500 companies between 2020-2024, focusing on productivity metrics, employee satisfaction, and organizational culture transformation.",
   
   "Design a comprehensive sustainability strategy for a multinational retail corporation that addresses supply chain emissions, packaging waste reduction, and renewable energy implementation while maintaining profitability.",
   
   "Evaluate the impact of artificial intelligence integration in healthcare diagnostics, considering ethical implications, accuracy rates compared to human practitioners, and cost-effectiveness across different medical specialties.",
]

# Generate embeddings for the sentences
embeddings = model.encode(sentences)

# Normalize embeddings for cosine similarity (if you plan on using cosine or LSH based on dot product)
faiss.normalize_L2(embeddings)

# Create an LSH index:
# - embedding_dim: dimensionality of the embeddings (e.g., 384 for 'all-MiniLM-L6-v2')
# - n_bits: number of bits in the hash (adjust based on desired granularity)
embedding_dim = embeddings.shape[1]
n_bits = 128  # You can adjust this value as needed

# Create the LSH index with Faiss
lsh_index = faiss.IndexLSH(embedding_dim, n_bits)

# Add the embeddings to the index
lsh_index.add(embeddings)

# Now, encode a new query sentence and search for similar sentences
query_sentence = "Outline a roadmap for modernizing legacy industrial operations through digital technologies, including automated monitoring systems, machine learning for maintenance optimization, and comprehensive employee training initiatives."
query_embedding = model.encode([query_sentence])
faiss.normalize_L2(query_embedding)

# Search for the top k most similar sentences
k = 4  # Number of nearest neighbors to retrieve
distances, indices = lsh_index.search(query_embedding, k)

print("Query:", query_sentence)
print("Most similar sentences:")
for idx in indices[0]:
    print(f" - {sentences[idx]}")
