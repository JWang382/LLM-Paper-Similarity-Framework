import pandas as pd
import numpy as np
import os
from openai import OpenAI  # Import OpenAI client
import random

SEED = 2025
np.random.seed(SEED)
random.seed(SEED)

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "data", "top_cited.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "embeddings")

# --- Model Configuration ---
# Example: using text-embedding-3-small (Dim: 1536)
# text-embedding-3-small costs approximately $0.02 per 1M tokens.
MODEL_NAME = "text-embedding-3-small"
OUTPUT_NPY = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.npy")

# --- OpenAI Client Setup ---
# It's better to set OPENAI_API_KEY as an environment variable
client = OpenAI(api_key="your-api-key-here")

# -------------------------------------------------------------------------
# NOTE: HOW TO IMPLEMENT DIFFERENT MODELS
# -------------------------------------------------------------------------
# 1. OpenAI (text-embedding-3-large):
#    - Dim: 3072. 
#    - Implementation: Use 'openai' Python library as well.
#
# 2. SPECTER (allenai/specter):
#    - Dim: 768. 
#    - Implementation: Use 'transformers' library.
#
# 3. ModernBERT (answerdotai/ModernBERT-base):
#    - Dim: 768. 
#    - Implementation: Use 'sentence-transformers'. 
#
# 4. Nomic-Embed (nomic-ai/nomic-embed-text-v1.5):
#    - Dim: 768. 
#
# 5. SciNCL (malteos/scincl):
#    - Dim: 768. 
# -------------------------------------------------------------------------

def generate_embeddings():
    """
    Step 1: Convert paper metadata into semantic embeddings.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Load dataset
    df = pd.read_csv(INPUT_CSV)
    
    # Preparing input text
    texts = df['abstract'].fillna("").tolist()
    
    # 2. Call OpenAI API in batches 
    batch_size = 100   # sample batch size
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # --- Real API Call ---
        response = client.embeddings.create(
            input=batch_texts,
            model=MODEL_NAME
        )
        
        # Extract embeddings from response
        batch_embeddings = [record.embedding for record in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"Progress: {min(i + batch_size, len(texts))}/{len(texts)} processed.")

    # 3. Convert to Numpy Array
    embeddings_array = np.array(all_embeddings).astype('float32')

    # CRITICAL: L2 Normalization 
    # Ensures dot product equals cosine similarity in Step 3
    embeddings_array /= np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    
    # 4. Save to disk
    
    np.save(OUTPUT_NPY, embeddings_array)

if __name__ == "__main__":
    generate_embeddings()
