import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm
import os
import random

SEED = 2025
np.random.seed(SEED)
random.seed(SEED)

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "data", "top_cited.csv")
INPUT_NPY = os.path.join(BASE_DIR, "embeddings", "text-embedding-3-small.npy")
INPUT_INDEX = os.path.join(BASE_DIR, "Annoy", "small_500.ann")

K_NEIGHBORS = 20   # For example

np.random.seed(2025)
def evaluate():
    """
    Step 3: Evaluate retrieval similarity and calculate local anisotropy.
    """

    df = pd.read_csv(INPUT_CSV)
    embeddings = np.load(INPUT_NPY)
    embedding_dim = embeddings.shape[1]
    ann = AnnoyIndex(embedding_dim, 'angular')
    ann.load(INPUT_INDEX) 
    
    results = []
    for q_idx in tqdm(df.index, desc="Retrieving"):
        query_vec = embeddings[q_idx]
        
        # 1. Annoy Search
        nn_indices = ann.get_nns_by_item(q_idx, K_NEIGHBORS + 1)
        neighbors = [i for i in nn_indices if i != q_idx][:K_NEIGHBORS]
        
        # 2. Similarity Metric
        sim_ann = np.mean(np.dot(embeddings[neighbors], query_vec))
        
        # 3. Random Baseline
        rand_idx = np.random.choice([i for i in range(len(df)) if i != q_idx], K_NEIGHBORS, replace=False)
        sim_rand = np.mean(np.dot(embeddings[rand_idx], query_vec))
      
        results.append({
            'sim_ann': sim_ann,
            'sim_rand': sim_rand,
            'margin': sim_ann - sim_rand,  
        })

    res_df = pd.DataFrame(results)
    return res_df

if __name__ == "__main__":
    res_df = evaluate()
    print(res_df.values)