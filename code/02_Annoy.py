import numpy as np
from annoy import AnnoyIndex
import os
import time
import random

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_NPY = os.path.join(BASE_DIR, "embeddings", "text-embedding-3-small.npy")
ANNOY_DIR = os.path.join(BASE_DIR, "Annoy")
OUTPUT_INDEX = os.path.join(ANNOY_DIR, "small_500.ann")

N_TREES = 500   # Discussed in the Appendix

SEED = 2025
np.random.seed(SEED)
random.seed(SEED)

def build_Annoy_tree():
    """
    Step 2: Build an Annoy tree for approximate nearest neighbor search.
    """
    if not os.path.exists(ANNOY_DIR):
        os.makedirs(ANNOY_DIR)

    embeddings = np.load(INPUT_NPY)
    embedding_dim = embeddings.shape[1]
    ann_index = AnnoyIndex(embedding_dim, 'angular')
    
    for i in range(len(embeddings)):
        ann_index.add_item(i, embeddings[i])
    
    ann_index.build(N_TREES)
    ann_index.save(OUTPUT_INDEX)

if __name__ == "__main__":
    build_Annoy_tree()