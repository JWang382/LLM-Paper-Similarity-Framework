# Semantic Retrieval and Embedding Analysis for Academic Papers

This repository contains the official implementation for our research on academic document embeddings. We evaluate the semantic retrieval performance of various embedding models.

## Project Structure

```text
Seanns
 ┣ code
 ┃ ┣ 01_gen_embeddings.py     # Stage 1: Text to Vectors (OpenAI API/Local Models)
 ┃ ┣ 02_build_index.py        # Stage 2: ANN Construction via Annoy
 ┃ ┗ 03_eval_retrieval.py     # Stage 3: Retrieval Evaluation
 ┣ data
 ┃ ┗ top_cited.csv            # [Input] Sub-corpus (Title, Abstract...)
 ┣ embeddings                 # [Generated] Store .npy files
 ┣ Annoy                      # [Generated] Store .ann files
 ┣ requirements.txt           # Environment dependencies
 ┗ README.md
```

## Getting Started

### 1. Environment Setup

Install the required dependencies using the specific versions used in our experiments:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Place your metadata file in `data/top_cited.csv`. The CSV should at least contain the following columns:

* `title`: The title used for embedding generation.
* `abstract`: Paper abstract.
* `publisher`: Journal or conference name. 
* `citations`: Citation counts for subset selection.
* `year`: Publication year.

### 3. Execution Pipeline

#### Step 1: Generate Embeddings

Run the embedding generation script. By default, it uses `text-embedding-3-small`.
*Note: Ensure your OpenAI API key is set in the script or environment.*

```bash
python code/01_gen_embeddings.py
```

* **Cost Estimate:** For `text-embedding-3-small`, the cost is approximately **$0.02 per 1M tokens**.
* **Other Models:** The script includes commented instructions for implementing **ModernBERT, SPECTER, Nomic-Embed, and SciNCL**.

#### Step 2: Build ANN Index

Construct the Annoy index (Forest of trees) for fast approximate nearest neighbor retrieval:

```bash
python code/02_build_index.py
```

#### Step 3: Evaluation

Perform the retrieval experiment and calculate the **Semantic Margin**:

```bash
python code/03_eval_retrieval.py
```




