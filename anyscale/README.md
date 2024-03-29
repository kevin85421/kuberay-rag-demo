# Building a RAG-based LLM application with Anyscale

## Step 1: Launch an Anyscale workspace

* Start a new [Anyscale workspace](https://docs.anyscale.com/get-started) using an `n1-standard-32-nvidia-t4-16gb-2` head node on GCP, which has 2 GPUs and 32 CPUs.
* Disable autoscaling by deleting the worker node types.
* Use the `default_cluster_env_2.10.0_py39` cluster environment.

## Step 2: Clone this repository

```bash
git clone https://github.com/kevin85421/kuberay-rag-demo.git
```

## Step 3: Start a Postgres database

```bash
# path: kuberay-rag-demo/anyscale/
bash setup-pgvector.sh

# path: kuberay-rag-demo/anyscale/
# Initialize the schema.
sudo -u postgres psql -f vector-768.sql
```

## Step 4 (option 1): Index Ray document embeddings into Postgres

You can download the pre-generated embeddings from Google Cloud Storage and index them into Postgres.

```bash
wget https://storage.googleapis.com/ray-docs-embedding-postgres-dump/gte-base_300_50.sql
sudo -u postgres psql -f gte-base_300_50.sql
```

## Step 4 (option 2): Generate embeddings with Ray Data and index them into Postgres

Follow the notebook [rag-demo-embedding-generation.ipynb](rag-demo-embedding-generation.ipynb) to generate embeddings with Ray Data and index them into Postgres.

* Further reading: [RAG at Scale: 10x Cheaper Embedding Computations with Anyscale and Pinecone](https://www.anyscale.com/blog/rag-at-scale-10x-cheaper-embedding-computations-with-anyscale-and-pinecone)

## Step 5: Verify the Postgres database

Check whether the embeddings are successfully indexed into the Postgres database.

```bash
sudo -u postgres psql
\dt
#           List of relations
#  Schema |   Name   | Type  |  Owner   
# --------+----------+-------+----------
#  public | document | table | postgres
SELECT * FROM document LIMIT 5;
```

## Step 6: Install the Python dependencies

```bash
# path: kuberay-rag-demo/anyscale/
pip install --user -r requirements.txt
```

## Step 7: Create a `.env` file to connect to LLM endpoints

* Set `ANYSCALE_API_BASE` and `ANYSCALE_API_KEY` in a `.env` file.
  ```sh
  # Example .env file
  ANYSCALE_API_BASE="https://api.endpoints.anyscale.com/v1"
  ANYSCALE_API_KEY="YOUR_ANYSCALE_ENDPOINT_API_KEY"  # https://app.endpoints.anyscale.com/credentials
  ```

* Note that the `.env` must be in the same directory as the [rag-demo-generation.ipynb](rag-demo-generation.ipynb).
* You can also use OpenAI's API or the open-source [ray-llm](https://github.com/ray-project/ray-llm) as the LLM endpoint, but you may need to modify the code in the Jupyter notebook a bit.


## Step 8: Follow the Jupyter notebook to generate responses.

* Follow [rag-demo-generation.ipynb](rag-demo-generation.ipynb) to generate responses.
* Example response:
    ```json
    {
    "question": "What is the default batch size for map_batches?",
    "sources": [
        "https://docs.ray.io/en/master/data/api/doc/ray.data.Dataset.map_batches.html#ray-data-dataset-map-batches",
        "https://docs.ray.io/en/master/data/batch_inference.html#configuring-batch-size",
        "https://docs.ray.io/en/master/data/batch_inference.html#configuring-batch-size",
        "https://docs.ray.io/en/master/tune/getting-started.html#setting-up-a-tuner-for-a-training-run-with-tune",
        "https://docs.ray.io/en/master/data/api/doc/ray.data.Dataset.map_batches.html#ray-data-dataset-map-batches"
    ],
    "answer": " The default batch size for map\\_batches is 1024.",
    "llm": "mistralai/Mixtral-8x7B-Instruct-v0.1"
    }
    ```