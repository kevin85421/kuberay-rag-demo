# Building a RAG-based LLM application with KubeRay

## Step 1: Launch a GKE cluster with a GPU node

```sh
# Create a GKE cluster with a GPU node.
gcloud container clusters create kuberay-gpu-cluster \
    --num-nodes=1 --min-nodes 0 --max-nodes 1 --enable-autoscaling \
    --zone=us-west1-b --machine-type n1-standard-32 \
    --accelerator type=nvidia-tesla-t4,count=2

# Install NVIDIA GPU device driver
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded-latest.yaml

# Verify that your nodes have allocatable GPUs 
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# NAME     GPU
# ......   2
```

## Step 2: Start a Postgres database and a RayCluster

```bash
# path: kuberay-rag-demo/kuberay/
kubectl apply -f kuberay-rag-demo.yaml
```

The YAML file creates a Postgres database and a 1-node RayCluster. 

## Step 3: Clone this repository in the Ray head Pod

```bash
# Log into the Ray head Pod
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
kubectl exec -it $HEAD_POD -- bash

# Install the Python dependencies on the Ray head Pod
git clone https://github.com/kevin85421/kuberay-rag-demo.git
cd kuberay-rag-demo
pip install --user -r requirements.txt

# Install the Postgres client on the Ray head Pod
sudo apt-get install -y postgresql-client
```

## Step 4 (option 1): Index Ray document embeddings into Postgres

You can download the pre-generated embeddings from Google Cloud Storage and index them into Postgres.

```bash
# Execute the following commands in the Ray head Pod
wget https://storage.googleapis.com/ray-docs-embedding-postgres-dump/gte-base_300_50.sql
sudo -u postgres psql -f gte-base_300_50.sql -h pgvector
# password: postgres
```

## Step 4 (option 2): Generate embeddings with Ray Data and index them into Postgres

```bash
python3 rag-demo-embedding-generation.py
```

* Further reading: [RAG at Scale: 10x Cheaper Embedding Computations with Anyscale and Pinecone](https://www.anyscale.com/blog/rag-at-scale-10x-cheaper-embedding-computations-with-anyscale-and-pinecone)

## Step 5: Verify the Postgres database

Check whether the embeddings are successfully indexed into the Postgres database.

```bash
# Log into the Postgres Pod
kubectl exec -it deployment/pgvector -- bash

# Verify the Postgres database
sudo -u postgres psql
\dt
#           List of relations
#  Schema |   Name   | Type  |  Owner   
# --------+----------+-------+----------
#  public | document | table | postgres
SELECT * FROM document LIMIT 5;
```

## Step 7: Create a `.env` file to connect to LLM endpoints

* Set `ANYSCALE_API_BASE` and `ANYSCALE_API_KEY` in a `.env` file.
  ```sh
  # Example .env file
  ANYSCALE_API_BASE="https://api.endpoints.anyscale.com/v1"
  ANYSCALE_API_KEY="YOUR_ANYSCALE_ENDPOINT_API_KEY"  # https://app.endpoints.anyscale.com/credentials
  ```

* Note that the `.env` must be in the same directory as the [rag-demo-generation.py](rag-demo-generation.py).
* You can also use OpenAI's API or the open-source [ray-llm](https://github.com/ray-project/ray-llm) as the LLM endpoint, but you may need to modify the Python code a bit.


## Step 8: Generate responses

```sh
python3 rag-demo-generation.py
```

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