import os
from dotenv import load_dotenv

load_dotenv()

# Zilliz Configuration
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = "news_articles"

# Vector DB Configuration
VECTOR_DIM = 384
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 128

# Index Configuration
INDEX_PARAMS = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}

SEARCH_PARAMS = {
    "metric_type": "L2",
    "params": {"nprobe": 16}
}

# Dataset Configuration
DATASET_NAME = "ag_news"
NUM_SAMPLES = 10000  # Reduced from 100000 to avoid memory issues
INSERT_BATCH_SIZE = 1000  # Number of records to insert at once 