import os
from dotenv import load_dotenv

load_dotenv()

# Couchbase Capella Configuration
CAPELLA_HOST = os.getenv("CAPELLA_HOST")
CAPELLA_USERNAME = os.getenv("CAPELLA_USERNAME")
CAPELLA_PASSWORD = os.getenv("CAPELLA_PASSWORD")
CAPELLA_BUCKET = "vector_search_demo"
CAPELLA_SCOPE = "news"
CAPELLA_COLLECTION = "articles"

# Vector Search Configuration
VECTOR_DIM = 384  # Using all-MiniLM-L6-v2 for simplicity
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Dataset Configuration
NUM_SAMPLES = 1000  # Number of articles to load
BATCH_SIZE = 100  # Batch size for insertions

# Search Configuration
VECTOR_FIELD = "embedding"
VECTOR_INDEX = "news_vector_index" 