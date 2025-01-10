import os
from dotenv import load_dotenv

load_dotenv()

# Zilliz Configuration
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = "news_articles_enhanced"

# Google API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Vector DB Configuration
TITLE_VECTOR_DIM = 384  # all-MiniLM-L6-v2
CONTENT_VECTOR_DIM = 768  # mpnet-base
SUMMARY_VECTOR_DIM = 1024  # e5-large

# Models Configuration
TITLE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CONTENT_MODEL = 'sentence-transformers/all-mpnet-base-v2'
SUMMARY_MODEL = 'intfloat/e5-large'

BATCH_SIZE = 64

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
NUM_SAMPLES = 100  # Total samples to process
INSERT_BATCH_SIZE = 100  # Number of records to insert at once
LLM_BATCH_SIZE = 100  # Increased from 10 to 100 documents per LLM call 