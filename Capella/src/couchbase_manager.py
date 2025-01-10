from couchbase.cluster import Cluster, ClusterOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.options import QueryOptions
from couchbase.management.queries import CreateQueryIndexOptions
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from typing import List, Dict, Any
import time
import json
from .config import *

class CouchbaseManager:
    def __init__(self):
        # Initialize connection
        auth = PasswordAuthenticator(CAPELLA_USERNAME, CAPELLA_PASSWORD)
        self.cluster = Cluster(
            f"couchbases://{CAPELLA_HOST}",
            ClusterOptions(auth)
        )
        
        # Initialize bucket, scope and collection
        self.bucket = self.cluster.bucket(CAPELLA_BUCKET)
        self.scope = self.bucket.scope(CAPELLA_SCOPE)
        self.collection = self.scope.collection(CAPELLA_COLLECTION)
        
        # Initialize embedding model
        self.model = SentenceTransformer(MODEL_NAME)
    
    def setup_database(self):
        """Setup database with required indexes"""
        # Create primary index if not exists
        query = f"""
        CREATE PRIMARY INDEX IF NOT EXISTS `{CAPELLA_BUCKET}_primary` 
        ON `{CAPELLA_BUCKET}`.`{CAPELLA_SCOPE}`.`{CAPELLA_COLLECTION}`
        """
        self.cluster.query(query).execute()
        
        # Vector search index is assumed to be already created in Capella UI
        print("Database setup completed")
    
    def load_dataset(self):
        """Load and process AG News dataset"""
        print(f"Loading {NUM_SAMPLES} samples from AG News dataset...")
        dataset = load_dataset("ag_news", split=f"train[:{NUM_SAMPLES}]")
        
        # Process in batches
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset[i:i + BATCH_SIZE]
            batch_docs = []
            
            for j, item in enumerate(batch):
                # Split into title and content
                text = item["text"]
                title = text.split("\n", 1)[0]
                content = text.split("\n", 1)[1] if "\n" in text else text
                
                # Generate embedding from title + content
                embedding = self.model.encode(f"{title} {content}").tolist()
                
                # Create document
                doc = {
                    "id": f"article_{i+j}",
                    "type": "article",
                    "title": title,
                    "content": content,
                    "category": item["label"],
                    "embedding": embedding
                }
                batch_docs.append(doc)
            
            # Insert batch
            for doc in batch_docs:
                self.collection.upsert(doc["id"], doc)
            
            print(f"Inserted batch {i} to {i + len(batch_docs)}")
    
    def vector_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Perform vector search using Couchbase"""
        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()
        
        # Construct vector search query
        search_query = f"""
        SELECT a.id, a.title, a.content, a.category,
               ARRAY_VECTOR_DISTANCE(a.embedding, $query_vector) as distance
        FROM `{CAPELLA_BUCKET}`.`{CAPELLA_SCOPE}`.`{CAPELLA_COLLECTION}` a
        WHERE type = 'article'
        ORDER BY ARRAY_VECTOR_DISTANCE(a.embedding, $query_vector)
        LIMIT $limit
        """
        
        # Execute search
        start_time = time.time()
        results = self.cluster.query(
            search_query,
            QueryOptions(
                named_parameters={
                    'query_vector': query_embedding,
                    'limit': limit
                }
            )
        )
        
        # Format results
        formatted_results = []
        for row in results:
            formatted_results.append({
                'id': row['id'],
                'title': row['title'],
                'content': row['content'],
                'category': row['category'],
                'distance': row['distance']
            })
        
        search_time = time.time() - start_time
        
        return formatted_results, search_time 