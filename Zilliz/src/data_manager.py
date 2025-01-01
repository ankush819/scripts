from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
from typing import List, Dict, Any
from .config import MODEL_NAME, BATCH_SIZE, DATASET_NAME, NUM_SAMPLES

class DataManager:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max_length characters"""
        return text[:max_length] if text else ""
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset with ground truth for evaluation"""
        print(f"Loading {NUM_SAMPLES} samples from {DATASET_NAME}...")
        dataset = load_dataset(DATASET_NAME, split=f"train[:{NUM_SAMPLES}]")
        
        # Map numeric labels to categories
        label_map = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Science/Technology"
        }
        
        processed_data = []
        start_time = time.time()
        
        # Create ground truth pairs for evaluation
        for i, item in enumerate(tqdm(dataset)):
            title = self._truncate_text(item["text"].split("\n")[0], 500)  # Truncate title to 500 chars
            content = self._truncate_text(item["text"], 2000)  # Truncate content to 2000 chars
            
            processed_data.append({
                "id": i,
                "title": title,
                "content": content,
                "category": label_map[item["label"]],
                "category_id": item["label"],
                "similar_articles": self._get_similar_articles(dataset, i, item["label"])
            })
            
        print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
        return processed_data
    
    def _get_similar_articles(self, dataset, current_id: int, category: int) -> List[int]:
        """Create ground truth by finding articles in same category"""
        similar_ids = []
        for i, item in enumerate(dataset):
            if i != current_id and item["label"] == category:
                similar_ids.append(i)
                if len(similar_ids) >= 10:  # Keep top 10 similar articles
                    break
        return similar_ids
    
    def generate_embeddings(self, data: List[Dict], batch_size: int = BATCH_SIZE) -> List[List[float]]:
        """Generate embeddings for the dataset"""
        texts = [f"{item['title']} {item['content']}" for item in data]
        
        embeddings = []
        start_time = time.time()
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings.tolist())
            
        print(f"Vectors generated in {time.time() - start_time:.2f} seconds")
        return embeddings
    
    def generate_query_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single query"""
        return self.model.encode(text).tolist() 