from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
from .config import MODEL_NAME, BATCH_SIZE

class DataLoader:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        
    def load_amazon_reviews(self, num_samples=100000):
        print("Loading Amazon reviews dataset...")
        dataset = load_dataset("amazon_reviews_multi", "en", split="train[:100000]")
        
        processed_data = []
        start_time = time.time()
        
        for i, item in enumerate(tqdm(dataset)):
            processed_data.append({
                "id": i,
                "product_title": item["product_title"],
                "review_title": item["review_title"],
                "review_body": item["review_body"],
                "stars": item["stars"]
            })
            
        print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
        return processed_data
    
    def prepare_vectors(self, data, batch_size=BATCH_SIZE):
        texts = [f"{item['product_title']} {item['review_title']} {item['review_body']}" 
                for item in data]
        
        embeddings = []
        start_time = time.time()
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings.tolist())
            
        print(f"Vectors generated in {time.time() - start_time:.2f} seconds")
        return embeddings 