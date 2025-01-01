from src.db_manager import DBManager
from src.collection_manager import CollectionManager
from src.data_manager import DataManager
from src.config import INSERT_BATCH_SIZE
import time

def insert_batch(collection, data, embeddings, start_idx, batch_size):
    end_idx = min(start_idx + batch_size, len(data))
    batch_data = data[start_idx:end_idx]
    batch_embeddings = embeddings[start_idx:end_idx]
    
    # Prepare batch data
    ids = [item["id"] for item in batch_data]
    titles = [item["title"] for item in batch_data]
    contents = [item["content"] for item in batch_data]
    categories = [item["category"] for item in batch_data]
    category_ids = [item["category_id"] for item in batch_data]
    similar_articles = [item["similar_articles"] for item in batch_data]
    
    # Insert batch
    collection.insert([
        ids,
        titles,
        contents,
        categories,
        category_ids,
        similar_articles,
        batch_embeddings
    ])
    print(f"Inserted batch {start_idx} to {end_idx}")

def main():
    try:
        print("Starting database setup...")
        DBManager.connect()
        
        # Drop existing collection if exists
        CollectionManager.drop_collection()
        
        # Initialize collection
        collection = CollectionManager.create_collection()
        
        print("Loading dataset and generating embeddings...")
        data_manager = DataManager()
        data = data_manager.load_dataset()
        embeddings = data_manager.generate_embeddings(data)
        
        # Insert data in batches
        print("Inserting data in batches...")
        start_time = time.time()
        
        for start_idx in range(0, len(data), INSERT_BATCH_SIZE):
            insert_batch(collection, data, embeddings, start_idx, INSERT_BATCH_SIZE)
            
        print(f"Total insertion time: {time.time() - start_time:.2f} seconds")
        print(f"Successfully inserted {len(data)} items")
            
    finally:
        DBManager.disconnect()

if __name__ == "__main__":
    main() 