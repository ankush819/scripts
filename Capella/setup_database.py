from src.couchbase_manager import CouchbaseManager
import time

def main():
    try:
        print("Starting database setup...")
        start_time = time.time()
        
        # Initialize manager
        manager = CouchbaseManager()
        
        # Setup database
        manager.setup_database()
        
        # Load dataset
        manager.load_dataset()
        
        print(f"Setup completed in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during setup: {e}")

if __name__ == "__main__":
    main() 