from src.langchain_manager import LangChainManager
import time

def main():
    try:
        print("Starting database setup with LangChain...")
        start_time = time.time()
        
        # Initialize LangChain manager
        manager = LangChainManager()
        
        # Load and process dataset
        print("Loading and processing dataset...")
        documents = manager.load_and_process_dataset()
        
        # Setup vector stores
        print("Setting up vector stores...")
        manager.setup_vectorstore(documents)
        
        print(f"Setup completed in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during setup: {e}")

if __name__ == "__main__":
    main() 