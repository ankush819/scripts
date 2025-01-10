from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import create_extraction_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from datasets import load_dataset
from typing import List, Dict, Any
from .config import (
    GOOGLE_API_KEY,
    ZILLIZ_URI,
    ZILLIZ_TOKEN,
    COLLECTION_NAME,
    NUM_SAMPLES,
    TITLE_MODEL,
    CONTENT_MODEL,
    SUMMARY_MODEL
)

class LangChainManager:
    def __init__(self):
        # Initialize LLM
        self.llm = GoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )
        
        # Initialize embeddings
        self.title_embeddings = HuggingFaceEmbeddings(model_name=TITLE_MODEL)
        self.content_embeddings = HuggingFaceEmbeddings(model_name=CONTENT_MODEL)
        self.summary_embeddings = HuggingFaceEmbeddings(model_name=SUMMARY_MODEL)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Initialize extraction chain for metadata
        self.extraction_prompt = PromptTemplate.from_template("""
        Extract the following information from the text:
        - A concise summary (max 150 chars)
        - 5 key phrases (comma-separated)
        - Document metadata including tone, complexity, target audience, and main topic
        
        Text: {text}
        
        Output in JSON format:
        {{"summary": "...", "keywords": "...", "metadata": {{"tone": "...", "complexity": "...", "target_audience": "...", "main_topic": "..."}}}}
        """)
        
        self.extraction_chain = create_extraction_chain(self.extraction_prompt, self.llm)
    
    def load_and_process_dataset(self):
        """Load AG News dataset and process with LangChain"""
        # Load dataset
        dataset = load_dataset("ag_news", split=f"train[:{NUM_SAMPLES}]")
        
        # Process documents in batches
        documents = []
        for item in dataset:
            text = item["text"]
            title = text.split("\n", 1)[0][:500]
            content = text.split("\n", 1)[1] if "\n" in text else text
            
            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    "id": len(documents),
                    "title": title,
                    "category": item["label"],
                    "source": "ag_news"
                }
            )
            documents.append(doc)
        
        # Process with LLM in batches
        processed_docs = []
        for i in range(0, len(documents), 10):
            batch = documents[i:i+10]
            results = self.extraction_chain.batch(batch)
            
            for doc, result in zip(batch, results):
                doc.metadata.update(result)
                processed_docs.append(doc)
        
        return processed_docs
    
    def setup_vectorstore(self, documents: List[Document]):
        """Initialize Milvus/Zilliz vector stores for each embedding type"""
        # Create vector stores
        self.title_store = Milvus.from_documents(
            documents,
            self.title_embeddings,
            collection_name=f"{COLLECTION_NAME}_title",
            connection_args={"uri": ZILLIZ_URI, "token": ZILLIZ_TOKEN}
        )
        
        self.content_store = Milvus.from_documents(
            documents,
            self.content_embeddings,
            collection_name=f"{COLLECTION_NAME}_content",
            connection_args={"uri": ZILLIZ_URI, "token": ZILLIZ_TOKEN}
        )
        
        self.summary_store = Milvus.from_documents(
            documents,
            self.summary_embeddings,
            collection_name=f"{COLLECTION_NAME}_summary",
            connection_args={"uri": ZILLIZ_URI, "token": ZILLIZ_TOKEN}
        )
    
    def semantic_search(self, query: str, k: int = 5):
        """Perform semantic search across all vector stores"""
        # Search each store
        title_results = self.title_store.similarity_search_with_score(query, k=k)
        content_results = self.content_store.similarity_search_with_score(query, k=k)
        summary_results = self.summary_store.similarity_search_with_score(query, k=k)
        
        # Combine and sort results
        all_results = []
        for doc, score in title_results + content_results + summary_results:
            all_results.append({
                "id": doc.metadata["id"],
                "title": doc.metadata["title"],
                "content": doc.page_content,
                "summary": doc.metadata.get("summary", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "category": doc.metadata["category"],
                "metadata": doc.metadata.get("metadata", {}),
                "distance": score
            })
        
        # Sort and deduplicate
        all_results.sort(key=lambda x: x["distance"])
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result["id"] not in seen_ids and len(unique_results) < k:
                seen_ids.add(result["id"])
                unique_results.append(result)
        
        return unique_results 