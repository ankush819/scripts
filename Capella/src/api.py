from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import time
from .couchbase_manager import CouchbaseManager

app = FastAPI(title="Couchbase Vector Search Demo")

# Initialize Couchbase manager
manager = CouchbaseManager()

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    metrics: Dict[str, Any]

@app.post("/search/vector", response_model=SearchResponse)
async def vector_search(request: SearchRequest):
    try:
        start_time = time.time()
        
        # Perform search
        results, search_time = manager.vector_search(request.query, request.limit)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            "total_time_ms": total_time * 1000,
            "search_time_ms": search_time * 1000,
            "network_time_ms": (total_time - search_time) * 1000,
            "num_results": len(results)
        }
        
        return SearchResponse(results=results, metrics=metrics)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 