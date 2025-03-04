from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import time
from .langchain_manager import LangChainManager

app = FastAPI(title="Vector Search API")

# Initialize LangChain manager
manager = LangChainManager()

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    metrics: Dict[str, Any]

@app.post("/search/semantic", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    try:
        start_time = time.time()
        
        # Perform search
        search_start = time.time()
        results = manager.semantic_search(request.query, k=request.limit)
        search_time = time.time() - search_start
        
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