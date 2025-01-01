from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from .db_manager import DBManager
from .collection_manager import CollectionManager
from .data_manager import DataManager
from .search_manager import SearchManager

app = FastAPI(title="Vector Search Demo")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    search_type: str
    limit: Optional[int] = 5

@app.on_event("startup")
async def startup_event():
    DBManager.connect()
    app.state.collection = CollectionManager.create_collection()
    app.state.data_manager = DataManager()
    app.state.search_manager = SearchManager(app.state.collection, app.state.data_manager)
    app.state.collection.load()

@app.on_event("shutdown")
async def shutdown_event():
    DBManager.disconnect()

@app.get("/search-types")
async def get_search_types():
    """Get available search types"""
    return {
        "search_types": [
            {"id": "semantic", "name": "Semantic Search", "description": "Pure vector similarity search"},
            {"id": "hybrid", "name": "Hybrid Search", "description": "Combined vector and keyword search"}
        ]
    }

@app.post("/search")
async def search(request: SearchRequest) -> Dict:
    """Perform search with specified strategy"""
    search_types = {
        "semantic": app.state.search_manager.semantic_search,
        "hybrid": app.state.search_manager.hybrid_search
    }
    
    if request.search_type not in search_types:
        raise HTTPException(status_code=400, detail="Invalid search type")
    
    search_func = search_types[request.search_type]
    results, metrics = search_func(request.query, request.limit)
    
    return {
        "results": results,
        "metrics": metrics
    } 