import time
from typing import Dict, List, Any, Tuple
from .config import SEARCH_PARAMS

class SearchMetrics:
    def __init__(self, ground_truth: List[int], results: List[Dict]):
        self.ground_truth = ground_truth
        self.results = results
        
    def calculate_recall(self) -> float:
        """Calculate recall@k based on ground truth"""
        result_ids = [hit["id"] for hit in self.results]
        relevant_found = len(set(result_ids) & set(self.ground_truth))
        return relevant_found / len(self.ground_truth) if self.ground_truth else 0
    
    def calculate_latency_breakdown(self, timings: Dict[str, float]) -> Dict[str, float]:
        """Calculate detailed latency metrics"""
        return {
            "total_time_ms": timings["total_time"] * 1000,
            "embedding_time_ms": timings["embedding_time"] * 1000,
            "search_time_ms": timings["search_time"] * 1000,
            "network_time_ms": timings["network_time"] * 1000
        }

class SearchManager:
    def __init__(self, collection, data_manager):
        self.collection = collection
        self.data_manager = data_manager
    
    def _measure_performance(func):
        def wrapper(self, *args, **kwargs) -> Tuple[List[Dict], Dict]:
            start_total = time.time()
            
            # Generate embedding
            start_embed = time.time()
            query_text = kwargs.get('query_text', args[0])
            embedding = self.data_manager.generate_query_embedding(query_text)
            embed_time = time.time() - start_embed
            
            # Perform search
            start_search = time.time()
            kwargs['embedding'] = embedding
            results, ground_truth = func(self, *args, **kwargs)
            search_time = time.time() - start_search
            
            total_time = time.time() - start_total
            
            # Calculate metrics
            metrics = SearchMetrics(ground_truth, results)
            performance = metrics.calculate_latency_breakdown({
                "total_time": total_time,
                "embedding_time": embed_time,
                "search_time": search_time,
                "network_time": total_time - (embed_time + search_time)
            })
            
            performance["recall"] = metrics.calculate_recall()
            
            return results, performance
        return wrapper
    
    @_measure_performance
    def semantic_search(self, query_text: str, limit: int = 5, embedding=None) -> Tuple[List[Dict], List[int]]:
        """Pure vector similarity search"""
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param=SEARCH_PARAMS,
            limit=limit,
            output_fields=["id", "title", "content", "category", "similar_articles"]
        )
        
        formatted_results = []
        ground_truth = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,  
                    "title": hit.title,  
                    "content": hit.content,
                    "category": hit.category,
                    "distance": hit.distance
                }
                formatted_results.append(result)
                if hasattr(hit, 'similar_articles'):
                    ground_truth.extend(hit.similar_articles)
        
        return formatted_results, ground_truth
    
    @_measure_performance
    def hybrid_search(self, query_text: str, limit: int = 5, embedding=None) -> Tuple[List[Dict], List[int]]:
        """Combined vector and keyword search"""
        expr = f'title like "%{query_text}%" or content like "%{query_text}%"'
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param=SEARCH_PARAMS,
            limit=limit,
            expr=expr,
            output_fields=["id", "title", "content", "category", "similar_articles"]
        )
        
        formatted_results = []
        ground_truth = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,  
                    "title": hit.title,  
                    "content": hit.content,
                    "category": hit.category,
                    "distance": hit.distance
                }
                formatted_results.append(result)
                if hasattr(hit, 'similar_articles'):
                    ground_truth.extend(hit.similar_articles)
        
        return formatted_results, ground_truth 