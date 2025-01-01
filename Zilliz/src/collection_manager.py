from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility
from .config import COLLECTION_NAME, VECTOR_DIM, INDEX_PARAMS

class CollectionManager:
    @staticmethod
    def create_collection():
        if utility.has_collection(COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' already exists")
            return Collection(COLLECTION_NAME)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="category_id", dtype=DataType.INT64),
            FieldSchema(name="similar_articles", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=10),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        ]
        schema = CollectionSchema(fields=fields, description="News articles collection")
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
        print(f"Created collection '{COLLECTION_NAME}' with index")
        return collection
    
    @staticmethod
    def drop_collection():
        if utility.has_collection(COLLECTION_NAME):
            utility.drop_collection(COLLECTION_NAME)
            print(f"Dropped collection '{COLLECTION_NAME}'") 