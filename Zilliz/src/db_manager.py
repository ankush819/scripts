from pymilvus import connections, utility
from .config import ZILLIZ_URI, ZILLIZ_TOKEN

class DBManager:
    @staticmethod
    def connect():
        connections.connect(
            alias="default",
            uri=ZILLIZ_URI,
            token=ZILLIZ_TOKEN
        )
        print("Connected to Zilliz Cloud")
    
    @staticmethod
    def disconnect():
        connections.disconnect("default") 