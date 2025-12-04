"""
Milvus Vector Database Client for Obligation Tracking System.
Handles storage and retrieval of contract embeddings and document chunks.
"""

import os
import yaml
import logging
from typing import Optional, Dict, Any, List
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
    db
)

logger = logging.getLogger(__name__)


class MilvusClient:
    """Milvus client for storing contract and document chunk embeddings."""
    
    def __init__(self, config_path: str = "config/milvus_config.yaml"):
        """
        Initialize Milvus client with configuration.
        
        Args:
            config_path: Path to Milvus configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        milvus_config = self.config['milvus']
        
        # Helper to handle YAML placeholders
        def get_config_value(key, default):
            val = milvus_config.get(key, default)
            if isinstance(val, str) and val.startswith('${'):
                return default
            return val
        
        # Connection parameters from environment or config
        self.host = os.getenv("MILVUS_HOST", get_config_value('host', 'localhost'))
        self.port = int(os.getenv("MILVUS_PORT", get_config_value('port', 19530)))
        self.database = os.getenv("MILVUS_DATABASE", get_config_value('database', 'default'))
        self.user = os.getenv("MILVUS_USER", get_config_value('user', ''))
        self.password = os.getenv("MILVUS_PASSWORD", get_config_value('password', ''))
        
        logger.info(f"Milvus Client initialized with host={self.host}, port={self.port}, db={self.database}")
        
        self.collections_config = self.config['collections']
        self._connected = False
        
    def _create_database_if_not_exists(self) -> None:
        """Create the database if it doesn't exist."""
        try:
            # Connect without specifying database first
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )

            # Check if database exists
            existing_dbs = db.list_database()
            if self.database not in existing_dbs:
                logger.info(f"Creating database: {self.database}")
                db.create_database(self.database)
                logger.info(f"Database {self.database} created successfully")
            else:
                logger.info(f"Database {self.database} already exists")

            # Disconnect to reconnect with the database
            connections.disconnect("default")
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            # Ensure disconnection on error
            try:
                connections.disconnect("default")
            except Exception:
                pass
            raise

    def connect(self) -> None:
        """Establish connection to Milvus database."""
        try:
            # First ensure database exists
            self._create_database_if_not_exists()

            # Now connect with the database
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db_name=self.database
            )
            self._connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}, database: {self.database}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close Milvus connection."""
        if self._connected:
            connections.disconnect("default")
            self._connected = False
            logger.info("Disconnected from Milvus")
    
    def _ensure_connected(self) -> None:
        """Ensure connection is established."""
        if not self._connected:
            self.connect()
    
    def insert_contract(self, contract_data: Dict[str, Any]) -> str:
        """Insert a contract into the collection."""
        self._ensure_connected()
        collection = Collection(self.collections_config['contracts']['name'])

        content_summary = contract_data.get('content_summary') or ''
        if len(content_summary) > 1900:
            content_summary = content_summary[:1900]

        data = [[contract_data.get('id') or ''],
                [contract_data.get('user_id') or ''],
                [contract_data.get('filename') or ''],
                [(contract_data.get('title') or '')[:450]],
                [(contract_data.get('parties') or '')[:900]],
                [contract_data.get('effective_date') or ''],
                [contract_data.get('expiration_date') or ''],
                [contract_data.get('contract_value') or ''],
                [content_summary],
                [contract_data.get('embedding') or [0.0] * 768]]

        collection.insert(data)
        collection.flush()

        logger.info(f"Inserted contract: {contract_data.get('id')}")
        return contract_data.get('id') or ''

    def create_contracts_collection(self) -> Collection:
        """Create the contracts collection if it doesn't exist."""
        self._ensure_connected()

        collection_name = self.collections_config['contracts']['name']
        dimension = self.collections_config['contracts']['dimension']

        if utility.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return Collection(collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="parties", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="effective_date", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="expiration_date", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="contract_value", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="content_summary", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]

        schema = CollectionSchema(fields=fields, description="Contract documents")
        collection = Collection(name=collection_name, schema=schema)

        index_params = {
            "metric_type": self.collections_config['contracts']['metric_type'],
            "index_type": self.collections_config['contracts']['index_type'],
            "params": {"nlist": self.collections_config['contracts']['nlist']}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

        logger.info(f"Created collection: {collection_name}")
        return collection

    def create_extract_data_collection(self) -> Collection:
        """Create the extract_data collection for storing extracted document chunks."""
        self._ensure_connected()

        collection_name = "extract_data"
        dimension = 768

        if utility.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return Collection(collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="contract_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]

        schema = CollectionSchema(fields=fields, description="Extracted document chunks for semantic search")
        collection = Collection(name=collection_name, schema=schema)

        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

        logger.info(f"Created collection: {collection_name}")
        return collection

    def insert_extract_data(self, data: Dict[str, Any]) -> str:
        """Insert extracted chunk into extract_data collection."""
        self._ensure_connected()
        self.create_extract_data_collection()
        collection = Collection("extract_data")

        content = (data.get('content', '') or '')[:7000]

        id_val = data.get('id', '')[:60]
        user_id = data.get('user_id', '')[:60]
        contract_id = data.get('contract_id', '')[:60]

        insert_data = [
            [id_val],
            [user_id],
            [contract_id],
            [data.get('chunk_index', 0)],
            [content],
            [data.get('embedding', [0.0] * 768)]
        ]

        collection.insert(insert_data)
        collection.flush()

        logger.info(f"Inserted extract_data chunk: {data.get('id')}")
        return data.get('id', '')

    def search_extract_data(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search extracted data by embedding similarity."""
        self._ensure_connected()
        self.create_extract_data_collection()
        collection = Collection("extract_data")
        collection.load()

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        filter_conditions = []
        if user_id:
            filter_conditions.append(f'user_id == "{user_id}"')
        if document_id:
            filter_conditions.append(f'contract_id == "{document_id}"')

        filters = " and ".join(filter_conditions) if filter_conditions else None

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filters,
            output_fields=["id", "user_id", "contract_id", "chunk_index", "content"]
        )

        chunks = []
        for hits in results:
            for hit in hits:
                chunks.append({
                    "id": hit.entity.get("id"),
                    "user_id": hit.entity.get("user_id"),
                    "contract_id": hit.entity.get("contract_id"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "content": hit.entity.get("content"),
                    "score": hit.score
                })

        return chunks

    # ==================== OBLIGATIONS COLLECTION ====================

    def create_obligations_collection(self) -> Collection:
        """Create the obligations collection for storing extracted obligations."""
        self._ensure_connected()

        collection_name = "obligations"
        dimension = 768

        if utility.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return Collection(collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="contract_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="obligation_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="due_date", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="party_responsible", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="recurrence", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="priority", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="original_text", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=30),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]

        schema = CollectionSchema(fields=fields, description="Extracted obligations from contracts")
        collection = Collection(name=collection_name, schema=schema)

        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

        logger.info(f"Created collection: {collection_name}")
        return collection

    def insert_obligation(self, obligation: Dict[str, Any], embedding: List[float]) -> str:
        """Insert an obligation into the obligations collection."""
        self._ensure_connected()
        self.create_obligations_collection()
        collection = Collection("obligations")

        insert_data = [
            [obligation.get("id", "")[:100]],
            [obligation.get("contract_id", "")[:64]],
            [obligation.get("user_id", "test_user")[:64]],
            [obligation.get("type", "other")[:50]],
            [obligation.get("description", "")[:500]],
            [obligation.get("due_date", "")[:50]],
            [obligation.get("party_responsible", "")[:200]],
            [obligation.get("recurrence", "none")[:20]],
            [obligation.get("priority", "medium")[:10]],
            [obligation.get("original_text", "")[:1000]],
            [obligation.get("status", "pending")[:20]],
            [obligation.get("created_at", "")[:30]],
            [embedding]
        ]

        collection.insert(insert_data)
        collection.flush()

        logger.info(f"Inserted obligation: {obligation.get('id')}")
        return obligation.get("id", "")

    def search_obligations(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        user_id: Optional[str] = None,
        contract_id: Optional[str] = None,
        obligation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search obligations by embedding similarity with optional filters."""
        self._ensure_connected()
        self.create_obligations_collection()
        collection = Collection("obligations")
        collection.load()

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        filter_conditions = []
        if user_id:
            filter_conditions.append(f'user_id == "{user_id}"')
        if contract_id:
            filter_conditions.append(f'contract_id == "{contract_id}"')
        if obligation_type:
            filter_conditions.append(f'obligation_type == "{obligation_type}"')

        filters = " and ".join(filter_conditions) if filter_conditions else None

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filters,
            output_fields=["id", "contract_id", "user_id", "obligation_type", "description",
                          "due_date", "party_responsible", "recurrence", "priority",
                          "original_text", "status", "created_at"]
        )

        obligations = []
        for hits in results:
            for hit in hits:
                obligations.append({
                    "id": hit.entity.get("id"),
                    "contract_id": hit.entity.get("contract_id"),
                    "user_id": hit.entity.get("user_id"),
                    "obligation_type": hit.entity.get("obligation_type"),
                    "description": hit.entity.get("description"),
                    "due_date": hit.entity.get("due_date"),
                    "party_responsible": hit.entity.get("party_responsible"),
                    "recurrence": hit.entity.get("recurrence"),
                    "priority": hit.entity.get("priority"),
                    "original_text": hit.entity.get("original_text"),
                    "status": hit.entity.get("status"),
                    "created_at": hit.entity.get("created_at"),
                    "score": hit.score
                })

        return obligations

    def get_obligations_by_contract(self, contract_id: str) -> List[Dict[str, Any]]:
        """Get all obligations for a specific contract."""
        self._ensure_connected()
        self.create_obligations_collection()
        collection = Collection("obligations")
        collection.load()

        results = collection.query(
            expr=f'contract_id == "{contract_id}"',
            output_fields=["id", "contract_id", "user_id", "obligation_type", "description",
                          "due_date", "party_responsible", "recurrence", "priority",
                          "original_text", "status", "created_at"]
        )

        logger.info(f"Found {len(results)} obligations for contract {contract_id}")
        return results
