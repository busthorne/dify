import json
import psycopg2.extras
import psycopg2.pool
from typing import Any
from uuid import uuid4
from contextlib import contextmanager

from configs import dify_config
from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.vector_factory import AbstractVectorFactory
from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.embedding.embedding_base import Embeddings
from core.rag.models.document import Document
from extensions.ext_redis import redis_client
from models.dataset import Dataset

_pool = psycopg2.pool.SimpleConnectionPool(
        dify_config.PGVECTO_RS_MIN_CONNECTIONS,
        dify_config.PGVECTO_RS_MAX_CONNECTIONS,
        host=dify_config.PGVECTO_RS_HOST,
        port=dify_config.PGVECTO_RS_PORT,
        user=dify_config.PGVECTO_RS_USER,
        password=dify_config.PGVECTO_RS_PASSWORD,
        database=dify_config.PGVECTO_RS_DATABASE)

class PGVectoRS(BaseVector):

    def __init__(self, collection_name: str):
        super().__init__(collection_name)

    def get_type(self) -> str:
        return VectorType.PGVECTO_RS

    @contextmanager
    def cursor(self):
        conn = _pool.getconn()
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
            conn.commit()
            _pool.putconn(conn)

    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        self.create_collection(len(embeddings[0]))
        self.add_texts(texts, embeddings)

    def create_collection(self, dimension: int):
        collection = self._collection_name
        lock_name = f"vector_indexing_lock_{collection}"
        with redis_client.lock(lock_name, timeout=20):
            race_control = f"vector_indexing_{collection}"
            if redis_client.get(race_control):
                return
            with self.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vectors")

                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {collection} (
                    id UUID PRIMARY KEY,
                    text TEXT NOT NULL,
                    meta JSONB NOT NULL,
                    vector vector({dimension}) NOT NULL
                ) using heap
                """)
                dense = f"{collection}_on_embedding"
                cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {dense}
                    ON {collection}
                    USING vectors (vector vector_l2_ops)
                    WITH (options = $$
                        optimizing.optimizing_threads = 30
                        segment.max_growing_segment_size = 2000
                        segment.max_sealed_segment_size = 30000000
                        [indexing.hnsw]
                        m=30
                        ef_construction=500
                        $$)
                """)
                gin = f"{collection}_on_metadata"
                cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {gin}
                    ON {collection}
                    USING gin (meta)
                """)
                gin = f"{collection}_on_tsvector"
                cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {gin}
                    ON {collection}
                    USING gin (to_tsvector('simple', text))
                """)
            redis_client.set(race_control, 1, ex=3600)

    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        collection = self._collection_name
        docs = []
        with self.cursor() as cur:
            for document, embedding in zip(documents, embeddings):
                docs.append((uuid4(),
                             document.page_content,
                             json.dumps(document.metadata),
                             embedding))
            psycopg2.extras.execute_values(
                    cur,
                    f"""
                    INSERT INTO {collection}
                        (id, text, meta, vector)
                        VALUES %s
                    """,
                    docs,
                    template="(%s, %s, %s, %s::real[])")
        return [d[0] for d in docs]

    def get_ids_by_metadata_field(self, key: str, value: str):
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT id
                    FROM {self._collection_name}
                    WHERE meta->>%s = %s
                """,
                (key, value))
            cur.fetchall()
            ids = cur.fetchall()
        return [x[0] for x in ids or []]

    def delete_by_metadata_field(self, key: str, value: str):
        with self.cursor() as cur:
            cur.execute(f"""
                DELETE
                    FROM {self._collection_name}
                    WHERE meta->>%s = %s
                """,
                (key, value))

    def delete_by_ids(self, ids: list[str]) -> None:
        with self.cursor() as cur:
            cur.execute(f"""
                DELETE
                    FROM {self._collection_name}
                    WHERE id IN %s
                """,
                (tuple(ids),))

    def delete(self) -> None:
        collection = self._collection_name
        with self.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {collection}")

    def text_exists(self, id: str) -> bool:
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT count(*)
                    FROM {self._collection_name}
                    WHERE id = %s
                """,
                (id,))
        return cur.fetchone()[0] > 0

    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        collection = self._collection_name
        top_k = int(kwargs.get("top_k", 2))
        threshold = float(kwargs.get("score_threshold") or 0.0)
        docs = []
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT
                    meta,
                    text,
                    vector <=> %s::real[] as distance
                FROM {collection}
                ORDER BY distance
                LIMIT {top_k}
                """,
                (query_vector,))
            for record in cur.fetchall():
                metadata, text, distance = record
                score = 1 - distance
                metadata["score"] = score
                if score > threshold:
                    docs.append(Document(
                        page_content=text,
                        metadata=metadata))
        return docs

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        collection = self._collection_name
        top_k = int(kwargs.get("top_k", 5))
        docs = []
        with self.cursor() as cur:
            cur.execute(f"""
            SELECT
                meta,
                text,
                ts_rank(to_tsvector(text), query) as score
            FROM {collection},
                 plainto_tsquery(%s) as query
            WHERE to_tsvector('simple', text) @@ query
            ORDER BY score DESC
            LIMIT {top_k}
            """,
            (query,))
            for record in cur.fetchall():
                metadata, text, score = record
                metadata["score"] = score
                docs.append(Document(
                    page_content=text,
                    metadata=metadata))
        return docs

class PGVectoRSFactory(AbstractVectorFactory):
    def init_vector(self, dataset: Dataset, attributes: list, embeddings: Embeddings) -> PGVectoRS:
        if dataset.index_struct_dict:
            store = dataset.index_struct_dict["vector_store"]
            collection_name = store["class_prefix"].lower()
        else:
            collection_name = dataset.id.split('-')[0]
            sd = self.gen_index_struct_dict(
                    VectorType.PGVECTO_RS,
                    collection_name)
            dataset.index_struct = json.dumps(sd)
        return PGVectoRS(collection_name)
