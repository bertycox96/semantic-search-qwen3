import os
import logging
import time
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, exceptions

# --- НАЛАШТУВАННЯ ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# [ОНОВЛЕНО] Глобальні константи для нової моделі
INDEX_NAME = "products_qwen3"  # Новий індекс, щоб уникнути конфліктів
EMBEDDING_DIMS = 4096  # Розмірність векторів для Qwen3-Embedding-8B
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")

# Ініціалізація клієнта Elasticsearch
try:
    es = Elasticsearch(os.getenv("ELASTICSEARCH_URL"))
    logging.info("Успішно налаштовано клієнт Elasticsearch.")
except Exception as e:
    logging.error(f"Критична помилка під час ініціалізації Elasticsearch: {e}")
    raise

# --- FastAPI App та CORS ---
app = FastAPI(title="Semantic Product Search API with Qwen3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Моделі даних Pydantic ---
class Product(BaseModel):
    product_id: str
    description: str
    category: str
    price: float

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10

# --- Логіка Elasticsearch ---
def create_index_if_not_exists():
    """Створює індекс з мапінгом, адаптованим під нову розмірність векторів."""
    if not es.indices.exists(index=INDEX_NAME):
        logging.info(f"Індекс '{INDEX_NAME}' не знайдено. Створюємо новий...")
        mapping = {
            "properties": {
                "product_id": {"type": "keyword"},
                "description": {"type": "text"},
                "category": {"type": "keyword"},
                "price": {"type": "float"},
                "embedding": {"type": "dense_vector", "dims": EMBEDDING_DIMS, "index": True, "similarity": "cosine"}
            }
        }
        es.indices.create(index=INDEX_NAME, mappings=mapping)
        logging.info(f"Створено індекс '{INDEX_NAME}' з розмірністю векторів {EMBEDDING_DIMS}.")

# --- Функція запуску з retry-логікою ---
@app.on_event("startup")
def startup_event():
    """Перевіряє доступність сервісів при старті з логікою повторних спроб."""
    max_retries = 30
    retry_delay = 5
    for i in range(max_retries):
        try:
            if es.ping():
                logging.info("Успішно підключено до Elasticsearch.")
                create_index_if_not_exists()
                return
        except exceptions.ConnectionError:
            logging.warning(f"Спроба {i+1}/{max_retries}: очікуємо Elasticsearch. Повтор через {retry_delay} сек...")
            time.sleep(retry_delay)
    raise RuntimeError("Не вдалося підключитися до Elasticsearch після багатьох спроб.")

# --- Логіка Гібридного Пошуку [ПОВНІСТЮ ПЕРЕРОБЛЕНО ЯДРО] ---
def generate_embedding(text: str) -> list[float]:
    """
    [ОНОВЛЕНО] Генерує ембединг через локальний сервіс text-embeddings-inference.
    Замість виклику API Google, робить локальний HTTP-запит.
    """
    if not EMBEDDING_API_URL:
        logging.error("Змінна середовища EMBEDDING_API_URL не встановлена!")
        return []

    try:
        response = requests.post(
            EMBEDDING_API_URL,
            json={"inputs": text},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        embedding = response.json()[0]
        return embedding
    except requests.exceptions.RequestException as e:
        logging.error(f"Помилка підключення до локального сервісу ембедингів: {e}")
        return []
    except (KeyError, IndexError) as e:
        logging.error(f"Отримана неочікувана відповідь від сервісу ембедингів: {e}")
        return []

def keyword_search(query: str, top_k: int) -> list[str]:
    search_query = {"query": {"match": {"description": {"query": query, "fuzziness": "AUTO"}}}}
    response = es.search(index=INDEX_NAME, body=search_query, size=top_k, _source=False)
    return [hit["_id"] for hit in response["hits"]["hits"]]

def vector_search(query_vector: list[float], top_k: int) -> list[str]:
    if not query_vector: return []
    knn_query = {"knn": {"field": "embedding", "query_vector": query_vector, "k": top_k, "num_candidates": 100}}
    response = es.search(index=INDEX_NAME, body=knn_query, size=top_k, _source=False)
    return [hit["_id"] for hit in response["hits"]["hits"]]

def reciprocal_rank_fusion(results_lists: list[list[str]], k: int = 60) -> list[str]:
    scores = {}
    for results in results_lists:
        for rank, doc_id in enumerate(results):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (rank + k)
    return sorted(scores, key=scores.get, reverse=True)

@app.post("/index-product", status_code=201)
def index_product(product: Product):
    embedding = generate_embedding(product.description)
    if not embedding:
        raise HTTPException(status_code=500, detail="Не вдалося згенерувати ембединг через локальний сервіс.")
    doc = product.model_dump()
    doc["embedding"] = embedding
    es.index(index=INDEX_NAME, id=product.product_id, document=doc)
    return {"status": "indexed", "product_id": product.product_id}

@app.post("/search")
def search(request: SearchRequest):
    logging.info(f"Отримано запит: '{request.query}'")
    query_vector = generate_embedding(request.query)
    keyword_results = keyword_search(request.query, request.top_k)
    vector_results = vector_search(query_vector, request.top_k)
    fused_ids = reciprocal_rank_fusion([keyword_results, vector_results])
    if not fused_ids:
        return {"results": []}
    top_ids = fused_ids[:request.top_k]
    response = es.mget(index=INDEX_NAME, body={"ids": top_ids})
    id_to_doc = {doc["_id"]: doc["_source"] for doc in response["docs"] if doc.get("found")}
    sorted_results = [id_to_doc[id] for id in top_ids if id in id_to_doc]
    return {"results": sorted_results}
