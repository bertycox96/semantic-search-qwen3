# Semantic Search with Qwen3-Embedding-8B + Elasticsearch (Docker)

Цей проєкт зібраний за вашою документацією: локальний сервіс ембедингів на GPU (Qwen3-Embedding-8B), Elasticsearch як векторна БД та FastAPI бекенд для гібридного пошуку.

## Структура
```
semantic-search-qwen3/
├─ docker-compose.yml
├─ backend/
│  ├─ Dockerfile
│  ├─ main.py
│  ├─ requirements.txt
│  ├─ .env
│  ├─ index_data.py
│  └─ sample_data.jsonl
```

## Попередні вимоги
- NVIDIA GPU (≈16GB VRAM для 8B)
- Встановлені драйвери NVIDIA (`nvidia-smi` працює)
- NVIDIA Container Toolkit
- Docker / Docker Compose

## Запуск
1. В корені:
   ```bash
   docker compose up -d --build
   ```
2. Перевірити логи сервісу ембедингів (перший запуск може завантажувати 16+ ГБ ваг):
   ```bash
   docker logs embedding-service -f
   ```
   Очікуйте повідомлення на кшталт: `Ready`

3. Індексувати демо-дані (з хоста):
   ```bash
   python backend/index_data.py
   ```

4. Тестовий пошук:
   ```bash
   curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{ "query": "акумуляторний шуруповерт", "top_k": 5 }'
   ```

## Нотатки
- Бекенд читає налаштування з `backend/.env`.
- Ендпоінти бекенда: `/index-product`, `/search`.
- Ембединги генеруються через локальний сервіс за `EMBEDDING_API_URL`.
- Індекс Elasticsearch: `products_qwen3` з `dense_vector` (4096 dims, cosine).
