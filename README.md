I Followed the stated steps to execute the code:
1. python -m venv venv ( create virtual environment)
2. .\venv\Scripts\activate (activating venv)
3. pip install --upgrade -r requirements.txt (install dependencies as in the .txt file)
4. uvicorn maint:app --host 0.0.0.0 --port 8000
5. http://127.0.0.1:8000/docs (visit to test endpoints)

Embedding & Data Setup :
Data Cleaning: The headers, footers, and quotes were removed to make sure the model is only being trained on actual post content. Posts that were shorter than 100 characters were ignored to filter out "one-liner" type of fragments.
Model Selection: The model used is multi-qa-MiniLM-L6-cos-v1. This model is fine-tuned for semantic search and is best used in conjunction with Cosine/Inner Product similarity for short to medium-length texts.
Fuzzy Clustering (GMM)
Why GMM?:Since "hard assignments" were deemed impossible in the prompt's requirements, Gaussian Mixture Models (GMM) were used. Unlike K-Means clustering, GMM offers the advantage of a probability distribution. This allows for the possibility of a single document being in multiple categories at the same time (e.g., 70% Space, 30% Electronics).
Number of Clusters: 15 clusters were chosen. Enough to distinguish between different topics while broad enough to encompass the "messy" semantics of the 20 Newsgroups dataset.

Cluster-Routed Semantic Cache
First Principles: No external caching middleware (Redis/Memcached) was used. The cache is built using a Python-native dictionary mapping and FAISS indices.
The Architecture: This system uses Cluster-Routed Caching. Instead of one flat index, queries are first routed to their dominant cluster's "cache bin." This ensures the system remains efficient as the cache grows.
The Tunable Decision: The similarity threshold is set to 0.88.
Observation: A higher threshold ensures high precision (only near-identical queries hit the cache). A lower threshold would increase the hit rate but risks returning semantically "loose" matches.

FastAPI State Management

The Vector Store, GMM model, and Cache are initialized as global states. This ensures the models are loaded once into memory, allowing the service to handle incoming requests with sub-second latency.
