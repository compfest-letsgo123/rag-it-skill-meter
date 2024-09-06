# Retrieval Augmented Generation (RAG)
## Compfest - Letsgo123
1. Hilmi Baskara Radanto
2. Syafiq Ziyadul Arifin
3. Riandra Diva Auzan

### How to Use
1. git clone https://github.com/compfest-letsgo123/rag-it-skill-meter
2. python3 -m venv venv
3. source venv/bin/activate
4. pip install -r requirements.txt
5. curl -fsSL https://ollama.com/install.sh | sh
6. ollama pull llama3.1:8b
7. ollama pull mxbai-embed-large
8. gunicorn --bind=0.0.0.0:8000 --workers=4 --worker-class uvicorn.workers.UvicornWorker --threads=4 main:app
