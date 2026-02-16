import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "phi3:mini"
TEMPERATURE = 0.7
MAX_TOKENS = 2048

# RAG Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RESULTS = 4

# Collections
DEFAULT_COLLECTION = "default"

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}