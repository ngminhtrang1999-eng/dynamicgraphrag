import os
from dotenv import load_dotenv

load_dotenv()

# Paths
INPUT_DIR = "data/LOTM/chapters"
OUTPUT_TRANS_DIR = "data/LOTM/translations"
DATA_DIR = "data/knowledge_base"

# Local Model Config
MODEL_NAME = "Qwen/Qwen3-4B"
MAX_NEW_TOKENS = 2000  # Tuned 2026-05-12: 16000 caused models to ramble meta-text into chunks; 2000 fits chunk translations comfortably

# Gemini Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") # Hãy set biến môi trường hoặc điền key vào đây
GEMINI_MODEL_NAME = "gemini-2.5-pro" # Hoặc gemini-2.5-pro # Cho evaluate: "gemini-3-pro-preview"

MODEL_TYPE = "strong"

# Groq Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Langdb Config
LANGDB_API_KEY = os.getenv("LANGDB_API_KEY", "")
LANGDB_PROJECT_ID = os.getenv("LANGDB_PROJECT_ID", "")

# OpenRouter Config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Nvidia Config
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# Cloudflare Config
CLOUDFARE_API_KEY = os.getenv("CLOUDFARE_API_KEY", "")
CLOUDFARE_ACCOUNT_ID = os.getenv("CLOUDFARE_ACCOUNT_ID", "")

# Ollama Config (local LLM via WSL)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# DeepSeek Config (judge model)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Strategy Params
WINDOW_SIZE = 5
START_CHAPTER = 6
END_CHAPTER = 10
# CHUNK_SIZE = 300 
# CHUNK_SIZE_MINING = 500
CHUNK_SIZE = 500 
CHUNK_SIZE_MINING = 700  

# Temperature Strategy
TEMP_MINING = 0.8
TEMP_INJECTION = 0.1  # Low temperature for deterministic entity annotation
TEMP_CREATIVE = 1.3
TEMP_EDITING = 0.3  # Low temperature for deterministic post-editing