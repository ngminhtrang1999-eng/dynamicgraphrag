import os
from dotenv import load_dotenv

load_dotenv()

# Paths
INPUT_DIR = "data/LOTM/chapters"
OUTPUT_TRANS_DIR = "data/LOTM/translations"
DATA_DIR = "data/knowledge_base"

# Local Model Config
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 16000

# Gemini Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") # Hãy set biến môi trường hoặc điền key vào đây
GEMINI_MODEL_NAME = "gemini-2.5-flash" # Hoặc gemini-2.5-pro # Cho evaluate: "gemini-3-pro-preview"

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

# Strategy Params
WINDOW_SIZE = 5
START_CHAPTER = 7
END_CHAPTER = 10
CHUNK_SIZE = 600 # Strong model = 1200, weak model = 750

# Temperature Strategy
TEMP_MINING = 0.05
TEMP_CREATIVE = 0.5