import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_URL = os.getenv("BASE_URL")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    DEFAULT_AI_MODEL = os.getenv("DEFAULT_AI_MODEL")
    
    MAX_RETRIES = 3
    MAX_FILE_SIZE = 1024*1024*10
    MAX_OUTPUT_TOKENS = 25000

config = Config()
