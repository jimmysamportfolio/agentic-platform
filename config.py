import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_URL = os.getenv("BASE_URL")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    DEFAULT_AI_MODEL = os.getenv("DEFAULT_AI_MODEL")
    
    MAX_RETRIES = 3

config = Config()
