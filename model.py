from openai import OpenAI 
from config import base_url , model_name 
from dotenv import load_dotenv 
import os 

def get_client() : 
    client = OpenAI(
        base_url = base_url,
        api_key = os.getenv("OLLAMA_API_KEY") 
    )
    return client 
