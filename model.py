from openai import OpenAI 
from config import model_base_url , model_name 
from dotenv import load_dotenv 
import os 

def get_client() : 
    client = OpenAI(
        base_url = model_base_url,
        api_key = os.getenv("OLLAMA_API_KEY") 
    )
    return client 
