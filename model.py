from openai import OpenAI 
import os 

def get_client() : 
    client = OpenAI(
        base_url = "http://localhost:11434/v1",
        api_key = "something key"
    )
    return client 
