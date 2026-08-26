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

def askQuestionToAI(q_stack,evaluation_mode = False ):
    if evaluation_mode == True :
        additional_propmpt = '''
    We have passed to you a piece of context and a question with it, now attempt to solve the question from that piece of context only , 
    only that piece of context never attempt to solve the question from your own knowledge  and in the case of solving it be impossible then write so , 
    also whenver you get the answer from the piece of context briefly mention that in the reponse 
    '''
    refrence_text = q_stack[0]
    question = q_stack[1]
    sources = q_stack[-1]
    client = get_client() # end point we are gonna use to communicate 
    response = client.chat.completions.create(
        model="qwen3.5:9b",
        messages = 
        [
            {"role":"system","content":"You are a helpful assistant that answers questions based on the context provided. If the answer is not in the context, say 'I don't know'."},
            {"role": "user", "content": f"CONTEXT:\n{refrence_text }\n\nQUESTION:\n{question}"}
        ]
    )
        
 


    print("\n\nThe answer to your question is : \n\n")
    print(response.choices[0].message.content)  

    if sources is not None :
        print("Sources for the answer are : \n\n")
        for source in sources:
            print(source)
    else :
        print("No Sources for the golden set ")
