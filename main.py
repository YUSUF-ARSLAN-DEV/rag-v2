from chunker import chunk
import ollama 
import os 
from embedder import embed_chunks , populate_index
from document_loader import load_document
import numpy as np 
from config import chunk_size , overlap_size  , file_paths, model_base_url , base_url
from model import get_client 

'''
test_paths = file_paths 
text_string = load_document(test_paths) 
chunks = chunk(text_string,chunk_size,overlap_size)
embed_chunks(chunks)
embeddings = embed_chunks(chunks)
index = populate_index(embeddings) 

# asking the question  then embedding the question 
question = input("Please write a question regarding the file that you just passed\n make sure that what you are asking about exists in the file\n").strip()
q_store = question 
question = [question] # since faiss accepts a 2d arra
q_embed = embed_chunks(question)


# Getting the closest 3 chunks  # we use the list of chunks and the index 
distances , indices = index.search(q_embed,3)

string_to_AI = "\n\n".join( chunks[k] for k in indices[0] ) 
# taking the indices then in the same loop retriveing the matching chunk then joining these 3 


client = get_client() 


response = client.chat.completions.create(
    model="qwen3.5:9b",
    messages = 
    [
        {"role":"system","content":"You are a helpful assistant that answers questions based on the context provided. If the answer is not in the context, say 'I don't know'."},
        {"role": "user", "content": f"CONTEXT:\n{string_to_AI}\n\nQUESTION:\n{q_store}"}
    ]
)

print("The answer to your question is : \n\n")
print(response.choices[0].message.content)  

'''

