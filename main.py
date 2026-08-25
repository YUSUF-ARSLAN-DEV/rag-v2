import faiss
from chunker import chunk ,save_chunks  , read_chunks
import ollama 
from  sentence_transformers import SentenceTransformer
import os 
import json 
from embedder import embed_chunks , populate_index , embed_question , read_embedding_index , save_embedding_index , save_embedding_index 
from document_loader import load_document
import numpy as np 
from config import chunk_size , overlap_size  , file_paths,  base_url
from model import get_client 
from datasets import load_dataset 









# returns [compiled string , source , q_string ]
def askQuestionToIndex(index,chunks):
    # asking the question  then embedding the question 
    question = input("Please write a question regarding the file that you just passed\n make sure that what you are asking about exists in the file\n").strip()
    q_string = question 
    q_embed = [embed_question(question)]
    q_final  = np.array(q_embed).astype("float32")  # convert the list to a numpy array and then to float32

    distances , indices =  index.search(q_final,3) # retrive the closest 3 answers 

    # chunks maintain the same order of creation as being populated in the index 
    # so the returned index is the same index as of the chunk's index  in the list 
    text_passed_to_AI = "\n\n".join (chunks[k]["text"] for k in indices[0])
    sources = [chunks[k]["source"] for k in indices[0]] # indices is a 2d array 
    return [text_passed_to_AI , sources,q_string ] 

# [retrived text , sources list , the actual question string ]

def askQuestionToAI(q_stack):
    refrence_text = q_stack[0]
    question = q_stack[-1]
    sources = q_stack[1]
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

    print("Sources for the answer are : \n\n")
    for source in sources:
        print(source)


def manual_initialization_pipeline(): #  chunk and embed manually every single time 
    
    test_paths = file_paths 
    text_string = load_document(test_paths)  # the raw strings 

    # strinsg that are now chunked int he format of tokens 
    chunks = chunk(text_string,chunk_size,overlap_size,test_paths)

    # These chunks  now become a list of vectors each with dimensions 1,384 
    embeddings = embed_chunks(chunks)

    # Creating and populating the index with the embeddings

    index = populate_index(embeddings) 
    return index ,chunks 

    
def automatic_initialization_pipeline(index_file_name , chunk_file_name):
    index = read_embedding_index(index_file_name )  # read the index from disk if it exists
    chunks = read_chunks(chunk_file_name)
    return index , chunks 

    
def gold_set_load_chunk_and_embed(): 
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open ("squad_gold.json","r", encoding = "utf_8") as f : 
        pairs = json.load(f) 

    chunks = [pair["context"] for pair in pairs ] # returns a list of strings 
    embedded = model.encode(chunks) # reutnr a list of vectors 
    index = faiss.IndexFlatL2(embedded.shape[1])
    index.add(embedded)
    # the index have been populated with the chunks 

    question_list = [pair["question"] for pair in pairs ] 
    # now we have a question list 
    embedded_questions = model.encode(question_list)
    # correct tally 
    hitat1 = 0 
    missat1 = 0 
    hitat5  = 0 
    missat5 = 0
    for question in embedded_questions : 
        # testing hit at 1 recall@k1
        for i, question in enumerate(embedded_questions):
            _, result_indices = index.search(question.reshape(1, -1), k=5)
            if result_indices[0][0] == i:
                hitat1 += 1
            else :
                missat1 +=1 

            if i in result_indices[0]:
                hitat5 += 1
            else :
                missat5 +=1 
    total_recall1 = hitat1+missat1
    total_recall5 = hitat5 +missat5

    hitat1 =f"Your Hitat1 accuracy is :\n{(hitat1/total_recall1)* 100 }% percent"
    hitat5 = f"Your Hitat5 accuracy is :\n{(hitat5/total_recall5)* 100 }% percent"
    print(hitat1)
    print(hitat5)





def main():
    gold_set_load_chunk_and_embed()






main() 