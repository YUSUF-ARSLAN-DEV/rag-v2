import faiss
import time 
from chunker import chunk ,save_chunks  , read_chunks
import ollama 
from  sentence_transformers import SentenceTransformer
import os 
import json 
from embedder import embed_chunks , populate_index , embed_question , read_embedding_index , save_embedding_index , save_embedding_index 
from document_loader import load_document
import numpy as np 
from config import chunk_size , overlap_size  , file_paths,  base_url
from model import get_client  , askQuestionToAI
from datasets import load_dataset 
from evaluate import evaluating_embedding_model ,reading_the_golden_set









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
    return [text_passed_to_AI , q_string,sources ] 

# [retrived text , sources list , the actual question string ]

# requires text_passed_to_AI;context
# requires source


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

    





def main():
    questions_embed , index , contexts , question_list  = reading_the_golden_set(False)

    for question in questions_embed :
       _,indices =  index.search(question.reshape(1,-1) , k=1)  # searching the index 
       context_piece = contexts[indices[0][0]]
       question_deencoded = question_list[indices[0][0]] # decoding the question then storing it as a decoded string
       q_stack = [context_piece,question_deencoded,None]
       askQuestionToAI(q_stack)
       time.sleep(10)

    







main() 