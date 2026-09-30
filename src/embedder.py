import os
from rank_bm25 import BM25Okapi
from sentence_transformers  import SentenceTransformer , CrossEncoder 
import faiss
from pathlib import Path 
from document_loader import load_document , save_user_chunk_mapping
from chunker import chunk  , save_chunks 
from config import hybdrid_embedding_top_k  ,c_rff_value , index_dimension , chunk_save_path  , shared_index_path 

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
reranker_model = CrossEncoder("BAAI/bge-reranker-base")
def fais_chunks_embedder(chunks) :
    # flattening the 2d lists of  tokens
    text_chunks = [c["text"] for c in chunks ] # extract the text from the chunks
    list_of_vectors = model.encode(text_chunks,show_progress_bar=True) # returns a list of vectors
    return list_of_vectors # returns n rows , 384 columsn

# BM25 is built ONCE at ingest time, from the chunks only - there is no "the question"
# yet at that point (many different questions get asked later, one at a time).

def build_bm25_index(chunks):
    tokenized_chunks = [c["text"].lower().split() for c in chunks]  # same tokenizer used at query time
    return BM25Okapi(tokenized_chunks)




    
# Called once per question, at query time, against the already-built index.
def bm25_search(bm25, question, k=hybdrid_embedding_top_k):
    tokenized_question = question.lower().split()  # must match build_bm25_index's tokenizer
    scores = bm25.get_scores(tokenized_question)
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k] # retriving the top indices 
    return top_k, [scores[i] for i in top_k]  # chunk indices + their BM25 scores, best first # the second returned is the actual scores 

def embed_question(question) :
    question_vector = model.encode(question,show_progress_bar=True) 
    return question_vector 

def RFF_TOP_PICKS(top_chunks_faiss_indices , top_chunks_bm25_indices,chunks) : 
    # calculating RFF scored for faiss 
    rff_faiss =  [1/(s+c_rff_value) for s, i in enumerate( top_chunks_faiss_indices,start=1 ) ]
    rff_bm25 = [1/(s+c_rff_value) for s, i in enumerate(top_chunks_bm25_indices,start = 1) ]
    master_dict = {} 
    i= 0 
    for f , b in zip(top_chunks_faiss_indices,top_chunks_bm25_indices ): # f,b being indices 
        if f not in master_dict : 
            master_dict[f] = rff_faiss[i] 
        else : 
            master_dict[f] += rff_faiss[i] 
        if b not in master_dict : #O(1) btw lolls 
            master_dict[b] = rff_bm25[i]
        else : 
             master_dict[b] += rff_bm25[i]
        i+=1 
    # we have built the dictionary 
    arranged = sorted([k for k in master_dict],key=lambda k:master_dict[k] , reverse = True ) # The highest RFF SCore 
    return [chunks[index]["text"] for index in arranged ][:hybdrid_embedding_top_k]
            
def reranker(question , chunks, k=5 ) :
    scores = reranker_model.predict(  [  (question , chunk)  for chunk in chunks  ] ) 
    # building the combined list  

    combined = [(chunk,score) for chunk ,score in zip(chunks,scores )]
    arranged = sorted(combined , key= lambda tup :tup[1] , reverse = True )[:k] # ararnging it using the score as the key but returning only the chunks 
    return arranged 


def populate_index(twodarray): # this method  returns a populated faiss index 

    d = twodarray.shape[1] # the dimensions of the vector # how many columns 
    index = faiss.IndexFlatL2(d) # 384 dimensions  - aka d 
    twodarray = twodarray.astype("float32")
    index.add(twodarray) 
    return index  # now we have a populated index 


def build_faiss_index( d, file_path=shared_index_path  ,read= False ):
    index = None 
    # checking if the file exists 
   
    if read == True : 
        if os.path.exists(file_path) :
            index = read_embedding_index(file_path)
        else :
            read = False 
            
    if read == False :  
        index = faiss.IndexFlatL2(d)

   
    return index 


def add_chunks_to_index(index ,chunks,vector,user_id_to_chunk_id , user_id , all_chunks ):
    start = index.ntotal 
    v_size = len(vector)
    index.add(vector)
    added_indices = [i for i in range(start,start+v_size)]
    all_chunks.extend(chunks) 
    user_id_to_chunk_id.setdefault(str(user_id) , []).extend(added_indices )
    save_user_chunk_mapping(user_id_to_chunk_id)
    save_chunks(all_chunks)
    save_embedding_index(index)
    # this methods adds the vectors to the index , updataes the list of all chunks , saves the chunks 
    # updates the list of user mpaping 

def save_embedding_index(index, file_path=shared_index_path) :
    file_path = Path(file_path)  # accept either a str or a Path, use it uniformly from here on
    file_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(file_path))  # faiss's C++ binding wants a plain str, not a Path
    print(f"Index saved successfully at {file_path}")


def read_embedding_index(file_path=shared_index_path) :
    file_path = Path(file_path)
    if not file_path.is_file() :
        print(f"The file {file_path} does not exist. Please save the index first before trying to read it.")
        return None
    return faiss.read_index(str(file_path))


