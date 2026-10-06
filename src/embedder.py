from rank_bm25 import BM25Okapi
from sentence_transformers  import SentenceTransformer , CrossEncoder 
import faiss
from document_loader import load_document
from chunker import chunk
from config import hybdrid_embedding_top_k  ,c_rff_value

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
    tokenized_chunks = [c[1].lower().split() for c in chunks]  # same tokenizer used at query time
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

def RFF_TOP_PICKS(faiss_rows , bm25_rows) :
    # both inputs are lists of (chunk_id, chunk_text) rows, best match first
    scores = {}   # chunk_id -> total RFF score
    texts = {}    # chunk_id -> chunk_text
    for rows in (faiss_rows , bm25_rows) :
        for rank , (chunk_id , text) in enumerate(rows , start=1) :
            scores[chunk_id] = scores.get(chunk_id , 0) + 1/(rank + c_rff_value) # same id in both lists -> scores add up
            texts[chunk_id] = text
    arranged = sorted(scores , key=scores.get , reverse=True ) # The highest RFF SCore
    return [texts[chunk_id] for chunk_id in arranged ][:hybdrid_embedding_top_k]
            
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
