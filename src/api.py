from fastapi import FastAPI , UploadFile , File , Form 
from contextlib import asynccontextmanager 
from pipelines import processing_file_uploads, answer_question
from embedder import build_faiss_index ,build_bm25_index
from document_loader import read_chunk_history , read_user_chunk_mapping
from config import index_dimension 
import json 

async def lifespan(app:FastAPI ,  ) : 
    faiss = build_faiss_index(index_dimension, read=True)
    app.state.faiss_index = faiss 
    app.state.bm25 = None 
    app.state.user_chunk_mapping = read_user_chunk_mapping() # it returns the file if they exist or creates thema nd returns 2 new empty files 
    app.state.all_chunks = read_chunk_history() 
    

    yield 


app = FastAPI(lifespan=lifespan) 





@app.post("/upload")
async def upload_file(user_id:int = Form(...) , file:UploadFile = File(...) ) : 
    contents = await file.read() 
    processing_file_uploads(contents , file.filename , app.state.faiss_index , app.state.all_chunks,user_id,app.state.user_chunk_mapping) 
    return {"status":"Successful"}



@app.post("/ask") 
def ask_question(question:str ,user_id:int) :
    faiss  = app.state.faiss_index # chunks already added to the index 
    this_users_chunk_indices = app.state.user_chunk_mapping[str(user_id)]
    this_users_chunks = [app.state.all_chunks[i] for i in this_users_chunk_indices ]
    text_from_chunks = [chunk["text"] for chunk in this_users_chunks ]
    bm25 = build_bm25_index(this_users_chunks) # build the bm25 index on the spot for that user 
    answer , refrence ,top_chunks= answer_question(question.strip(),faiss,app.state.all_chunks,bm25,this_users_chunk_indices) 
    if top_chunks == None :
        return {"This is the AI's Answer":answer , "Refrence Used":refrence}
    else :
        return {"Model was unable to answer":answer , "top_chunks":top_chunks }



