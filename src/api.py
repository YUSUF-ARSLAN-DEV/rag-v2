from fastapi import FastAPI , UploadFile , File , Form 
from contextlib import asynccontextmanager 
from pipelines import processing_file_uploads, answer_question
from embedder import build_faiss_index ,build_bm25_index
from document_loader import read_chunk_history , read_user_chunk_mapping
from config import index_dimension 
from db import create_connection , conn_string
import json 

async def lifespan(app:FastAPI ,  ) : 
    app.state.db_connection = create_connection(conn_string)
    yield 


app = FastAPI(lifespan=lifespan) 





@app.post("/upload")
async def upload_file(user_id:int = Form(...) , file:UploadFile = File(...) ) : 
    contents = await file.read() 
    flag = processing_file_uploads(contents, file.filename , user_id , app.state.db_connection ) 
    if flag : 
        return {"status":"Successful"}
    else : 
        return {"status":"Failed"}



@app.post("/ask") 
def ask_question(question:str ,user_id:int) :
    check = app.state.db_connection.execute("SELECT 1 FROM chunks WHERE user_id=%s LIMIT 1",(user_id,)).fetchone() 
    if check == None :
        return {"answered": False, "answer": "The user has no files uploaded.", "reference": None, "top_chunks": []}
    return answer_question(question.strip(), user_id, app.state.db_connection)


