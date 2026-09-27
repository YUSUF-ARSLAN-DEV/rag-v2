from fastapi import FastAPI 
from embedder import build_indices 
from contextlib import asynccontextmanager 
from embedder import build_faiss_index ,build_bm25_index
from config import d 
import json 
app = FastAPI() 




async def lifespan(app:FastAPI ,  ) : 
    faiss = build_faiss_index(d,True)
    app.state.faiss_index = faiss 
    app.state.bm25 = None 
    app.sate.user_chunk_mapping = 

    yield 