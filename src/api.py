from fastapi import FastAPI , UploadFile , File , Form
from contextlib import asynccontextmanager
from pipelines import processing_file_uploads, answer_question
from db import conn_string , create_pool


@asynccontextmanager
async def lifespan(app:FastAPI) :
    app.state.pool = create_pool(conn_string)  # one pool for the whole app, connections are borrowed per query
    yield
    app.state.pool.close()


app = FastAPI(lifespan=lifespan)


@app.post("/upload")
def upload_file(user_id:int = Form(...) , file:UploadFile = File(...) ) :
    contents = file.file.read()
    return processing_file_uploads(contents, file.filename , user_id , app.state.pool )


@app.post("/ask")
def ask_question(question:str ,user_id:int) :
    with app.state.pool.connection() as conn :
        check = conn.execute("SELECT 1 FROM chunks WHERE user_id=%s LIMIT 1",(user_id,)).fetchone()
    if check == None :
        return {"answered": False, "answer": "The user has no files uploaded.", "reference": None, "top_chunks": []}
    return answer_question(question.strip(), user_id, app.state.pool)
