from fastapi import FastAPI , UploadFile , File , Form , Depends , HTTPException
from contextlib import asynccontextmanager
from pipelines import processing_file_uploads, answer_question
from db import conn_string , create_pool , init_users_table , create_user , get_password_hash
from auth import hash_password , verify_password , create_token , get_current_user


@asynccontextmanager
async def lifespan(app:FastAPI) :
    app.state.pool = create_pool(conn_string)  # one pool for the whole app, connections are borrowed per query
    init_users_table(app.state.pool)
    yield
    app.state.pool.close()


app = FastAPI(lifespan=lifespan)


@app.post("/register")
def register(user_id:int = Form(...) , password:str = Form(...)) :
    if len(password) < 8 :
        raise HTTPException(status_code=400 , detail="Password must be at least 8 characters")
    if not create_user(app.state.pool , user_id , hash_password(password)) :
        raise HTTPException(status_code=409 , detail="That user ID is already taken")
    return {"registered": True , "user_id": user_id}


@app.post("/login")
def login(user_id:int = Form(...) , password:str = Form(...)) :
    stored_hash = get_password_hash(app.state.pool , user_id)
    # same message for "no such user" and "wrong password" so the API doesn't reveal which IDs exist
    if stored_hash is None or not verify_password(password , stored_hash) :
        raise HTTPException(status_code=401 , detail="Wrong user ID or password")
    return {"access_token": create_token(user_id) , "token_type": "bearer"}


@app.post("/upload")
def upload_file(file:UploadFile = File(...) , user_id:int = Depends(get_current_user)) :
    contents = file.file.read()
    return processing_file_uploads(contents, file.filename , user_id , app.state.pool )


@app.post("/ask")
def ask_question(question:str , user_id:int = Depends(get_current_user)) :
    with app.state.pool.connection() as conn :
        check = conn.execute("SELECT 1 FROM chunks WHERE user_id=%s LIMIT 1",(user_id,)).fetchone()
    if check == None :
        return {"answered": False, "answer": "The user has no files uploaded.", "reference": None, "top_chunks": []}
    return answer_question(question.strip(), user_id, app.state.pool)
