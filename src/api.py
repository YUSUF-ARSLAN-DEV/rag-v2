from fastapi import FastAPI , UploadFile , File , Form , Depends , HTTPException, Query 
from fastapi.responses import FileResponse
from pathlib import Path
from contextlib import asynccontextmanager
from pipelines import processing_file_uploads, answer_question
from db import conn_string , create_pool , init_schema , get_user_documents , create_user , get_user_by_username , list_conversations , get_conversation_messages , delete_conversation
from auth import hash_password , verify_password , create_token , get_current_user


@asynccontextmanager
async def lifespan(app:FastAPI) :
    init_schema(conn_string)  # creates the extension + tables on a fresh database
    app.state.pool = create_pool(conn_string)  # one pool for the whole app, connections are borrowed per query
    yield
    app.state.pool.close()


app = FastAPI(lifespan=lifespan)


@app.get("/" , include_in_schema=False)
def home() :
    # the single-page UI; it talks to the routes below with fetch()
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/register")
def register(username:str = Form(...) , password:str = Form(...)) :
    username = username.strip().lower()  # lowercase so "Yusuf" and "yusuf" count as the same username
    if not username :
        raise HTTPException(status_code=400 , detail="Username cannot be empty")
    if len(password) < 8 :
        raise HTTPException(status_code=400 , detail="Password must be at least 8 characters")
    user_id = create_user(app.state.pool , username , hash_password(password))
    if user_id is None :
        raise HTTPException(status_code=409 , detail="That username is already taken")
    return {"registered": True , "username": username}


@app.post("/login")
def login(username:str = Form(...) , password:str = Form(...)) :
    user = get_user_by_username(app.state.pool , username.strip().lower())
    # same message for "no such user" and "wrong password" so the API doesn't reveal which usernames exist
    if user is None or not verify_password(password , user[1]) :
        raise HTTPException(status_code=401 , detail="Wrong username or password")
    return {"access_token": create_token(user[0]) , "token_type": "bearer"}  # the token carries the internal user_id


@app.post("/upload")
def upload_file(file:UploadFile = File(...) , user_id:int = Depends(get_current_user)) :
    contents = file.file.read()
    return processing_file_uploads(contents, file.filename , user_id , app.state.pool )


@app.get("/documents")
def list_documents(user_id:int = Depends(get_current_user)) :
    return {"documents": get_user_documents(app.state.pool , user_id)}


@app.get("/conversations")
def get_conversations(user_id:int = Depends(get_current_user)) :
    # this user's chats, most recently active first
    return {"conversations": list_conversations(app.state.pool , user_id)}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id:int , user_id:int = Depends(get_current_user)) :
    messages = get_conversation_messages(app.state.pool , user_id , conversation_id)
    if messages is None :   # doesn't exist OR belongs to someone else: same answer on purpose
        raise HTTPException(status_code=404 , detail="Conversation not found")
    return {"conversation_id": conversation_id , "messages": messages}


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id:int , user_id:int = Depends(get_current_user)) :
    if not delete_conversation(app.state.pool , user_id , conversation_id) :   # False = not found / not yours
        raise HTTPException(status_code=404 , detail="Conversation not found")
    return {"deleted": True , "conversation_id": conversation_id}


@app.post("/ask")
def ask_question(question:str ,documents_id:list[int]=Query(default=[]) ,user_id:int = Depends(get_current_user), ) :
    with app.state.pool.connection() as conn :
        if not documents_id :  # below is a check to make sure that the user has chunks or not 
            check = conn.execute("SELECT 1 FROM chunks WHERE user_id=%s  LIMIT 1",(user_id,)).fetchone()
        else:
            check = conn.execute("SELECT 1 FROM chunks WHERE user_id=%s AND document_id = Any(%s) LIMIT 1",(user_id,documents_id)).fetchone()
        if check == None :
            return {"answered": False, "answer": "The user has no files uploaded.", "reference": None, "top_chunks": []}
    return answer_question(question.strip(), user_id, app.state.pool ,documents_id  = documents_id)
