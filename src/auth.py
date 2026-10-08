import os
import time
import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import HTTPException , Depends
from fastapi.security import HTTPBearer , HTTPAuthorizationCredentials

load_dotenv()
JWT_SECRET = os.getenv("JWT_SECRET")  # long random string in .env, never in the code
if not JWT_SECRET :
    raise RuntimeError("JWT_SECRET is not set in .env")
TOKEN_LIFETIME_SECONDS = 60 * 60 * 8  # a login stays valid for 8 hours

bearer = HTTPBearer()  # reads the "Authorization: Bearer <token>" header


def hash_password(password:str) -> str :
    # bcrypt adds a random salt and stores it inside the hash, so two users with the same password get different hashes
    return bcrypt.hashpw(password.encode("utf-8") , bcrypt.gensalt()).decode("utf-8")


def verify_password(password:str , password_hash:str) -> bool :
    return bcrypt.checkpw(password.encode("utf-8") , password_hash.encode("utf-8"))


def create_token(user_id:int) -> str :
    # the token says "this is user N until <exp>" and is signed with JWT_SECRET, so it can't be edited by the client
    payload = {"sub": str(user_id) , "exp": int(time.time()) + TOKEN_LIFETIME_SECONDS}
    return jwt.encode(payload , JWT_SECRET , algorithm="HS256")


def get_current_user(credentials:HTTPAuthorizationCredentials = Depends(bearer)) -> int :
    # FastAPI runs this before any route that lists it with Depends(); the route then receives the user_id
    try :
        payload = jwt.decode(credentials.credentials , JWT_SECRET , algorithms=["HS256"])
        return int(payload["sub"])
    except (jwt.PyJWTError , KeyError , ValueError) :
        raise HTTPException(status_code=401 , detail="Invalid or expired token")
