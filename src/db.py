from psycopg_pool import ConnectionPool 
from pgvector.psycopg import register_vector 
from dotenv import load_dotenv 
import os 
import hashlib 
load_dotenv() 
host = os.getenv("POSTGRES_HOST","localhost")
pw = os.getenv("POSTGRES_PASSWORD")
conn_string = f"postgresql://postgres:{pw}@{host}:5432/postgres"

def create_pool(conn_string) : 
    return ConnectionPool(conn_string,min_size=2,max_size=5 , configure=register_vector )
def get_user_chunks(pool , user_id):
    # borrows a connection only for this query, then returns it to the pool
    with pool.connection() as conn :
        return conn.execute("SELECT id,chunk_text FROM chunks WHERE user_id=%s ORDER BY id",(user_id,),).fetchall()


def search_user_vectors(pool , user_id , question_vector , k ):
    with pool.connection() as conn :
        return conn.execute("SELECT  id , chunk_text FROM chunks WHERE user_id=%s ORDER BY chunk_vector <=> %s LIMIT %s",(user_id,question_vector,k )).fetchall()
    

def hash_text(list_of_strings):
    joined_strings = " ".join(list_of_strings)  # seperator is # do not change or else ahshing chanes 
    encoded = joined_strings.encode("utf-8")
    hashed_content =  hashlib.sha256(encoded).hexdigest() 
    return hashed_content
