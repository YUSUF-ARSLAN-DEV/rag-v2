import psycopg  as pgre
from dotenv import load_dotenv 
import os 
load_dotenv() 
pw = os.getenv("POSTGRES_PASSWORD")
conn = pgre.connect(f"postgresql://postgres:{pw}@localhost:5432/postgres")


# Engineering the Database 
# creating the database and the tables 



conn.execute("CREATE TABLE IF NOT EXISTS DOCUMENTS (id SERIAL PRIMARY KEY , user_id INTEGER NOT NULL , filename TEXT , content_hash TEXT NOT NULL , UNIQUE(user_id , content_hash)); ")
conn.execute(" CREATE TABLE IF NOT EXISTS  CHUNKS (id SERIAL PRIMARY KEY , user_id INTEGER NOT NULL , document_id INTEGER REFERENCES DOCUMENTS(id) , chunk_text TEXT  , chunk_vector vector(1024)  );")
conn.commit() 