from db import conn_string  , init_schema  , create_pool  , create_user 
from pathlib import Path 
from pipelines import processing_file_uploads 

# init_schema runs several SQL commands to iniitalize the schema in the db 

init_schema(conn_string) 
conn_pool = create_pool(conn_string) 
u_id = create_user(conn_pool,"evaluator_the_creator","FAKE_HASH_LAH")

# preparing the doc we will use in this workflow 
            # the parent direcotyr ofd this file 
pdf_path = Path(__file__).resolve().parent/"test_sets"/"wf.pdf"
result  = processing_file_uploads(pdf_path.read_bytes(),pdf_path.name , u_id , conn_pool ) # seeds our file as achunks into the database 
if result["status"] != "Successful":
    raise SystemExit(f"Seeding failed: {result}")
print(u_id) 
conn_pool.close()
