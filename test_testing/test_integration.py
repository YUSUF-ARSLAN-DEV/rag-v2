from db import init_schema , create_user  , create_pool ,  conn_string  ,     create_conversation


import pytest 
from fastapi.testclient import TestClient 

import uuid 

@pytest.fixture 
def two_users(): # we  are preparing the resoruces needed for this test , in this case 2 users creates , 2 conversation created 
    # initiliazing the schem a
    init_schema(conn_string) 
    pool = create_pool(conn_string) 
    from auth import hash_password 

    suffix = uuid.uuid4().hex[:8]
    b_name, b_password = f"user_b_{suffix}", "b-test-password" # creating credentials for user B 
    a_id = create_user(pool, f"user_a_{suffix}", hash_password("a-test-password"))# creating a record for user a in the databse 
    create_user(pool, b_name, hash_password(b_password)) # creating a record for user B in the databse 

    a_convo_id = create_conversation(pool, a_id) # creating a converstaion record for user a 

    yield {"a_convo_id": a_convo_id, "b_name": b_name, "b_password": b_password}


    pool.close() 

@pytest.mark.integration 
def test_user_b_accesses_user_A_message(two_users):
    from api import app 
    with TestClient(app) as client : 
        login = client.post("/login" , data={"username": two_users["b_name"],
                                            "password": two_users["b_password"]})
        assert login.status_code == 200 
        token = login.json()["access_token"]


        response  = client.get(f"/conversations/{two_users['a_convo_id']}",headers={"Authorization":f"Bearer {token}"}) # accesing the convo for user_1 
        assert response.status_code == 404 