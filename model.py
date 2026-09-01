from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

def get_client(local=False):
    if local:
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
    else:
        client = OpenAI(
            base_url=os.getenv("LOCAL_SERVER_LINK"),
            api_key=os.getenv("LOCAL_SERVER_API_KEY")
        )
    return client

def askQuestionToAI(q_stack, local=False):
    refrence_text = q_stack[0]
    question = q_stack[1]
    sources = q_stack[-1]
    client = get_client(local)
    model_name = "qwen3.5:9b" if local else os.getenv("LOCAL_SERVER_MODEL_NAME")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the context provided. If the answer is not in the context, say 'I don't know'."
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{refrence_text}\n\nQUESTION:\n{question}"
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": define_LLM_EVALUATION_SCHEMA(1)
        }
        }
    )

    TheAIResponse = response.choices[0].message.content


    if sources is not None:
        print("Sources for the answer are : \n\n")
        for source in sources:
            print(source)
    else:
        print("No Sources ")

    return TheAIResponse , client , model_name , refrence_text , question    # the JSON object the AI returns # passing up the client 
     # so that we do not define it again 

def ask_AI_TO_EVALUTE_RESPONSE(response , client , model_name,refrence_text , question,expected_answer  ): # evaluates a single resposne 
    response_dict = json.loads(response)
    evaluation_schema = define_LLM_EVALUATION_SCHEMA(1)
    answer  = response_dict["answer"] 
    response = client.chat.completions.create(
        model  = model_name , 
        messages = 
        [
            {"role":"system","content":"You are an AI evlauation engineer , of the highest skilll and rank , working on evaluating the reponse of  a model , the model is given a piece of context and a question then it is asked to answer the question using the context , your job is to inspect the question , the context , and make judgement on wether the model answered correctly using the context you will have the schema that you must fill provided to you "}, 
            {"role":"user","content":f"This is the CONTEXT:\n{refrence_text}\n\n and this is the QUESTION:\n{question}\n\nThis is the model's answer:{answer} and finally this is expected and correct answer for this question:{expected_answer} The is correct part of the output is purely determined on wether the model's answer and the expected correct answer for the question match in terms of meaning "}
        ], 
        response_format = {"type":"json_schema", "json_schema":{"name":"THE_JSON_SCHEMA" , "schema":define_LLM_EVALUATION_SCHEMA(0)}}
    )
    EVALUATION_RESULT = json.loads(response.choices[0].message.content)
    is_faithful = EVALUATION_RESULT["is_faithful"]
    print(is_faithful) 
    is_correct = EVALUATION_RESULT["is_correct"]
  

    return is_faithful , is_correct






def define_LLM_EVALUATION_SCHEMA(schema=0):
    # 0 mode outputs a set schema for a set test
    # in this case the test is us testing , the LLM ability to distinguish and evaluate the other LLM's response
    # schema 0 - aka the evaluation schema :
    if schema == 0:
        judge_schema = {
            "type": "object",
            "properties": {
                "is_faithful": {"type": "boolean", "description": "True if the answer only uses information from the provided context"},
                "is_correct": {"type": "boolean", "description": "True if the answer matches the expected answer in terms of meaning"},
                "reasoning": {"type": "string", "description": "Brief Explanation as to why the previous judgements were made"}
            },
            "required": ["is_faithful", "is_correct", "reasoning"]
        }
        return judge_schema
    elif schema == 1 : 
        answer_schema =  {
            "type":"object" , 
            "properties":  {
                "answer": {"type":"string","description":"Just Based on the given context write the answer or if you are unable To extract the answer solely from the context just write False - strictly as False "},
                "answered": {"type":"boolean","description":"When you are able to answer place a value of True here when not able to place  a value of False here "}
            },
            "required":["answer","answered"]
        }
        return answer_schema 
