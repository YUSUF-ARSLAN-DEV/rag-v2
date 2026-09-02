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

    if local == False : 
        print("Sending your Queries to the Server we have at work ")
    else: 
        print("Asking your local OLLAMA MODEL RUNNING ON YOUR WEAK ASS GPU ")
    refrence_text = q_stack[0]
    question = q_stack[1]
    sources = q_stack[-1]
    client = get_client(local)
    model_name = "qwen3.5:9b" if local else os.getenv("LOCAL_SERVER_MODEL_NAME")
    response = client.chat.completions.create(
        model=model_name,
        temperature = 0 ,
        messages=[
            {
                "role": "system",
                "content": "You answer strictly and only from the CONTEXT provided below. Do not use any outside or prior knowledge. If the answer is not explicitly stated in the CONTEXT, reply exactly with \"I don't know\" and nothing else. Do not guess, infer beyond the text, or add information the CONTEXT does not contain. Quote or paraphrase only what the CONTEXT says."
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



    return TheAIResponse , client , model_name , refrence_text , question    # the JSON object the AI returns # passing up the client 
     # so that we do not define it again 

def ask_AI_TO_EVALUTE_RESPONSE(response , client , model_name,refrence_text , question,expected_answer  ): # evaluates a single resposne 
    response_dict = json.loads(response)
    if expected_answer == None : 
        expected_answer = "This question has no answer in the context correct = the model refused "
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
    val_from_AI  = response.choices[0].message.content
    try : 
        py_dict = json.loads(val_from_AI) # we break this into two steps so that we can print some of what the AI brought back 
    except json.JSONDecodeError:
        return None , None ,  f"JUDGE_PARSE_FAIL: {val_from_AI[:200]!r}"
    
    EVALUATION_RESULT = json.loads(response.choices[0].message.content)

    is_faithful = EVALUATION_RESULT["is_faithful"]
    reasoning = EVALUATION_RESULT["reasoning"]    
    is_correct = EVALUATION_RESULT["is_correct"]
  

    return is_faithful , is_correct , reasoning 






def define_LLM_EVALUATION_SCHEMA(schema=0):
    # 0 mode outputs a set schema for a set test
    # in this case the test is us testing , the LLM ability to distinguish and evaluate the other LLM's response
    # schema 0 - aka the evaluation schema :
    if schema == 0:
        judge_schema = {
            "type": "object",
            "properties": {
                "is_faithful": {"type": "boolean", "description": "True if the answer only uses information from the provided context"},
                "is_correct": {"type": "boolean", "description": "True if the answer matches the expected answer in terms of meaning if no correct answer exists in the context , is_correct is True only when the model refuses to answer ( said it doesn't know / not in context ) and false if it made up an answer that is not possible to come up with from the context "},
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
