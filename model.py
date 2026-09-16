from openai import OpenAI
from document_loader import load_document
import anthropic
from config import enhancement_source_file_path 
from dotenv import load_dotenv
import os
import json

load_dotenv()

# Shared prompts - used by both the OpenAI-compatible path and the Claude path
# so the two backends are tested with identical instructions.
ANSWERER_SYSTEM_PROMPT = "You answer strictly and only from the CONTEXT provided below. Do not use any outside or prior knowledge. Before answering, thoroughly review the CONTEXT and verify that every entity, name, date, place, and relationship named in the QUESTION actually appears in the CONTEXT and means the same thing. If the QUESTION assumes a fact the CONTEXT does not state (for example a different century, a swapped or reversed relationship, or a person/place not mentioned), treat it as unanswerable. If the answer is explicitly stated in the CONTEXT, set answered to true and put the answer text in answer. If the answer is not explicitly stated in the CONTEXT, set answered to false and leave answer as an empty string. Do not guess, infer beyond the text, or add information the CONTEXT does not contain. Quote or paraphrase only what the CONTEXT says."

JUDGE_SYSTEM_PROMPT = "You are an AI evlauation engineer , of the highest skilll and rank , working on evaluating the reponse of  a model , the model is given a piece of context and a question then it is asked to answer the question using the context , your job is to inspect the question , the context , and make judgement on wether the model answered correctly using the context you will have the schema that you must fill provided to you "

CLAUDE_MODEL_NAME = "claude-sonnet-5"


def get_claude_client():
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
                "content": ANSWERER_SYSTEM_PROMPT
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
    answered = response_dict["answered"]  # whether the answering model claims it could answer from the context
    response = client.chat.completions.create(
        model  = model_name , 
        messages = 
        [
            {"role":"system","content":JUDGE_SYSTEM_PROMPT},
            {"role":"user","content":build_judge_user_prompt(refrence_text, question, answered, answer, expected_answer)}
        ], 
        response_format = {"type":"json_schema", "json_schema":{"name":"THE_JSON_SCHEMA" , "schema":define_LLM_EVALUATION_SCHEMA(0)}}
    )
    val_from_AI  = response.choices[0].message.content
    try : 
        py_dict = json.loads(val_from_AI) # we break this into two steps so that we can print some of what the AI brought back 
    except json.JSONDecodeError:
        return None , None ,  f"JUDGE_PARSE_FAIL: {val_from_AI[:200]!r}" , answered
    
    EVALUATION_RESULT = json.loads(response.choices[0].message.content)

    is_faithful = EVALUATION_RESULT["is_faithful"]
    reasoning = EVALUATION_RESULT["reasoning"]    
    is_correct = EVALUATION_RESULT["is_correct"]
  

    return is_faithful , is_correct , reasoning , answered






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
                "answer": {"type":"string","description":"The answer text extracted solely from the context. If you cannot answer from the context, leave this as an empty string."},
                "answered": {"type":"boolean","description":"True if you answered the question from the context, False if the context does not contain the answer."}
            },
            "required":["answer","answered"]
        }
        return answer_schema
    elif schema == 2 : 
        schemaa   =  { 
            "type":"object" , 
            "properties" :  
            {
                "context_blurb" : {"type":"string" , "description":"A 1-2 sentence blurb situating this chunk within the document: what topic/section it's from, with any vague references resolved. Do not summarize the chunk itself, do not add information the document doesn't state."}
            } , 
            "required" : ["context_blurb"]
        }
        return schemaa 


def build_judge_user_prompt(refrence_text, question, answered, answer, expected_answer):
    return (f"This is the CONTEXT:\n{refrence_text}\n\n and this is the QUESTION:\n{question}\n\n"
            f"The model set answered={answered} (False means it refused / said the context does not contain the answer). "
            f"This is the model's answer:{answer} and finally this is expected and correct answer for this question:{expected_answer} "
            f"The is correct part of the output is purely determined on wether the model's answer and the expected correct answer for the question match in terms of meaning . "
            f"If the expected answer states the question has no answer in the context, is_correct is True only when the model refused (answered=False).")


# ---------------------------------------------------------------------------
# Claude backend - twins of askQuestionToAI / ask_AI_TO_EVALUTE_RESPONSE.
# Same inputs, same return shapes, so main.py can swap backends without changes.
# Structured output is done with a forced single-tool call (Anthropic has no
# response_format=json_schema); the tool input IS the JSON object.
# ---------------------------------------------------------------------------

def _claude_structured_call(client, system_prompt, user_prompt, schema, tool_name):
    tool = {"name": tool_name, "description": "Return the structured result.", "input_schema": schema}
    resp = client.messages.create(
        model=CLAUDE_MODEL_NAME,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        tools=[tool],
        tool_choice={"type": "tool", "name": tool_name},
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    return block.input  # dict matching `schema`


def askQuestionToClaude(q_stack, local=False):  # local kept for signature parity, ignored
    print("Sending your Queries to CLAUDE (claude-sonnet-5)")
    refrence_text = q_stack[0]
    question = q_stack[1]
    client = get_claude_client()
    result = _claude_structured_call(
        client, ANSWERER_SYSTEM_PROMPT,
        f"CONTEXT:\n{refrence_text}\n\nQUESTION:\n{question}",
        define_LLM_EVALUATION_SCHEMA(1), "submit_answer",
    )
    TheAIResponse = json.dumps(result)  # keep the downstream contract: a JSON string with answer/answered
    return TheAIResponse , client , CLAUDE_MODEL_NAME , refrence_text , question


def ask_CLAUDE_TO_EVALUATE_RESPONSE(response , client , model_name , refrence_text , question , expected_answer ):
    response_dict = json.loads(response)
    if expected_answer == None :
        expected_answer = "This question has no answer in the context correct = the model refused "
    answer = response_dict["answer"]
    answered = response_dict["answered"]
    try :
        result = _claude_structured_call(
            client, JUDGE_SYSTEM_PROMPT,
            build_judge_user_prompt(refrence_text, question, answered, answer, expected_answer),
            define_LLM_EVALUATION_SCHEMA(0), "submit_judgement",
        )
    except Exception as e :
        return None , None , f"JUDGE_CALL_FAIL: {e!r}" , answered
    return result["is_faithful"] , result["is_correct"] , result["reasoning"] , answered


 # since it returns a list of string not simply one string 

def enhancing_chunks(chunk,doc_text,file_path=enhancement_source_file_path ,claude_mode = False ):
    enhancment_schema = define_LLM_EVALUATION_SCHEMA(2)
    user_prompt = [
        {"type": "text", "text": f"Here is the full DOCUMENT for context:\n{doc_text[0]}", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": f"---\n\nHere is the specific CHUNK...\n{chunk['text']}\n\nWrite the blurb now."}
    ]
    
    enhancement_system_prompt = "You are given a full document and one chunk extracted from it. Write a short blurb (1-2 sentences) that situates this chunk within the document - what section or topic it is from, with any vague references (like 'this study' or 'the program') resolved using information elsewhere in the document. The purpose is to make the chunk easier to find in a search index when it actually contains the answer to a question. Use only information explicitly stated in the document - do not guess or add outside knowledge. Do not summarize or repeat the chunk's own content."

    if claude_mode == True :
        client = get_claude_client() 
        response = _claude_structured_call(client ,enhancement_system_prompt, user_prompt , enhancment_schema , "submit_chunk_context") 
    else : 
        response = openAICompatibleAPI_structured_call(enhancement_system_prompt,user_prompt)

    if claude_mode == True :
        return response["context_blurb"]
    else : 
        blurb_dict = json.loads(response)
    return blurb_dict["context_blurb"]


def openAICompatibleAPI_structured_call(system_prompt, user_prompt ):
    client  = get_client()
    the_model_name = os.getenv("LOCAL_SERVER_MODEL_NAME")
    response = client.chat.completions.create(
        model=the_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format= {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": define_LLM_EVALUATION_SCHEMA(2)
            }
        }
    )
    return response.choices[0].message.content 
