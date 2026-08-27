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
        ]
    )

    TheAIResponse = response.choices[0].message.content
    print("\n\nThe answer to your question is : \n\n")
    print(TheAIResponse)

    if sources is not None:
        print("Sources for the answer are : \n\n")
        for source in sources:
            print(source)
    else:
        print("No Sources ")
    return TheAIResponse

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
