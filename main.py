import faiss
import time 
from collections import Counter 
from chunker import chunk ,save_chunks  , read_chunks
import ollama 
from  sentence_transformers import SentenceTransformer
import os 
import json 
from embedder import embed_chunks , populate_index , embed_question , read_embedding_index , save_embedding_index , save_embedding_index 
from document_loader import load_document
import numpy as np 
from config import chunk_size , overlap_size  , file_paths,  base_url
from model import get_client  , askQuestionToAI , ask_AI_TO_EVALUTE_RESPONSE , askQuestionToClaude , ask_CLAUDE_TO_EVALUATE_RESPONSE
from datasets import load_dataset 
from evaluate import evaluating_embedding_model ,reading_the_golden_set









# returns [compiled string , source , q_string ]
def askQuestionToIndex(index,chunks):
    # asking the question  then embedding the question 
    question = input("Please write a question regarding the file that you just passed\n make sure that what you are asking about exists in the file\n").strip()
    q_string = question 
    q_embed = [embed_question(question)]
    q_final  = np.array(q_embed).astype("float32")  # convert the list to a numpy array and then to float32

    distances , indices =  index.search(q_final,3) # retrive the closest 3 answers 

    # chunks maintain the same order of creation as being populated in the index 
    # so the returned index is the same index as of the chunk's index  in the list 
    text_passed_to_AI = "\n\n".join (chunks[k]["text"] for k in indices[0])
    sources = [chunks[k]["source"] for k in indices[0]] # indices is a 2d array 
    return [text_passed_to_AI , q_string,sources ] 

# [retrived text , sources list , the actual question string ]

# requires text_passed_to_AI;context
# requires source


def manual_initialization_pipeline(): #  chunk and embed manually every single time 
    
    test_paths = file_paths 
    text_string = load_document(test_paths)  # the raw strings 

    # strinsg that are now chunked int he format of tokens 
    chunks = chunk(text_string,chunk_size,overlap_size,test_paths)

    # These chunks  now become a list of vectors each with dimensions 1,384 
    embeddings = embed_chunks(chunks)

    # Creating and populating the index with the embeddings

    index = populate_index(embeddings) 
    return index ,chunks 

    
def automatic_initialization_pipeline(index_file_name , chunk_file_name):
    index = read_embedding_index(index_file_name )  # read the index from disk if it exists
    chunks = read_chunks(chunk_file_name)
    return index , chunks 

    





def main():
    questions_embed , index , contexts , question_list ,expected_answers , impossibles   = reading_the_golden_set(False)
    faithful = not_faithful = correct = not_correct = 0
    question_count = 0 
    rows = [] 
    for  zindex ,question in enumerate(questions_embed) :
       
       _,indices =  index.search(question.reshape(1,-1) , k=1)  # searching the index 
       retrieval_hit = indices[0][0] == zindex or contexts[indices[0][0]] == contexts[zindex ] # checking if the correct chunk mapping to the question was retrived 
       context_piece = contexts[indices[0][0]]
       question_deencoded = question_list[zindex] # decoding the question then storing it as a decoded string
       q_stack = [context_piece,question_deencoded,None]
       TheAIResponse , client , model_name , refrence_text , question =askQuestionToAI(q_stack , os.getenv("ASKLOCAL").strip().lower() =="true" )
       print(f"The AI has answered Question No: {question_count}")
       is_faithful , is_correct , reasoning , was_answered  = ask_AI_TO_EVALUTE_RESPONSE(TheAIResponse,client,model_name , refrence_text,question ,expected_answers[zindex])
       if is_faithful == None or is_correct == None :
           print("The Model has returned an empty or truncated resposne here is a part of it:\n{reasoning}")
           continue 
       #print(f"The AI has evaluated the answer of Question No: {question_count}") - diagnostic lines 
       #print(f"Here is the reasoning behind it: {reasoning}") - - diagnostic lines  
       if bool(is_faithful) == True :
           faithful  +=1 
       else : 
            not_faithful +=1 
       if bool (is_correct) == True :
           correct  += 1 
       else :
           not_correct +=1 

       question_count +=1
       try :
           model_answer_text = json.loads(TheAIResponse)["answer"]
       except Exception :
           model_answer_text = TheAIResponse
       rows.append({"question_index":zindex, "hit":retrieval_hit , "faithful":bool(is_faithful),"is_impossible":impossibles[zindex],"Answered":bool(was_answered ),
                    "question_text":question_deencoded , "context":context_piece , "model_answer":model_answer_text ,
                    "expected_answer":expected_answers[zindex] , "reasoning":reasoning})

    
       # rows list is there for us to check if the RAG hits when its faithful or maybe its not faithful desipte hitting 
       # or it is not faithful because it is not hitting - basically figuring out why faithfullness is lower than accuracy 


    # corss tabulation 
    c = Counter(( r["hit"], r["faithful"]) for r in rows  )
    z = Counter((r["is_impossible"] ,r["faithful"]) for r in rows  ) 
    f = Counter ( (r["Answered"],r["is_impossible"]) for r in rows )
    f_total = faithful + not_faithful
    c_total = correct + not_correct
    faithfullness = ((faithful / f_total) * 100) if f_total else 0
    accuracy      = ((correct  / c_total )* 100) if c_total else 0

    print(f"Faiithfullness Is the fact wether the model sticks to the given context when answering a given question and a specific context\nn")
    print(f"Failthfullness Percentage:\n{faithfullness }\n\n")
    print(f"Accuracy represents the rate at which the Model's Answer actually matches the correct answer when it comes to meaning Aka does the model answer correctly\n\n")
    print(f"Accuracy Rate:\n{accuracy}")
    print(c)
    print(z)
    print(f)

    #diagnose_hallucinations(rows)  # TEMP diagnostic - remove after inspection


# TEMP diagnostic method: prints every impossible question the model still answered,
# so we can eyeball whether it's a real hallucination or a judge mis-evaluation.
def diagnose_hallucinations(rows):
    hallucinated = [r for r in rows if r["is_impossible"] and r["Answered"]]
    print(f"\n\n===== {len(hallucinated)} HALLUCINATED (impossible question answered anyway) =====\n")
    for n, r in enumerate(hallucinated, 1):
        print(f"--- {n}. question_index={r['question_index']}  hit={r['hit']}  faithful={r['faithful']} ---")
        print(f"QUESTION      : {r['question_text']}")
        print(f"EXPECTED      : {r['expected_answer']}")
        print(f"MODEL ANSWER  : {r['model_answer']}")
        print(f"JUDGE REASON  : {r['reasoning']}")
        print(f"CONTEXT       :\n{r['context']}\n")


def Checking_claude(limit=None):
    # Same retrieval + evaluation loop as main(), but the answering model and the
    # judge are both claude-sonnet-5. Pass limit=N to only run the first N
    # questions (keeps API spend small while testing).
    questions_embed , index , contexts , question_list , expected_answers , impossibles = reading_the_golden_set(False)
    faithful = not_faithful = correct = not_correct = 0
    question_count = 0
    rows = []
    for zindex , question in enumerate(questions_embed):
        if limit is not None and zindex >= limit :
            break
        _, indices = index.search(question.reshape(1, -1), k=1)
        retrieval_hit = indices[0][0] == zindex or contexts[indices[0][0]] == contexts[zindex]
        context_piece = contexts[indices[0][0]]
        question_deencoded = question_list[zindex]
        q_stack = [context_piece, question_deencoded, None]

        TheAIResponse , client , model_name , refrence_text , question = askQuestionToClaude(q_stack)
        print(f"Claude has answered Question No: {question_count}")
        is_faithful , is_correct , reasoning , was_answered = ask_CLAUDE_TO_EVALUATE_RESPONSE(
            TheAIResponse, client, model_name, refrence_text, question, expected_answers[zindex])
        if is_faithful is None or is_correct is None :
            print(f"Claude judge returned nothing usable:\n{reasoning}")
            continue
        print(f"Claude has evaluated Question No: {question_count} -> {reasoning}")

        if bool(is_faithful) : faithful += 1
        else : not_faithful += 1
        if bool(is_correct) : correct += 1
        else : not_correct += 1

        question_count += 1
        try :
            model_answer_text = json.loads(TheAIResponse)["answer"]
        except Exception :
            model_answer_text = TheAIResponse
        rows.append({"question_index":zindex, "hit":retrieval_hit , "faithful":bool(is_faithful), "is_impossible":impossibles[zindex], "Answered":bool(was_answered),
                     "question_text":question_deencoded , "context":context_piece , "model_answer":model_answer_text ,
                     "expected_answer":expected_answers[zindex] , "reasoning":reasoning})

    c = Counter((r["hit"], r["faithful"]) for r in rows)
    z = Counter((r["is_impossible"], r["faithful"]) for r in rows)
    f = Counter((r["Answered"], r["is_impossible"]) for r in rows)
    f_total = faithful + not_faithful
    c_total = correct + not_correct
    faithfullness = ((faithful / f_total) * 100) if f_total else 0
    accuracy = ((correct / c_total) * 100) if c_total else 0

    print(f"\n===== CLAUDE (claude-sonnet-5) on {len(rows)} questions =====")
    print(f"Faithfulness Percentage:\n{faithfullness}\n")
    print(f"Accuracy Rate:\n{accuracy}")
    print("(hit, faithful)        :", c)
    print("(is_impossible, faithful):", z)
    print("(Answered, is_impossible):", f)
    diagnose_hallucinations(rows)

           
           

          
       



       
       

#main()
Checking_claude()  # full dataset