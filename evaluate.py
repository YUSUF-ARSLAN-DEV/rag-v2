from sentence_transformers import SentenceTransformer 
import faiss 
import json 

def reading_the_golden_set(skip_impossible = True ) : # if skip_impossible is False , then you keep the impossible pairs


    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open ("squad_gold.json","r", encoding = "utf_8") as f : 
        pairs = json.load(f) 

    # checkipng if we want to skip the impossibles : 
    if skip_impossible == True : 
        pairs = [pair for pair in pairs if pair["is_impossible"] == False ]
    else :
        pass 

    # cleaning the pairs 


    contexts = [pair["context"] for pair in pairs  ] # returns a list of strings 
    impossibles = [pair["is_impossible"] for pair in pairs ]

    embedded = model.encode(contexts) # reutnr a list of vectors 
    index = faiss.IndexFlatL2(embedded.shape[1])
    index.add(embedded)
    # the index have been populated with the chunks 

    question_list = [pair["question"] for pair in pairs   ] 
    # now we have a question list 
    embedded_questions = model.encode(question_list)
    expected_answers = [pair["answers"]["text"][0] if  len(pair["answers"]["text"] ) != 0     else None for pair in pairs  ]

    return embedded_questions , index , contexts ,question_list , expected_answers  , impossibles 
def evaluating_embedding_model(): 

    # reading the golden set 
    embedded_questions,index,contexts ,question_list , expected_answers  = reading_the_golden_set()
    # correct tally 
    hitat1 = 0 
    missat1 = 0 
    hitat5  = 0 
    missat5 = 0
    MRR = 0 
        # testing hit at 1 recall@k1 and recall@5k and Mean Reciprocal Rank MRR 
    for i, question in enumerate(embedded_questions):

        _, result_indices = index.search(question.reshape(1, -1), k=5)
        if contexts[result_indices[0][0]] == contexts[i]:
            hitat1 += 1
            # temp valid test
        else :
            missat1 +=1 

        if any(contexts[idx]==contexts[i] for idx in result_indices[0]):
            hitat5 += 1
            temp = result_indices[0].tolist()
            position = [position for position, indices in  enumerate(temp) if contexts[indices]==contexts[i] ]
            val_index =position[0] # getting the first val aka the first value 

            MRR += 1/ ( val_index+1)
        else : 
            missat5 +=1 
        

        # calculating the  totals
        total_recall1 = hitat1+missat1
        total_recall5 = hitat5 +missat5
    # calculating accuracy measures :  and accuracy measures  
    hitat1accuracy =f"Your Hitat1 accuracy is :\n{(hitat1/total_recall1)* 100 }% percent"
    hitat5accuracy = f"Your Hitat5 accuracy is :\n{(hitat5/total_recall5)* 100 }% percent"
    len_question = len(embedded_questions)
    LEN_QUESTIONS = f"The total number of questions that were evaluated in this test was:{len_question}"
    MRR_SCORE = f"Your MRR Mean Reciprocal Rank is: {MRR/len_question}"
    HIT_RATE = hitat5/total_recall5
    # hit rate is the percentage of questions where at least one correct chunk / vector appeared
    print(hitat1accuracy)
    print(hitat5accuracy)
    print(MRR_SCORE)
    print(LEN_QUESTIONS)


