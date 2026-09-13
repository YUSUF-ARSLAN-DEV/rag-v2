import numpy as np 
from pipelines import main, Checking_claude , manual_initialization_pipeline , eval_set_loader
from config import file_paths 
from embedder import embed_question

if __name__ == "__main__":
    # main()
    index , chunks = manual_initialization_pipeline(file_paths) # returns a populated index and a list of chunks 
    list_of_questions_and_answers = eval_set_loader() 
   
    total_5 = 0 
    hitat5 = 0 
    for  dictionary in list_of_questions_and_answers : 
        question_string = dictionary["question"]
        print(question_string)
        embedded_question = np.array(embed_question(question_string)).reshape(1, -1)
        _  , indices = index.search(embedded_question,k=5)

        retrieved_chunks = [chunks[i]["text"] for i in indices[0]]

        if type(dictionary["source_snippet"]) == list :
            target_hits = len(dictionary["source_snippet"]) 
            local_hits = 0 
            for snippet in dictionary["source_snippet"]:
                if any(snippet in s for s in retrieved_chunks ) :
                    local_hits +=1 
                if local_hits == target_hits :
                    hitat5+=1 
                    break 
                     
        else :
            if any(dictionary["source_snippet"] in s for s in retrieved_chunks ) :
                print(dictionary["source_snippet"])
                print(retrieved_chunks[0:-1])
                break 
                hitat5 +=1 
        total_5 +=1 

    print(f"This is the hit at 5 rate:{(hitat5/total_5)*100} %")
            

    