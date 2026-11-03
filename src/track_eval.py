import mlflow
import os 
import argparse 
import numpy as np 
from pipelines import evaluate_recall_at_5
from config import (chunk_size, overlap_size, c_rff_value, index_dimension)

mlflow.set_experiment("rag-retrieval-5") # this is us creating an experiement in ml flow


# creating an argument parser object - an object that reads arguments that you pass from the command line 
parser = argparse.ArgumentParser() 
parser.add_argument("--user-id", type=int , required=True )
parser.add_argument("--gate" , action="store_true")
args= parser.parse_args() # reading the value
# the grid: each dict is one run. run from the repo root: python src/track_eval.py
configs = [
    {"name": "a_vector_only",       "use_bm25": False, "use_rerank": False, "top_k": 20},
    {"name": "b_hybrid_no_rerank",  "use_bm25": True,  "use_rerank": False, "top_k": 20},
    {"name": "c_hybrid_rerank_k20", "use_bm25": True,  "use_rerank": True,  "top_k": 20},
    {"name": "d_hybrid_rerank_k10", "use_bm25": True,  "use_rerank": True,  "top_k": 10},
    {"name": "e_hybrid_rerank_k40", "use_bm25": True,  "use_rerank": True,  "top_k": 40},
]
to_run =  [] 
if args.gate :
    for o in configs: # itearting the dictionaries
        if o["name"] == "c_hybrid_rerank_k20" : 
            to_run.append(o)
else : 
    to_run = configs 
    
results_for_all = {}
for cfg in to_run  :
    switches = {k: v for k, v in cfg.items() if k != "name"}
    with mlflow.start_run(run_name=cfg["name"]) :
        results = evaluate_recall_at_5(args.user_id , **switches)
        mlflow.log_params({**switches,
                           "chunk_size":chunk_size,
                           "overlap_size":overlap_size,
                           "c_rff_value":c_rff_value,
                           "index_dimension":index_dimension})
        # metrics are values that are measured by the run

        mlflow.log_metrics({"recall_at_5":results["recall"],
                            "hits":results["hits"],
                            "answerable_question_count":results["answerable_question_count"]})
        mlflow.log_dict({"missed_questions":results["missed_questions"]},"missed_questions.json")
    results_for_all[cfg["name"]] = results 



if args.gate : 




    

    #_____________________________ recording the summary in the step file 
    summary = os.getenv("GITHUB_STEP_SUMMARY") # getting the path for the step file 
    c = results_for_all["c_hybrid_rerank_k20"]
        #__ Answqering the question : if we picked a different set of 44 questiosn how different would the results have been - sampling dstribution or sampling Error 
        # 1- converting the sequence of resutls into a binary list of 1s and 0s for every hit and miss 
    correct =np.array( [1 for i in range(0,c["hits"])] ) 
    wrong =np.array( [0 for i in range(0, len(c["missed_questions"]))] ) 
    full_list = np.concatenate([correct,wrong])

    rng = np.random.default_rng(42)
    all_samples = []
    for i in range (0,10000) : 
        sample = rng.choice(full_list , size=len(full_list) , replace=True )
        all_samples.append(sample.mean())
    interval = np.percentile(all_samples , [2.5, 97.5])
    line = (
        f"recall@5 (c_hybrid_rerank_k20): {c['recall']*100:.1f}% "
        f"({c['hits']}/{c['answerable_question_count']})\n\n"
        f"95% bootstrap interval: [{interval[0]*100:.1f}%, {interval[1]*100:.1f}%]\n\n"
        f"Only 2.5% of resamples scored below {interval[0]*100:.1f}%, "
        f"and only 2.5% scored above {interval[1]*100:.1f}%."
    )
    
    if summary : # meaning if there is actual a file for that stpe 
        with open (summary , "a"  , encoding="utf-8") as f : # then we actually append a summary to that file 
            f.write(f'### RETRIEVAL EVAL\n\n{line}\n')


if args.gate and results_for_all["c_hybrid_rerank_k20"]["recall"]  < 0.65   :
    raise SystemExit("The recall_at_5 fell below the required threshold ")
