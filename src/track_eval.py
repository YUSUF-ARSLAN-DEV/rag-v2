import mlflow 
from pipelines import evaluate_recall_at_5 
from config import (chunk_size, overlap_size, hybdrid_embedding_top_k, c_rff_value,
                    activate_hybrid_embedding, activate_rerank, index_dimension)
mlflow.set_experiment("rag-retrieval") # this is us creating an experiement in ml flow 
with mlflow.start_run() : 
    results = evaluate_recall_at_5(99)
    mlflow.log_params({"chunk_size":chunk_size, 
     "overlap_size":overlap_size, 
     "hybrid_on":activate_hybrid_embedding,
     "hybrid_top_k":hybdrid_embedding_top_k, 
     "c_rff_value":c_rff_value, 
      "rerank_on":activate_rerank , 
       "index_dimension":index_dimension})
    # metrics are values that are measured by the run 

    mlflow.log_metrics({"recall_at_5":results["recall"],
                        "hits":results["hits"], 
                        "answerable_question_count":results["answerable_question_count"]})

    
    mlflow.log_dict({"missed_questions":results["missed_questions"]},"missed_questions.json")