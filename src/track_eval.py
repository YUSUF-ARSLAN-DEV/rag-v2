import mlflow
from pipelines import evaluate_recall_at_5
from config import (chunk_size, overlap_size, c_rff_value, index_dimension)

mlflow.set_experiment("rag-retrieval") # this is us creating an experiement in ml flow

# the grid: each dict is one run. run from the repo root: python src/track_eval.py
configs = [
    {"name": "a_vector_only",       "use_bm25": False, "use_rerank": False, "top_k": 20},
    {"name": "b_hybrid_no_rerank",  "use_bm25": True,  "use_rerank": False, "top_k": 20},
    {"name": "c_hybrid_rerank_k20", "use_bm25": True,  "use_rerank": True,  "top_k": 20},
    {"name": "d_hybrid_rerank_k10", "use_bm25": True,  "use_rerank": True,  "top_k": 10},
    {"name": "e_hybrid_rerank_k40", "use_bm25": True,  "use_rerank": True,  "top_k": 40},
]

for cfg in configs :
    switches = {k: v for k, v in cfg.items() if k != "name"}
    with mlflow.start_run(run_name=cfg["name"]) :
        results = evaluate_recall_at_5(99, **switches)
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
