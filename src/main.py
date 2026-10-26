from pipelines import evaluate_recall_at_5

if __name__ == "__main__":
    evaluate_recall_at_5(6)  # retrieval eval against Postgres, user_id 6 = eval_user, who owns the eval document
