from pipelines import main, Checking_claude, evaluate_recall_at_5

if __name__ == "__main__":
    evaluate_recall_at_5(99)  # retrieval eval against Postgres, user_id 999 = the eval document
    # main()             # generation eval with the local qwen model
    # Checking_claude()  # generation eval with Claude
