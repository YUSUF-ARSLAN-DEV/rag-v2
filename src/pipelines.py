import os
import re
import json
from collections import Counter
import time 
import numpy as np

from chunker import chunk
from timing import timed , log_timings
from db import hash_text , get_user_chunks  , search_user_vectors , create_pool ,conn_string , add_message 
from embedder import fais_chunks_embedder, populate_index, embed_question, build_bm25_index, bm25_search , RFF_TOP_PICKS  , reranker
from document_loader import load_document
from config import chunk_size, overlap_size, activate_hybrid_embedding , hybdrid_embedding_top_k 
from model  import askQuestionToLLM , rewrite_question
import tempfile 

# ---------------------------------------------------------------------------
# Interactive RAG pipeline (ask a question about a real document you loaded)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Hand-written eval set (eval_set_one.json) — one JSON object per line, e.g.:
# {"question": "...", "expected_answer": "..." or null, "is_impossible": bool,
#  "type": "fact|exact_token|multi_chunk|impossible",
#  "source_snippet": "verbatim quote" or ["quote 1", "quote 2", ...]}
# ---------------------------------------------------------------------------

def eval_set_loader(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_sets", "eval_set_one.json")):   # anchored on this file, not on the folder you run from
    questions = []
    n_question = 0 
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:  # skip blank lines
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {e}") from e
    return questions , len(questions)


# ---------------------------------------------------------------------------
# Generation po — shared loop: retrieve -> ask -> judge -> tally -> cross-tab.
# main() runs it against the server/local model; Checking_claude() against Claude.
# Both `ask_fn` and `judge_fn` must match askQuestionToAI / ask_AI_TO_EVALUTE_RESPONSE's
# signatures and return shapes (see model.py).
# ---------------------------------------------------------------------------


def print_scorecard(label, rows, faithful, not_faithful, correct, not_correct):
    f_total = faithful + not_faithful
    c_total = correct + not_correct
    faithfulness = (faithful / f_total * 100) if f_total else 0
    accuracy = (correct / c_total * 100) if c_total else 0

    print(f"\n===== {label} on {len(rows)} questions =====")
    print("Faithfulness = did the answer stick to only the given context (no outside knowledge)?")
    print(f"Faithfulness: {faithfulness:.1f}%")
    print("Accuracy = does the answer match the expected answer in meaning?")
    print(f"Accuracy:     {accuracy:.1f}%")
    print("(hit, faithful)          :", Counter((r["hit"], r["faithful"]) for r in rows))
    print("(is_impossible, faithful):", Counter((r["is_impossible"], r["faithful"]) for r in rows))
    print("(Answered, is_impossible):", Counter((r["Answered"], r["is_impossible"]) for r in rows))


# TEMP diagnostic: prints every impossible question the model answered anyway,
# so we can eyeball whether it's a real hallucination or a judge/label mis-evaluation.
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



def normalize_for_match(text):
    # PDF text carries hyphenation ("moni- toring"), curly quotes and markup noise that the
    # hand-typed snippets don't have, so compare on lowercase letters + digits only.
    return re.sub(r"[^a-z0-9]", "", text.lower())


def snippets_hit(snippets, texts):
    # True if EVERY snippet (str or list of str) appears in at least one of `texts`.
    if isinstance(snippets, str):
        snippets = [snippets]
    normalized_texts = [normalize_for_match(t) for t in texts]
    return all(any(normalize_for_match(s) in t for t in normalized_texts) for s in snippets)


def ask_question_phase1(question, user_id , pool , timings , eval_mode=False ,
                        use_bm25=True , use_rerank=True , top_k=hybdrid_embedding_top_k ,document_ids=None):
    # use_bm25 / use_rerank / top_k are experiment switches; the defaults = the behavior the API always had
    with timed(timings , "embed_question") :
        embedded_question = np.array([embed_question(question)]).astype("float32")[0] # extracting the list
    if not  document_ids :
        document_ids = None
    bm25_top_rows = []
    if use_bm25 :
        with timed(timings , "db_read_chunks") :
            all_chunks_for_user = get_user_chunks(pool,user_id,document_ids)

        with timed(timings , "bm25") :
            bm25 = build_bm25_index(all_chunks_for_user )
            bm25_indices, _ = bm25_search(bm25, question) # gets embedded inside
            bm25_top_rows = [all_chunks_for_user[i] for i in bm25_indices ]

    with timed(timings , "vector_search") :
        FAISS_TOP = search_user_vectors(pool , user_id , embedded_question,top_k,document_ids)

    with timed(timings , "rrf_fusion") :
        hybrid_chunks = RFF_TOP_PICKS(FAISS_TOP, bm25_top_rows , top_k)  # no BM25 rows -> just the vector order

    if use_rerank :
        with timed(timings , "rerank") :
            reranked = reranker(question, hybrid_chunks)
        top_chunks = [r[0] for r in reranked]
    else :
        top_chunks = hybrid_chunks[:5]  # no reranker: keep the fused order, take the top 5

    if eval_mode :
        return top_chunks  # the eval only needs the retrieved chunks, not the LLM prompt

    evidence_text = "\n\n".join(top_chunks)
    q_stack = [evidence_text, question]
    return q_stack ,top_chunks

def ask_question_phase2(q_stack,timings,top_chunks) :
    with timed(timings , "llm") :
        raw, _, _, _, _ = askQuestionToLLM(q_stack)  # backend picked by the LLM_BACKEND env var
        parsed = json.loads(raw)
    
    timings["total"] = round(sum(timings.values()), 3)
    log_timings("ask", timings)

    if parsed["answered"]:
        return {"answered": True, "answer": parsed["answer"], "reference": parsed["specific_refrence"], "top_chunks": top_chunks}
    return {"answered": False, "answer": "The model could not answer from the provided documents.", "reference": None, "top_chunks": top_chunks}

    
def answer_question( question, user_id , pool ,documents_id=None, history = None):
    timings = {}  # seconds spent in each stage, written to the container log
    with timed(timings , "rewrite"):
        search_question = rewrite_question(history , question)

    q_stack,top_chunks = ask_question_phase1(search_question , user_id , pool,timings ,document_ids =documents_id ) 
  
    
    return ask_question_phase2(q_stack , timings,top_chunks ) # returns the JSON 
    
    

def evaluate_recall_at_5(user_id=6 , **retrieval_switches):  # switches: use_bm25, use_rerank, top_k
    # recall@5 = fraction of answerable questions whose gold snippet(s) appear in the top-5 retrieved chunks
    # runs directly (not through the API) against the SAME Postgres data and retrieval code the API uses
    question_set , _ = eval_set_loader()  # list of dicts, one per question
    pool = create_pool(conn_string)
    hits = 0
    missed_questions = [] 
    answerable = 0
    try :
        for item in question_set :
            if item["is_impossible"] :
                continue  # impossible questions have no source chunk, so recall does not apply to them
            answerable += 1
            top_chunks = ask_question_phase1(item["question"].strip() , user_id , pool , {} , eval_mode=True , **retrieval_switches)
            # snippets_hit lowercases and strips everything except letters/digits on both sides (the PDF
            # hyphenation/curly-quote fix), then checks EVERY snippet appears in at least one retrieved chunk
            if snippets_hit(item["source_snippet"] , top_chunks) :
                hits += 1
            else :
                missed_questions.append(item["question"])
    finally :
        pool.close()
    recall_at_5 = hits / answerable
    results  = {"recall":recall_at_5 , "hits":hits , "answerable_question_count":answerable , "missed_questions":missed_questions }
    return results


def processing_file_uploads(contents:bytes , file_name:str , user_id , pool ):
    # saves one uploaded document (chunks + vectors) in Postgres, always returns a dict
    # a connection is borrowed from the pool ONLY while we touch the database, not during the slow chunking/embedding
    timings = {}  # seconds spent in each stage, written to the container log
    suffix = os.path.splitext(file_name)[1]
    tmp_path = None
    try :
        # saving the bytes as a temporary file so the loaders can read it
        with tempfile.NamedTemporaryFile(delete=False , suffix = suffix ) as tmp :
            tmp.write(contents)
            tmp_path = tmp.name

        with timed(timings , "load_and_hash") :
            strings_list = load_document([tmp_path])
            content_hash = hash_text(strings_list)

        # same user + same content = already uploaded, skip the slow chunking/embedding
        with timed(timings , "duplicate_check") :
            with pool.connection() as conn :
                existing = conn.execute("SELECT 1 FROM documents WHERE user_id = %s AND content_hash = %s",(user_id , content_hash)).fetchone()
        if existing :
            log_timings("upload (skipped duplicate)", timings)
            return {"status":"Skipped" , "message":"This file was already uploaded."}

        with timed(timings , "chunking") :
            chunks = chunk(strings_list , user_id)
        if not chunks :
            return {"status":"Failed" , "message":"No text could be extracted from this file."}

        with timed(timings , "embedding") :
            vectors = fais_chunks_embedder(chunks)  # the document vectorized

        # one transaction: the document row + all its chunks, saved together or not at all
        with timed(timings , "db_insert") :
            with pool.connection() as conn :
                doc_id = conn.execute("INSERT INTO documents (user_id,filename,content_hash) VALUES (%s,%s,%s) RETURNING id ",(user_id , file_name , content_hash)).fetchone()[0]
                rows = [(user_id , doc_id , c["text"] , v) for c , v in zip(chunks , vectors)]
                sql_statement = "INSERT INTO chunks (user_id,document_id,chunk_text,chunk_vector) VALUES (%s,%s,%s,%s)"
                with conn.cursor() as cur :
                    cur.executemany(sql_statement , rows)
        # leaving the with-block commits (or rolls back on an exception) and returns the connection to the pool

        timings["total"] = round(sum(timings.values()), 3)
        log_timings("upload", timings)
        return {"status":"Successful" , "chunks_stored":len(rows)}
    except Exception as e :
        return {"status":"Failed" , "message":str(e)}
    finally :
        if tmp_path and os.path.exists(tmp_path) :
            os.remove(tmp_path)



        