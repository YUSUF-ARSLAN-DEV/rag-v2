import os
import re
import json
from collections import Counter
import time 
import numpy as np

from chunker import chunk
from timing import timed , log_timings
from db import hash_text , get_user_chunks  , search_user_vectors , create_pool ,conn_string
from embedder import fais_chunks_embedder, populate_index, embed_question, build_bm25_index, bm25_search , RFF_TOP_PICKS  , reranker
from document_loader import load_document
from config import chunk_size, overlap_size, file_paths , activate_hybrid_embedding , hybdrid_embedding_top_k , activate_rerank
from evaluate import reading_the_golden_set
from model  import askQuestionToAI ,ask_AI_TO_EVALUTE_RESPONSE , ask_CLAUDE_TO_EVALUATE_RESPONSE ,askQuestionToClaude , askQuestionToLLM
import tempfile 

# ---------------------------------------------------------------------------
# Interactive RAG pipeline (ask a question about a real document you loaded)
# ---------------------------------------------------------------------------

def askQuestionToIndex(index, chunks):
    # asking the question then embedding the question
    question = input("Please write a question regarding the file that you just passed\n"
                      "make sure that what you are asking about exists in the file\n").strip()
    q_final = np.array([embed_question(question)]).astype("float32")

    _, indices = index.search(q_final, 3)  # retrieve the closest 3 chunks

    # chunks maintain the same order as when they were populated into the index,
    # so the returned index lines up with the chunk's position in the list
    text_passed_to_AI = "\n\n".join(chunks[k]["text"] for k in indices[0])
    sources = [chunks[k]["source"] for k in indices[0]]
    return [text_passed_to_AI, question, sources]  # [context, question, sources]


def manual_initialization_pipeline(file_paths, activate_hybrid=False,activate_chunk_enhancement=False ):  # chunk and embed from scratch every time
    text_string = load_document(file_paths)
    chunks = chunk(text_string, chunk_size, overlap_size, file_paths)

    if activate_chunk_enhancement : 
         t_start = time.time()
         for i in range(len(chunks)):
            t0 = time.time()
            chunks[i]["text"] = enhancing_chunks(chunks[i], text_string) + " " + chunks[i]["text"] 
            dt = time.time() - t0
            elapsed = time.time() - t_start
            print(f"[enhancement] chunk {i+1}/{len(chunks)} done in {dt:.1f}s | total {elapsed:.1f}s", flush=True)

    # bm25 is built once here (ingest time) from chunks only - querying it with an
    # actual question happens later, per-question, via bm25_search(bm25, question).
    bm25 = build_bm25_index(chunks) if activate_hybrid else None
    embeddings = fais_chunks_embedder(chunks)
    index = populate_index(embeddings)

    return index, chunks, bm25  # bm25 is None when activate_hybrid=False - always 3 values


# ---------------------------------------------------------------------------
# Hand-written eval set (eval_set_one.json) — one JSON object per line, e.g.:
# {"question": "...", "expected_answer": "..." or null, "is_impossible": bool,
#  "type": "fact|exact_token|multi_chunk|impossible",
#  "source_snippet": "verbatim quote" or ["quote 1", "quote 2", ...]}
# ---------------------------------------------------------------------------

def eval_set_loader(path="test_sets/eval_set_one.json"):
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

def run_generation_eval(ask_fn, judge_fn, label, limit=None):
    questions_embed, index, contexts, question_list, expected_answers, impossibles = reading_the_golden_set(False)
    faithful = not_faithful = correct = not_correct = 0
    question_count = 0
    rows = []

    for zindex, question in enumerate(questions_embed):
        if limit is not None and zindex >= limit:
            break

        _, indices = index.search(question.reshape(1, -1), k=1)
        # did the correct chunk get retrieved? (string fallback: SQuAD reuses paragraphs across questions)
        retrieval_hit = indices[0][0] == zindex or contexts[indices[0][0]] == contexts[zindex]
        context_piece = contexts[indices[0][0]]
        question_text = question_list[zindex]
        q_stack = [context_piece, question_text, None]

        response, client, model_name, refrence_text, question_text = ask_fn(q_stack)
        print(f"[{label}] answered question {question_count}")

        is_faithful, is_correct, reasoning, was_answered = judge_fn(
            response, client, model_name, refrence_text, question_text, expected_answers[zindex])
        if is_faithful is None or is_correct is None:
            print(f"[{label}] judge returned nothing usable:\n{reasoning}")
            continue
        print(f"[{label}] evaluated question {question_count} -> {reasoning}")

        if is_faithful:
            faithful += 1
        else:
            not_faithful += 1
        if is_correct:
            correct += 1
        else:
            not_correct += 1
        question_count += 1

        try:
            model_answer_text = json.loads(response)["answer"]
        except Exception:
            model_answer_text = response

        # rows let us cross-tabulate: does the RAG hit when it's faithful, or is it
        # unfaithful despite hitting, or unfaithful because it missed retrieval?
        rows.append({
            "question_index": zindex, "hit": retrieval_hit, "faithful": bool(is_faithful),
            "is_impossible": impossibles[zindex], "Answered": bool(was_answered),
            "question_text": question_text, "context": context_piece, "model_answer": model_answer_text,
            "expected_answer": expected_answers[zindex], "reasoning": reasoning,
        })

    print_scorecard(label, rows, faithful, not_faithful, correct, not_correct)
    return rows


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


def main():
    local = os.getenv("ASKLOCAL", "false").strip().lower() == "true"
    ask_fn = lambda q_stack: askQuestionToAI(q_stack, local)
    return run_generation_eval(ask_fn, ask_AI_TO_EVALUTE_RESPONSE, label="QWEN")


def Checking_claude(limit=None):
    # limit=N caps the run to the first N questions - keeps API spend small while testing.
    rows = run_generation_eval(askQuestionToClaude, ask_CLAUDE_TO_EVALUATE_RESPONSE, label="CLAUDE", limit=limit)
    diagnose_hallucinations(rows)
    return rows

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


def testing_chunk_sanity():
    text_list = load_document(file_paths)
    chunks = chunk(text_list)
    eval_set = eval_set_loader()
    absolute_existence = 0
    for question in eval_set:
        if snippets_hit(question["source_snippet"], [c["text"] for c in chunks]):
            absolute_existence += 1
        else:
            print("MISSING:", question["question"])
    print(f"{absolute_existence}/{len(eval_set)} snippets exist in chunks")



def build_index(file_paths):
    if not isinstance(file_paths, list):
        file_paths = [file_paths]
    text = load_document(file_paths)
    chunks = chunk(text, chunk_size, overlap_size, file_paths)
    faiss_embedded = fais_chunks_embedder(chunks)
    faiss_index = populate_index(faiss_embedded)
    bm25 = build_bm25_index(chunks)
    return faiss_index, chunks, bm25


def ask_question_phase1(question, user_id , pool , timings , eval_mode=False):
    with timed(timings , "embed_question") :
        embedded_question = np.array([embed_question(question)]).astype("float32")[0] # extracting the list

    with timed(timings , "db_read_chunks") :
        all_chunks_for_user = get_user_chunks(pool,user_id )

    with timed(timings , "bm25") :
        bm25 = build_bm25_index(all_chunks_for_user )
        bm25_indices, _ = bm25_search(bm25, question) # gets embedded inside
        bm25_top_rows = [all_chunks_for_user[i] for i in bm25_indices ]

    with timed(timings , "vector_search") :
        FAISS_TOP = search_user_vectors(pool , user_id , embedded_question,hybdrid_embedding_top_k)

    with timed(timings , "rrf_fusion") :
        hybrid_chunks = RFF_TOP_PICKS(FAISS_TOP, bm25_top_rows)

    with timed(timings , "rerank") :
        reranked = reranker(question, hybrid_chunks)
    top_chunks = [r[0] for r in reranked]

    if eval_mode :
        return top_chunks  # the eval only needs the retrieved chunks, not the LLM prompt

    evidence_text = "\n\n".join(top_chunks)
    q_stack = [evidence_text, question, None]
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

    
def answer_question( question, user_id , pool ):
    timings = {}  # seconds spent in each stage, written to the container log

    q_stack,top_chunks = ask_question_phase1(question , user_id , pool,timings  ) 
    return  ask_question_phase2(q_stack , timings,top_chunks ) # returns the JSON 
    

def evaluate_recall_at_5(user_id=999):
    # recall@5 = fraction of answerable questions whose gold snippet(s) appear in the top-5 retrieved chunks
    # runs directly (not through the API) against the SAME Postgres data and retrieval code the API uses
    question_set , _ = eval_set_loader()  # list of dicts, one per question
    pool = create_pool(conn_string)
    hits = 0
    answerable = 0
    try :
        for item in question_set :
            if item["is_impossible"] :
                continue  # impossible questions have no source chunk, so recall does not apply to them
            answerable += 1
            top_chunks = ask_question_phase1(item["question"].strip() , user_id , pool , {} , eval_mode=True)
            # snippets_hit lowercases and strips everything except letters/digits on both sides (the PDF
            # hyphenation/curly-quote fix), then checks EVERY snippet appears in at least one retrieved chunk
            if snippets_hit(item["source_snippet"] , top_chunks) :
                hits += 1
            else :
                print("MISS:" , item["question"])
    finally :
        pool.close()
    recall = hits / answerable
    print(f"recall@5: {recall*100:.1f}%  ({hits}/{answerable} answerable questions)")
    return recall


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



        