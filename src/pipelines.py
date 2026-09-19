import os
import json
from collections import Counter
import time 
import faiss
import numpy as np

from chunker import chunk, save_chunks, read_chunks
from embedder import fais_chunks_embedder, populate_index, embed_question, read_embedding_index, save_embedding_index , build_bm25_index, bm25_search
from document_loader import load_document
from config import chunk_size, overlap_size, file_paths
from evaluate import reading_the_golden_set


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


def automatic_initialization_pipeline(index_file_name, chunk_file_name):
    index = read_embedding_index(index_file_name)  # read the index from disk if it exists
    chunks = read_chunks(chunk_file_name)
    return index, chunks


# ---------------------------------------------------------------------------
# Hand-written eval set (eval_set_one.json) — one JSON object per line, e.g.:
# {"question": "...", "expected_answer": "..." or null, "is_impossible": bool,
#  "type": "fact|exact_token|multi_chunk|impossible",
#  "source_snippet": "verbatim quote" or ["quote 1", "quote 2", ...]}
# ---------------------------------------------------------------------------

def eval_set_loader(path="eval_set_one.json"):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:  # skip blank lines
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {e}") from e
    return questions


# ---------------------------------------------------------------------------
# Generation eval — shared loop: retrieve -> ask -> judge -> tally -> cross-tab.
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
