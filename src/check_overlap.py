import json
import os
import re
import sys

from document_loader import load_document

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(HERE, "test_sets", "wf.pdf")
DRAFT_PATH = os.path.join(HERE, "test_sets", "eval_set_draft.json")
N = 3  # a question sharing this many consecutive words with the paper is flagged


def words(text):
    # lowercase, punctuation removed, split on whitespace
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def ngrams(word_list, n):
    return {tuple(word_list[i:i + n]) for i in range(len(word_list) - n + 1)}


def letters_and_digits(text):
    # same normalisation as normalize_for_match in pipelines.py (kept here so this script doesn't load the models)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def load_draft(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(pdf_path=PDF_PATH, draft_path=DRAFT_PATH):
    pdf_text = load_document([pdf_path])[0]
    pdf_grams = ngrams(words(pdf_text), N)
    pdf_normalized = letters_and_digits(pdf_text)
    questions = load_draft(draft_path)

    problems = 0
    for number, item in enumerate(questions, start=1):
        shared = ngrams(words(item["question"]), N) & pdf_grams
        if shared:
            problems += 1
            phrases = sorted(" ".join(g) for g in shared)
            print(f"Q{number} OVERLAP: {item['question']}\n    shared {N}-word phrases: {phrases}")

        snippets = item["source_snippet"]
        if isinstance(snippets, str):
            snippets = [snippets]
        for snippet in snippets:
            if snippet not in pdf_text:
                problems += 1
                exact_note = "found after normalising" if letters_and_digits(snippet) in pdf_normalized else "NOT FOUND at all"
                print(f"Q{number} SNIPPET not an exact copy ({exact_note}): {snippet[:90]}...")

    print(f"\n{len(questions)} questions checked, {problems} problem(s)")
    return problems


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
