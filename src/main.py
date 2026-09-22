import numpy as np
from pipelines import main, Checking_claude , manual_initialization_pipeline , eval_set_loader , testing_chunk_sanity , snippets_hit ,question_pipeline 
from config import file_paths , activate_hybrid_embedding , hybdrid_embedding_top_k  , enhancement_source_file_path , activate_rerank 
from embedder import embed_question , bm25_search  , RFF_TOP_PICKS , reranker
from document_loader import load_document, pick_files 
from model import enhancing_chunks
import os 
if __name__ == "__main__":
    # main() # enhancing the chunks
    paths, question = pick_files()
    if paths:
        question_pipeline(question, paths)
  
