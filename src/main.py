import numpy as np
from pipelines import main, Checking_claude , manual_initialization_pipeline , eval_set_loader , testing_chunk_sanity , snippets_hit
from config import file_paths , activate_hybrid_embedding , hybdrid_embedding_top_k  , enhancement_source_file_path
from embedder import embed_question , bm25_search  , RFF_TOP_PICKS , reranker
from document_loader import load_document
from model import enhancing_chunks

if __name__ == "__main__":
    # main() # enhancing the chunks
    index , chunks , bm25 = manual_initialization_pipeline(file_paths,activate_hybrid_embedding,False ) # returns a populated FAISS index, the chunks, and a BM25 index
    list_of_questions_and_answers = eval_set_loader()
    total_5 = 0
    hitat5 = 0
    for  dictionary in list_of_questions_and_answers :
        question_string = dictionary["question"]

        embedded_question = np.array(embed_question(question_string)).reshape(1, -1)
        _  , indices = index.search(embedded_question,k=hybdrid_embedding_top_k)
        if activate_hybrid_embedding :
            top_k,_ = bm25_search(bm25 ,question_string)
            # after getting the bm25 one we start calculating the RFF
            retrieved_chunks = RFF_TOP_PICKS(indices[0],top_k,chunks)
        else : retrieved_chunks = [chunks[i]["text"] for i in indices[0]]


        # on Top of hybrid embeddings we are going to Apply reranking

        reranked = reranker(question_string,retrieved_chunks)
        filtered_chunks = [r[0] for r in reranked ]




        if snippets_hit(dictionary["source_snippet"], filtered_chunks):
            hitat5 += 1
        total_5 +=1


    print(f"hit-rate@5 (= recall@5 here):{(hitat5/total_5)*100} %")
