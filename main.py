import faiss
from chunker import chunk ,save_chunks  , read_chunks
import ollama 
import os 
from embedder import embed_chunks , populate_index , embed_question , read_embedding_index , save_embedding_index , save_embedding_index 
from document_loader import load_document
import numpy as np 
from config import chunk_size , overlap_size  , file_paths,  base_url
from model import get_client 









# returns [compiled string , source , q_string ]
def askQuestionToIndex(index,chunks):
    # asking the question  then embedding the question 
    question = input("Please write a question regarding the file that you just passed\n make sure that what you are asking about exists in the file\n").strip()
    q_string = question 
    q_embed = [embed_question(question)]
    q_final  = np.array(q_embed).astype("float32")  # convert the list to a numpy array and then to float32

    distances , indices =  index.search(q_final,3) # retrive the closest 3 answers 

    # chunks maintain the same order of creation as being populated in the index 
    # so the returned index is the same index as of the chunk's index  in the list 
    text_passed_to_AI = "\n\n".join (chunks[k]["text"] for k in indices[0])
    sources = [chunks[k]["source"] for k in indices[0]] # indices is a 2d array 
    return [text_passed_to_AI , sources,q_string ] 

# [retrived text , sources list , the actual question string ]

def askQuestionToAI(q_stack):
    refrence_text = q_stack[0]
    question = q_stack[-1]
    sources = q_stack[1]
    client = get_client() # end point we are gonna use to communicate 
    response = client.chat.completions.create(
        model="qwen3.5:9b",
        messages = 
        [
            {"role":"system","content":"You are a helpful assistant that answers questions based on the context provided. If the answer is not in the context, say 'I don't know'."},
            {"role": "user", "content": f"CONTEXT:\n{refrence_text }\n\nQUESTION:\n{question}"}
        ]
    )
        
 


    print("\n\nThe answer to your question is : \n\n")
    print(response.choices[0].message.content)  

    print("Sources for the answer are : \n\n")
    for source in sources:
        print(source)


def manual_initialization_pipeline(): #  chunk and embed manually every single time 
    test_paths = file_paths 

    text_string = load_document(test_paths)  # the raw strings 

    # strinsg that are now chunked int he format of tokens 
    chunks = chunk(text_string,chunk_size,overlap_size,test_paths)

    # These chunks  now become a list of vectors each with dimensions 1,384 
    embeddings = embed_chunks(chunks)

    # Creating and populating the index with the embeddings

    index = populate_index(embeddings) 
    return index ,chunks 

    
def automatic_initialization_pipeline(index_file_name , chunk_file_name):
    index = read_embedding_index(index_file_name )  # read the index from disk if it exists
    chunks = read_chunks(chunk_file_name)
    return index , chunks 

    


def main():
  # 
  '''
  index , chunk =  automatic_initialization_pipeline("test1.faiss","chunk1.json")
  if index == None or chunk == None :
    index,chunks =  manual_initialization_pipeline()
    save_chunks(chunks)
    save_embedding_index(index)  # saving the index and chunks in case that they wer enot saved from the start 

  context , sources_list , question_string = askQuestionToIndex(index,chunk)
  q_stack = [context,sources_list,question_string]
  askQuestionToAI(q_stack)
  '''

  #verifying the golden set # ntoe chunks is a list of dicts
chunks =  read_chunks("chunk1.json")

target = None 
for dic in chunks :
    if dic['id'] =='source_0chunk_10650__10850':
        target = dic

print(f"The number of existing chunkis is:\n{len(chunks)}")

print("Testinf if chunk with ID:\nsource_0chunk_10650__10850\n contains the answer to this question\nwhat is a pure function according to the book ")

if target != None :
    print(target)
else :
    print("were not able to extract dictionary")


main()




