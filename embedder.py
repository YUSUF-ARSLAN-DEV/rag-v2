import os

from sentence_transformers  import SentenceTransformer 
import faiss 
 
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks) :
    # flattening the 2d lists of  tokens 
    text_chunks = [c["text"] for c in chunks ] # extract the text from the chunks
    list_of_vectors = model.encode(text_chunks,show_progress_bar=True) # returns a list of vectors
    return list_of_vectors # returns n rows , 384 columsn 

def embed_question(question) :
    question_vector = model.encode(question,show_progress_bar=True) 
    return question_vector 

def populate_index(twodarray): # this method  returns a populated faiss index 

    d = twodarray.shape[1] # the dimensions of the vector # how many columns 
    index = faiss.IndexFlatL2(d) # 384 dimensions  - aka d 
    twodarray = twodarray.astype("float32")
    index.add(twodarray) 
    return index  # now we have a populated index 


def save_embedding_index(index, file_path=None ) : 
    if file_path is None:
        print("Please provide a file path to save the index.")
        file_path = input("please paste your file path here").strip()
    final = "embedding_indices/" + file_path
    clean = final.strip('"')  # Remove any surrounding quotes
    if not os.path.isfile(clean) :
        os.makedirs(os.path.dirname(clean), exist_ok=True)
        faiss.write_index(index,clean)
        print(f"Index saved successfully at {clean}")
    else :
        faiss.write_index(index,clean)

    

def read_embedding_index(file_path=None) : 

    if file_path is None:
        print("Please provide a file path to read the index.")
        file_path = input("please paste your file path here").strip()
    final = "embedding_indices/"+file_path
    clean = final.strip('"')  # Remove any surrounding quotes
    if not os.path.isfile(clean) :
        try:
        
            raise FileNotFoundError(f"The file {clean  } does not exist.")
        
        except FileNotFoundError as e:
            print(e)
            return None 
        finally: 
            print("Please make sure to save the index first before trying to read it.")
    index = faiss.read_index(clean)
    return index    

