import os

from sentence_transformers  import SentenceTransformer 
import faiss 
 
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks) :
    # flattening the 2d lists of  tokens 
    if isinstance(chunks, str):
        chunks = [[chunks]] # if i is a single string we make a list of strings
    flat_list_of_tokens = [thetoken for innerlist in chunks for thetoken in innerlist ]
    list_of_vectors = model.encode(flat_list_of_tokens)
    return list_of_vectors # returns n rows , 384 columsn 

def populate_index(twodarray): # returns a populated faiss index 

    d = twodarray.shape[1] # the dimensions of the vector # how many columns 
    index = faiss.IndexFlatL2(d) # 384 dimensions  - aka d 
    twodarray = twodarray.astype("float32")
    index.add(twodarray) 
    return index  # now we have a populated index 


def save_embedding_index(index, file_path=None ) : 
    if file_path is None:
        file_path = input("please paste your file path here").strip()
    if not os.path.isfile(file_path) :
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    else :
        faiss.write_index(index,file_path)

    return 

def read_embedding_index(file_path=None) : 
    if file_path is None:
        file_path = input("please paste your file path here").strip()
    if not os.path.isfile(file_path) :
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    else :
        index = faiss.read_index(file_path)
    return index    

