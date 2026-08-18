from sentence_transformers  import SentenceTransformer 
import faiss 
 
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks) :
    list_of_vectors = model.encode(chunks)
    return list_of_vectors

def populate_index(twodarray): # returns a populated faiss index 

    d = twodarray.shape[1]
    index = faiss.IndexFlatL2(d) # 384 dimensions  - aka d 
    twodarray.astype("float32")
    index.add(twodarray) 
    return index  # now we have a populated index 
