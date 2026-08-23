import tiktoken 
from config import * 
import os 
from config import *

encoder = tiktoken.get_encoding("cl100k_base") 

# 200 tokens per chunk & last 50 tokens can overlap from each chunk
# text is basically a list of strings where each string represents a document 

# returns a list of dictionary where each dictionary has a key "text"
#  and the value is the chunked text
def chunk(text , chunk_size = 200  , overlap = 50,test_paths = file_paths) :
    # text is a list of strings where each string represents a documentq 


    chunks = []  #  a list of dictionaries where each dictionary
                #has a key "text" and the value is the chunked text
    tokens_list  = encoder.encode_batch(text) 
    # tokens_list  stores 3 lists of tokens aka or n lists of tokens


    # the thing is this is the full file just converetd into tokens not 

    step_size = chunk_size - overlap # dynamically calculating the step size 


    # so that the overlap actually applies 
    for id , token_list  in enumerate(tokens_list) : # list is not an index but an actual list of tokens for a file 
        source = os.path.basename(file_paths[id])  # get the file name from the path
        for i in range(0,len(token_list) ,step_size) : # we do not care about the value
            chunk = token_list[i:i+chunk_size ] # numbers aka tokens
            chunk = encoder.decode(chunk) # reverts this bunch of tokens back to a string 
            chunk_dict = {
                "id":"source_"+str(id)+"chunk_"+str(i)+"__"+str(i+chunk_size),    # unique id for each chunk
                "text": chunk, 
                "start_token":i, 
                "source":source ,
                "token_count": len( token_list[i:i+chunk_size ] )  # actual slice length
            }


            chunks.append(chunk_dict)  # appending the chunk to the list of chunks
    return chunks       # a list of of lists 

# now we have chunks for various files basically batch operation 


