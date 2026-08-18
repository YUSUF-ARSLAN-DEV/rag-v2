import tiktoken 
encoder = tiktoken.get_encoding("cl100k_base") 

# 200 tokens per chunk & last 50 tokens can overlap from each chunk
def chunk(text , chunk_size = 200  , overlap = 50 ):
    chunks = []
    tokens = encoder.encode(text) 

    step_size = chunk_size - overlap 
    for i in range(0,len(tokens) ,step_size) : # we do not care about the value
        chunk = tokens[i:i+chunk_size ]
        chunk = encoder.decode(chunk)
        chunks.append(chunk)
    return chunks       



