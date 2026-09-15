chunk_size=200  # measured in tokens 
overlap_size =50  # measured in tokens
model_name = "qwen3.5:9b"
base_url="http://localhost:11434/v1"
file_paths = [
    r"C:\Users\aonli\Desktop\INTERNSHIP PROJECTS\Independent Learning\RAG LEARNING\rag-v2\sample_sources\healthcare-sample-research-paper.pdf"
   ]

activate_hybrid_embedding = True 
hybdrid_embedding_top_k =  20
c_rff_value = 40

# keep these here for now to save time 
# r"C:\Users\aonli\Desktop\Books That I am Reading- IN SHA ALLAH\Programming Books\Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf",
#r"C:\Users\aonli\Desktop\Books That I am Reading- IN SHA ALLAH\Programming Books\validationAlgorithms.pdf",

enhancement_source_file_path = r"C:\Users\aonli\Desktop\INTERNSHIP PROJECTS\Independent Learning\RAG LEARNING\rag-v2\sample_sources\healthcare-sample-research-paper.pdf"