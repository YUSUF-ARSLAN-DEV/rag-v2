chunk_size=200  # measured in tokens 
overlap_size =50  # measured in tokens
model_name = "qwen3.5:9b"
from pathlib import Path 
base_url="http://localhost:11434/v1"
file_paths = [
    r"C:\Users\aonli\Desktop\2nd year sem 2\INTERNSHIP WORK\2509 First Internship Briefing\internship paper mangament\2-02 Content Guidelines for Industrial Placement Report.pdf",
    r"C:\Users\aonli\Desktop\2nd year sem 2\INTERNSHIP WORK\2509 First Internship Briefing\internship paper mangament\2-06 Internship Company Supervisor Assessment Form.docx",
    r"C:\Users\aonli\Desktop\2nd year sem 2\INTERNSHIP WORK\2509 First Internship Briefing\internship paper mangament\2-07 Internship Mentor Visit Report.docx",
    r"C:\Users\aonli\Desktop\2nd year sem 2\INTERNSHIP WORK\2509 First Internship Briefing\internship paper mangament\01-3 Indemnity Letter.docx",
    r"C:\Users\aonli\Desktop\2nd year sem 2\INTERNSHIP WORK\2509 First Internship Briefing\internship paper mangament\Industry Project Request -Template.docx",
]

activate_hybrid_embedding = True 
hybdrid_embedding_top_k =  20
c_rff_value = 60
activate_rerank = True  
index_dimension = 384


BASE_PATH =Path(__file__).resolve().parent.parent # this finds this file parent directory parent

chunk_save_path = BASE_PATH/"INDEX_STATE"/"all_chunks.json"
shared_index_path =BASE_PATH/"VECTOR_EMBEDDING"/"shared.faiss"
USER_CHUNK_MAPPING=BASE_PATH/"INDEX_STATE"/"user_chunk_mapping.json"
# keep these here for now to save time 
# r"C:\Users\aonli\Desktop\Books That I am Reading- IN SHA ALLAH\Programming Books\Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf",
#r"C:\Users\aonli\Desktop\Books That I am Reading- IN SHA ALLAH\Programming Books\validationAlgorithms.pdf",

enhancement_source_file_path = r"C:\Users\aonli\Desktop\INTERNSHIP PROJECTS\Independent Learning\RAG LEARNING\rag-v2\sample_sources\healthcare-sample-research-paper.pdf"