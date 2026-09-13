from pypdf import PdfReader
import os
import re
from docx import Document

def read_txt(filepath): 
    with open (filepath,"r",encoding= "utf-8") as file :
        return file.read()  


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text is not None:  # Check if text extraction was successful
            # collapse PDF line-wrap newlines/whitespace runs to a single space, so
            # extracted text matches plainly-transcribed quotes (e.g. eval set snippets)
            text += re.sub(r'\s+', ' ', page_text) + ' '

    return text

def read_docx(file_path):
    doc = Document(file_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n" 
    return text
       

# now we can pass multiple files 
def load_document(file_path ):
    # list of strings 
    strings = [] 
    if not  (len(file_path)   == 0 ) and  file_path[0] ==  None :
        file_path = input("please paste your file path here").strip()
        # you can batch upload files 
    elif  (len(file_path) == 0 ) : 
        file_path = input("please paste your file path here").strip()   

    # checking for 0 or 1 path 
    for file_path in file_path : 
        if not os.path.isfile(file_path) : 
            raise FileNotFoundError(f"The file {file_path} does not exist.")        
        # just makesing sure it is not empty 
        if os.path.isfile(file_path) : 
            ext = os.path.splitext(file_path)[1].lower() 
            if ext == ".txt" : 
                strings.append(read_txt(file_path)) 
            elif ext == ".pdf" : 
                strings.append(read_pdf(file_path)) 
            elif ext == ".docx" : 
                strings.append(read_docx(file_path)) 
            else : 
                raise ValueError("Unsupported file format. Please provide a .txt, .pdf, or .docx file.")
        else : 
            raise FileNotFoundError(f"The file {file_path} does not exist.")        
        # just makesing sure it is not empty 
    return strings 