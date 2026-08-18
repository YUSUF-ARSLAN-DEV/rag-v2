from pypdf import PdfReader 
import os 
from docx import Document

def read_txt(filepath): 
    with open (filepath,"r",encoding= "utf-8") as file :
        return file.read()  


def read_pdf(file_path):
    reader = PdfReader(file_path) 
    text = '' 
    for page in reader.pages: 
        text += page.extract_text() 
    return text 

def read_docx(file_path):
    doc = Document(file_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n" 
    return text
       

def load_document(file_path ):
    if file_path == None :
        file_path = input("please paste your file path here").strip()
    file_path= file_path.strip('"')
    extension = os.path.splitext(file_path)[1].lower() 
    # we are reading the extension of the given file 
    if extension == '.txt':
        return read_txt(file_path)
    elif extension == '.pdf':
        return read_pdf(file_path)
    elif extension == '.docx':
        return read_docx(file_path)
    else: 
        raise ValueError (f"An Unsupported file tpye was passed {extension}")