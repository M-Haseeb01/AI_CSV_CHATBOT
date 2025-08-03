# =================== Required Libraries ===================
import os
from pathlib import Path
import pytesseract
import cv2
from PIL import Image
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter

# =================== Configuration ===================




from langchain_google_genai import ChatGoogleGenerativeAI
os.environ["GOOGLE_API_KEY"] = open("key.txt").read().strip()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



# =================== Image Data Extraction ===================

def extract_text_enhanced(image_path):
    try:
        print(f"📖 Processing image: {image_path}")

        
        if not Path(image_path).suffix.lower() in [".jpg", ".jpeg", ".png"]:
            raise ValueError("Not a supported image format")

        
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("Image could not be read")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        
        text = pytesseract.image_to_string(thresh)
        
        
        return text

    except Exception as e:
        print(f"Error: {e}")
        return ""
    
# =================== CSV Agent ===================

def talk_with_csv(csv_path, queries):
    if not Path(csv_path).is_file():
        print(f"CSV not found: {csv_path}")
        return

    

    agent = create_csv_agent(llm, str(csv_path), allow_dangerous_code=True)

    for q in queries:
        print(f"Querying : {q}")
        response = agent.invoke({"input": q})
        print(response["output"] if isinstance(response, dict) and "output" in response else response)
        print("=" * 50)  


# =================== Excel Agent ===================

import os
import pandas as pd
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


def talk_with_excel(excel_path, queries):
    if not Path(excel_path).is_file():
        print(f"Excel file not found: {excel_path}")
        return

    
   

    df = pd.read_excel(str(excel_path))  

    agent = create_pandas_dataframe_agent(llm, df, allow_dangerous_code=True)

    
    for q in queries:

        print(f"Querying Excel: {q}")
        response = agent.invoke({"input": q})
        print(response["output"] if isinstance(response, dict) and "output" in response else response)
        print("=" * 50)  



def chat_with_excel(df, queries):
    
    agent = create_pandas_dataframe_agent(llm, df, allow_dangerous_code=True)

    for q in queries:

        print(f"Querying: {q}")
        response = agent.invoke({"input": q})
        print(response["output"] if isinstance(response, dict) and "output" in response else response)
        print("=" * 50)  
#================== PDF Helper Function ==========

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
import os

def ask_about_pdf(pdf_path, query):
  
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()

    
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(pages)

    
    context = "\n".join(doc.page_content for doc in docs)

    
   

    prompt = f"Here is the invoice content:\n{context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    print(response.content)
    return response.content
