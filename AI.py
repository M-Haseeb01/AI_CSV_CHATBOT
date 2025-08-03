import pytesseract
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import pytesseract
import os
import os
from pathlib import Path
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from tools import extract_text_enhanced,talk_with_csv,talk_with_excel,ask_about_pdf
#=================================#
q=[]
queries = []

i=0

print("Uploading Files.....")
print(
    "\n📄 Welcome to the AI Invoice Assistant!\n"
    "I'm here to help you analyze your invoices.\n"
    "Please choose a file type to begin:\n"
    
    )
while i<=3:
    
    choice=''
    choice = input(

    " - Type 'IMG' for image-based invoices\n"
    " - Type 'CSV' for CSV files\n"
    " - Type 'PDF' for CSV files\n"
    " - Type 'Excel' for Excel spreadsheets\n\n> "
    )
    
    if choice=="IMG":
        
        image_file_path_str = r"F:\STORAGE DOWNLOADS\NEW Storage\Projects\AI AGENTS\INVOICE\Screenshot csv.png" 
        image_path = Path(image_file_path_str).resolve()
        t = extract_text_enhanced(image_path)
        with open('key.txt','r') as f:
            key=f.readline().strip()
        
        from langchain_google_genai import ChatGoogleGenerativeAI

        import os

        os.environ["GOOGLE_API_KEY"] = open("key.txt").read().strip()

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        q1=input('ASK:')
        q1+=f'from{t}'
        response=llm.invoke([
            HumanMessage(content=q1)
        ])
        
        print(response.content)
        print("=" * 20)

        
    elif choice=='CSV':
        s=input('ASK:')
        queries.append(s)
        csv_file_path_str = r"F:\STORAGE DOWNLOADS\NEW Storage\Projects\AI AGENTS\invoices.csv"
        csv_path = Path(csv_file_path_str).resolve()
        talk_with_csv(csv_path,queries)
        
    elif choice=='Excel':
        s=input('ASK:')
        q.append(s)
        csv_file_path_str = r"F:\STORAGE DOWNLOADS\NEW Storage\Projects\AI AGENTS\INVOICE\invoices.xlsx"
        csv_path = Path(csv_file_path_str).resolve()
        talk_with_excel(csv_file_path_str,q)
        
    elif choice=='PDF':
        pdf_path = r'F:\STORAGE DOWNLOADS\NEW Storage\Projects\AI AGENTS\INVOICE\Sample Vendor Invoice.pdf'
        c=input("ASK:")
        ask_about_pdf(pdf_path,c)
    else:
        print("None selected")
    i=i+1
        


