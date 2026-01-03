import fitz  # PyMuPDF
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import base64

def extract_documents_from_pdf(pdf_path, source_name="강의자료"):
    """
    텍스트가 없으면 Gemini Vision을 사용하여 OCR을 수행하는 배포용 로직
    """
    docs = []
    # OCR을 위해 Gemini 모델 호출 (이미지 분석용)
    vision_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            
            # 1. 텍스트가 없는 '이미지 페이지'인 경우 Gemini에게 시킴
            if not text:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 화질을 2배로 높임
                img_data = pix.tobytes("png")
                base64_image = base64.b64encode(img_data).decode('utf-8')
                
                # Gemini에게 이미지와 함께 프롬프트 전달
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": "이 이미지에 포함된 모든 텍스트를 한글자도 빠짐없이 추출해서 텍스트만 출력해줘."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                )
                response = vision_model.invoke([message])
                text = response.content
            
            metadata = {"source": source_name, "page": page_num}
            docs.append(Document(page_content=text, metadata=metadata))
            
    return docs

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    return text_splitter.split_documents(documents)