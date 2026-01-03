import fitz  # PyMuPDF
import os
from langchain_openai import ChatOpenAI  # 구글 대신 OpenAI 사용
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import base64

def extract_documents_from_pdf(pdf_path, source_name="강의자료"):
    docs = []
    # 배포용 GPT-4o-mini 모델 호출 (비용이 저렴하고 시각 지능이 뛰어납니다)
    vision_model = ChatOpenAI(model="gpt-4o-mini")
    
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            
            # 1. 텍스트가 없는 '이미지 페이지'인 경우 GPT에게 OCR 요청
            if not text:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                base64_image = base64.b64encode(img_data).decode('utf-8')
                
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": "이 이미지에 포함된 모든 텍스트를 추출해서 텍스트만 출력해줘."},
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
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)