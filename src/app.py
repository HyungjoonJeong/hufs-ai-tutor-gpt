import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# 1. 핵심 설계도 (Core)
from langchain_core.prompts import PromptTemplate

# 2. 구글 AI 연결 (Google GenAI)
# 기존 Gemini 임포트 삭제 또는 주석 처리
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# OpenAI 임포트 추가
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 3. 데이터베이스 (Community)
from langchain_community.vectorstores import FAISS  # Chroma 대신 FAISS가 있는지 확인

# 4. 메모리 (가장 안전한 최신 경로로 변경)
# 만약 여기서 에러가 나면 from langchain_community.chat_message_histories import ... 로 선회해야 합니다.
#try:
#    from langchain.memory import ConversationBufferMemory
#except ImportError:
#    from langchain_community.memory import ConversationBufferMemory

# 5. 검색 엔진 (Classic)
from langchain_classic.chains import RetrievalQA

# 6. 직접 만든 모듈
from extract_text import extract_documents_from_pdf, split_documents

# --------------------------------
# 기본 설정
# --------------------------------
load_dotenv()

st.set_page_config(
    page_title="HUFS AI Tutor",
    layout="wide"
)

st.title("HUFS RAG 기반 AI 튜터(GPT)")
st.caption("강의 자료 기반으로 답변하며 출처를 명확히 제시합니다.")

# --------------------------------
# 세션 상태
# --------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --------------------------------
# 질문 분류기
# --------------------------------
def classify_question(question: str) -> str:
    # 수정 전: llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm = ChatOpenAI(model="gpt-5.2", temperature=0.3) # gpt-4o-mini는 속도가 빠르고 저렴합니다.


    prompt = f"""
다음 질문을 유형으로 분류하라.
아래 중 하나만 출력하라.

- concept
- calculation
- summary

질문:
{question}
"""
    result = llm.invoke(prompt)
    return result.content.strip()

# --------------------------------
# 계산 문제 전용 체인
# --------------------------------
def run_calculation_chain(question: str):
    # 수정 전: llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm = ChatOpenAI(model="gpt-5.2", temperature=0) # gpt-4o-mini는 속도가 빠르고 저렴합니다.

    docs = st.session_state.vector_db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    template = """
너는 대학 과목 계산 문제를 푸는 조교이다.

[규칙]
1. 풀이 과정을 단계별로 번호를 매겨 설명하라.
2. 수식을 명확히 제시하라.
3. 마지막에 최종 답을 정리하라.
4. 문맥에 없는 정보는 사용하지 마라.

[문맥]
{context}

[문제]
{question}

[풀이]
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    response = llm.invoke(
        prompt.format(
            context=context,
            question=question
        )
    )

    return response.content, docs

# --------------------------------
# 일반 RAG 체인
# --------------------------------
def run_rag(question: str, answer_style: str):
    # 수정 전: llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm = ChatOpenAI(model="gpt-5.2", temperature=0) # gpt-4o-mini는 속도가 빠르고 저렴합니다.

    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)

    context = "\n\n".join([d.page_content for d in docs])

    chat_history = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
    )

    
    # 자세히 설명할 때의 지침을 훨씬 더 구체화합니다.
    if answer_style == "자세히":
        detail_instruction = """
        1. 대학생에게 강의하는 친절한 튜터처럼 설명하라.
        2. 주요 개념은 전문 용어와 함께 쉬운 풀이를 병행하라.
        3. 답변의 구조를 '개요 - 상세 설명 - 요약' 순서로 구성하라.
        4. 문맥에 풍부한 내용이 있다면 최대한 상세히 인용하라.
        5. 가독성을 위해 불렛 포인트(*)나 번호를 적극 사용하라.
        """
    else:
        detail_instruction = "핵심만 3줄 이내로 간결하게 요약하라."

    prompt = f"""
너는 한국외국어대학교(HUFS)의 실력 있는 AI 튜터이다. 
제공된 [문맥]을 바탕으로 학생의 질문에 답변하라.

[지시사항]
{detail_instruction}

[이전 대화]
{chat_history}

[문맥]
{context}

[학생의 질문]
{question}

답변 (친절하고 상세하게):
"""
    
    
    length_instruction = (
        "핵심만 간결하게 답하라."
        if answer_style == "짧게"
        else
        "초보자도 이해할 수 있도록 자세히 설명하라."
    )

    prompt = f"""
너는 대학 강의를 돕는 AI 튜터이다.

[규칙]
1. 반드시 문맥에 근거해 답하라.
2. 없는 내용은 추측하지 마라.
3. 마지막에 참고 자료를 명시하라.
4. {length_instruction}

[이전 대화]
{chat_history}

[문맥]
{context}

[질문]
{question}

답변:
"""

    response = llm.invoke(prompt)
    return response.content, docs


# --------------------------------
# 사이드바
# --------------------------------
with st.sidebar:
    st.header("설정")

    answer_style = st.radio(
        "답변 길이",
        ["짧게", "자세히"],
        index=1
    )

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    uploaded_files = st.file_uploader(
        "PDF 업로드",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.button("학습 시작"):
        with st.spinner("자료 분석 중..."):
            all_docs = []

            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name

                docs = extract_documents_from_pdf(
                    tmp_path,
                    source_name=file.name
                )
                all_docs.extend(docs)
                os.remove(tmp_path)

            chunks = split_documents(all_docs)

            # 수정 전: embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # 가성비와 성능이 가장 좋은 모델


            # 기존 코드 (에러 발생)
            # st.session_state.vector_db = Chroma.from_documents(...)

            # 수정 코드
            st.session_state.vector_db = FAISS.from_documents(
                chunks, embedding=embeddings
)

            st.success("학습 완료")

# --------------------------------
# 채팅 UI
# --------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("질문을 입력하세요"):
    if st.session_state.vector_db is None:
        st.warning("먼저 PDF를 학습시켜주세요.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        q_type = classify_question(question)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                if q_type == "calculation":
                    answer, sources = run_calculation_chain(question)
                else:
                    answer, sources = run_rag(question, answer_style)


                refs = set()
                for d in sources:
                    refs.add(
                        f"- {d.metadata['source']} p.{d.metadata['page'] + 1}"
                    )

                final_answer = (
                    f"{answer}\n\n---\n"
                    f"참고 자료:\n" + "\n".join(sorted(refs))
                )

                st.markdown(final_answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer}
                )
