import os
import sys
import openai
import logging
import warnings
import uuid
import traceback
import tempfile
import json
import hashlib
from pathlib import Path
from datetime import datetime
from threading import Lock
import time
import re
import streamlit as st
from dotenv import load_dotenv
import faiss 
import tiktoken
import fitz  # PyMuPDF

from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from sentence_transformers import SentenceTransformer, util

# =========================
# 1. CONFIG 딕셔너리 정의
# =========================

CONFIG = {
    "model_name": "gpt-4o-mini",  # LLM 모델명
    "embedding_model": "text-embedding-3-large",  # 임베딩 모델명
    "embedding_size": 3072,  # 임베딩 차원 크기
    "max_token_threshold": 4000,  # 토큰 최대 한도
    "max_file_size": 20 * 1024 * 1024,  # 20MB로 제한
    "vectorstore_path": "faiss_vectorstore",  # 벡터스토어 경로
    "hash_file_path": "faiss_vectorstore/document_hashes.json",  # 문서 해시 파일 경로
    "chunk_size": 3000,  # 청크 크기 2000에서 3000으로 변경
    "chunk_overlap": 150,  # 청크 중복 크기
    "api_keys": {
        "openai": None,  # OpenAI API 키만 남김
    }
}

# =========================
# 2. 로깅 설정
# =========================

# Custom Formatter for Logging with blank lines
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # 구분선 위와 아래에 빈 줄 추가
        return f"\n--- {record.levelname} ---\n{super().format(record)}\n"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chat_server.log', encoding='utf-8')
    ]
)

# 핸들러에 커스텀 포맷터 적용
for handler in logging.root.handlers:
    handler.setFormatter(CustomFormatter())

logger = logging.getLogger(__name__)

# DeprecationWarning 경고 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================
# 3. API 키 로드
# =========================

# .env 파일에서 API 키 로드
load_dotenv()

# OpenAI API 설정
CONFIG["api_keys"]["openai"] = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# OpenAI API 키 존재 여부 확인
if not CONFIG["api_keys"]["openai"]:
    logger.error("OpenAI API 키가 설정되지 않았습니다. `.env` 파일이나 Streamlit Cloud Secrets를 확인하세요.")
    st.error("OPENAI_API_KEY가 설정되지 않았습니다.")
    st.stop()

# OpenAI API 키 설정
openai.api_key = CONFIG["api_keys"]["openai"]

# =========================
# 4. 유틸리티 함수 정의
# =========================

def extract_law_name(file_name):
    """
    파일 이름에서 법률명을 추출하는 함수
    """
    name = Path(file_name).stem
    name = re.sub(r'[_\-]', ' ', name)
    return name

def get_document_hash(doc_content: str, doc_name: str) -> str:
    """문서 내용과 이름으로 해시값 생성"""
    content = f"{doc_name}:{doc_content}".encode('utf-8')
    return hashlib.sha256(content).hexdigest()

def load_document_hashes(hash_file_path: Path) -> dict:
    """저장된 문서 해시 로드"""
    try:
        return json.loads(hash_file_path.read_text())
    except Exception as e:
        logger.error(f"해시 파일 로드 실패: {e}")
        return {}

def save_document_hashes(hash_file_path: Path, hashes: dict):
    """문서 해시 저장"""
    try:
        hash_file_path.write_text(json.dumps(hashes, ensure_ascii=False, indent=2))
    except Exception as e:
        logger.error(f"해시 파일 저장 실패: {e}")

# =========================
# 5. 문서 분할 클래스 정의
# =========================

class LawDocumentTextSplitter:
    def __init__(self, chunk_size=CONFIG["chunk_size"], chunk_overlap=CONFIG["chunk_overlap"]):
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, docs):
        law_sections = []
        for doc in docs:
            content = doc.page_content
            law_name = doc.metadata.get('law_name', '관련 법률')

            # RecursiveCharacterTextSplitter를 이용하여 텍스트 분할
            chunks = self.chunker.split_text(content)

            for chunk in chunks:
                metadata = {
                    "law_name": law_name,
                    "page": doc.metadata.get('page', 'unknown')
                }
                law_sections.append(Document(page_content=chunk, metadata=metadata))

        return law_sections

# =========================
# 6. 벡터스토어 초기화 함수
# =========================

def initialize_vectorstore(embedding_model):
    VECTORSTORE_PATH = Path(CONFIG["vectorstore_path"])
    HASH_FILE_PATH = Path(CONFIG["hash_file_path"])

    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

    if not HASH_FILE_PATH.exists():
        HASH_FILE_PATH.write_text("{}")

    embeddings = OpenAIEmbeddings(model=embedding_model)

    if 'vectorstore' in st.session_state:
        logger.info("세션에서 벡터스토어를 재사용합니다.")
        return st.session_state['vectorstore'], HASH_FILE_PATH

    vectorstore = None
    vectorstore_files_exist = (VECTORSTORE_PATH / 'index.faiss').exists() and (VECTORSTORE_PATH / 'index.pkl').exists()

    if vectorstore_files_exist:
        try:
            vectorstore = LangChainFAISS.load_local(
                str(VECTORSTORE_PATH), 
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("기존 벡터스토어를 로드했습니다.")
            logger.info(f"로드된 벡터스토어의 문서 수: {vectorstore.index.ntotal}")
        except Exception as e:
            logger.error(f"벡터스토어 로드 중 오류 발생: {e}")
            logger.info("새로운 벡터스토어를 생성합니다.")
            vectorstore = create_new_vectorstore(embeddings)
    else:
        logger.info("벡터스토어 파일이 존재하지 않습니다. 새로운 벡터스토어를 생성합니다.")
        vectorstore = create_new_vectorstore(embeddings)
        # 벡터스토어가 비어 있으므로 초기 로드를 수행합니다.
        with st.spinner("벡터스토어 초기화 중..."):
            initial_load(vectorstore)
            vectorstore.save_local(str(VECTORSTORE_PATH))

    st.session_state['vectorstore'] = vectorstore
    return vectorstore, HASH_FILE_PATH

def create_new_vectorstore(embeddings):
    embedding_size = CONFIG["embedding_size"]  # CONFIG에서 임베딩 크기 가져오기
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = LangChainFAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )
    return vectorstore

# =========================
# 7. 벡터스토어와 메모리 초기화 함수
# =========================

def initialize_vectorstore_and_memory():
    vectorstore, hash_file_path = initialize_vectorstore(CONFIG["embedding_model"])

    if 'memory' not in st.session_state:
        st.session_state['memory'] = ConversationBufferMemory(
            memory_key="chat_history",
            human_prefix="질문",
            ai_prefix="답변",
            return_messages=True,
            output_key='answer'
        )
    return vectorstore, hash_file_path

def reset_vectorstore():
    """벡터스토어와 관련 파일들을 삭제하는 함수"""
    try:
        # 벡터스토어 파일 삭제
        vectorstore_path = Path(CONFIG["vectorstore_path"])
        if (vectorstore_path / "index.faiss").exists():
            (vectorstore_path / "index.faiss").unlink()
        if (vectorstore_path / "index.pkl").exists():
            (vectorstore_path / "index.pkl").unlink()
            
        # 해시 파일 초기화
        hash_file_path = Path(CONFIG["hash_file_path"])
        if hash_file_path.exists():
            hash_file_path.write_text("{}")
            
        # 세션 스테이트 초기화
        if 'vectorstore' in st.session_state:
            del st.session_state['vectorstore']
            
        logger.info("벡터스토어가 성공적으로 리셋되었습니다.")
        return True
    except Exception as e:
        logger.error(f"벡터스토어 리셋 중 오류 발생: {e}")
        return False

# =========================
# 8. 초기 로드 시 모든 파일 처리 함수
# =========================

def load_all_pdfs_in_directory(directory_path):
    """디렉토리 내 모든 PDF 파일을 PyMuPDF를 사용하여 로드하여 하나의 문서 리스트로 반환합니다."""
    logger.info("=== 문서 로드 시작 ===")
    all_docs = []
    pdf_directory = Path(directory_path)
    pdf_files = [f for f in pdf_directory.iterdir() if f.suffix.lower() == '.pdf']

    for pdf_file in pdf_files:
        try:
            # PyMuPDF로 PDF 파일 로드
            pdf_document = fitz.open(pdf_file)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text = page.get_text()
                
                if text.strip():  # 빈 페이지가 아닌 경우만 처리
                    document = Document(
                        page_content=text,
                        metadata={
                            'law_name': extract_law_name(pdf_file.name),
                            'page': page_num + 1,
                            'source': str(pdf_file)
                        }
                    )
                    all_docs.append(document)
            
            pdf_document.close()
            logger.info(f"문서 '{pdf_file.name}' 로드 완료.")
        except Exception as e:
            logger.error(f"❌ {pdf_file.name} 처리 오류: {e}")
            logger.error(f"상세 오류: {traceback.format_exc()}")

    logger.info("=== 문서 로드 종료 ===")
    return all_docs

def initial_load(vectorstore):
    logger.info("=== 벡터스토어 초기 로드 시작 ===")
    docs = []
    pdf_directory = Path('data')
    
    # 사이드바에 진행 상황 표시
    with st.sidebar:
        st.markdown("### 📊 벡터스토어 초기화")
        progress_text = st.empty()
        progress_bar = st.progress(0)
        # 상태 메시지를 위한 컨테이너들을 리스트로 관리
        status_messages = []
    
    pdf_files = [f for f in pdf_directory.iterdir() if f.suffix.lower() == '.pdf']
    total_files = len(pdf_files)
    
    for idx, pdf_file in enumerate(pdf_files):
        try:
            # 사이드바에 진행 상황 업데이트
            with st.sidebar:
                progress_text.text(f"처리 중: {pdf_file.name}")
                progress_bar.progress((idx + 1) / total_files)
            
            # PyMuPDF를 사용하여 PDF 파일을 로드
            doc = fitz.open(pdf_file)
            content = "\n".join(page.get_text() for page in doc)
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={
                        'law_name': extract_law_name(pdf_file.name),
                        'page': 'unknown',
                        'source': str(pdf_file)
                    }
                ))
            # 새로운 상태 메시지 추가
            with st.sidebar:
                status_messages.append(f"✅ '{pdf_file.name}' 로드 완료")
                # 모든 상태 메시지를 하나의 문자열로 결합하여 표시
                st.markdown("\n".join(status_messages))
            
            logger.info(f"문서 '{pdf_file.name}' 로드 완료.")
        except Exception as e:
            # 오류 메시지 추가
            with st.sidebar:
                status_messages.append(f"❌ {pdf_file.name} 처리 오류")
                st.markdown("\n".join(status_messages))
            logger.error(f"❌ {pdf_file.name} 처리 오류: {e}")
            logger.error(f"상세 오류: {traceback.format_exc()}")

    if docs:
        with st.sidebar:
            with st.spinner("문서 분할 및 벡터화 중..."):
                text_splitter = LawDocumentTextSplitter(
                    chunk_size=CONFIG["chunk_size"], 
                    chunk_overlap=CONFIG["chunk_overlap"]
                )
                splits = text_splitter.split_documents(docs)
                vectorstore.add_documents(splits)
                VECTORSTORE_PATH = Path(CONFIG["vectorstore_path"])
                vectorstore.save_local(str(VECTORSTORE_PATH))
                
                # 최종 결과 표시
                progress_text.empty()
                progress_bar.empty()
                st.success(f"""
                ✅ 벡터스토어 초기화 완료
                - 총 문서 수: {len(docs)}개
                - 총 청크 수: {len(splits)}개
                """)

    logger.info("=== 벡터스토어 초기 로드 종료 ===")

# =========================
# 9. 파일 업로드 처리 함수
# =========================

def process_uploaded_files(uploaded_files, vectorstore, hash_file_path):
    logger.info("=== 업로드된 파일 처리 시작 ===")
    document_hashes = load_document_hashes(hash_file_path)
    skipped_files, processed_files = [], []

    # 사이드바에 진행 상황 표시
    with st.sidebar:
        upload_status = st.empty()  # 업로드 상태 섹션을 위한 컨테이너
        with upload_status.container():  # 모든 업로드 관련 UI를 하나의 컨테이너로 묶음
            st.markdown("### 📤 파일 업로드 진행상황")
            progress_text = st.empty()
            progress_bar = st.progress(0)
            status_messages = []
            status_container = st.empty()

    total_files = len(uploaded_files)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # 진행 상황 업데이트
            with st.sidebar:
                progress_text.text(f"처리 중: {uploaded_file.name}")
                progress_bar.progress((idx + 1) / total_files)

            if uploaded_file.size > CONFIG["max_file_size"]:
                status_messages.append(f"❌ {uploaded_file.name}: 파일 크기 초과")
                status_container.markdown("\n".join(status_messages))
                continue

            if uploaded_file.type != "application/pdf":
                status_messages.append(f"❌ {uploaded_file.name}: PDF 파일이 아님")
                status_container.markdown("\n".join(status_messages))
                continue

            # 파일 처리 로직...
            temp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}.pdf"
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # 문서 해시 계산
            pdf_document = fitz.open(temp_path)
            full_text = ""
            for page in pdf_document:
                full_text += page.get_text()
            
            doc_hash = get_document_hash(full_text, uploaded_file.name)

            if doc_hash in document_hashes:
                status_messages.append(f"ℹ️ {uploaded_file.name}: 이미 처리된 파일")
                status_container.markdown("\n".join(status_messages))
                skipped_files.append(uploaded_file.name)
                continue

            # 문서 처리 및 벡터스토어에 추가
            documents = []
            for page_num in range(len(pdf_document)):
                text = pdf_document[page_num].get_text()
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_num + 1,
                            "doc_hash": doc_hash,
                            "law_name": extract_law_name(uploaded_file.name),
                            "processed_date": datetime.now().isoformat()
                        }
                    ))

            # 문서 분할 및 벡터스토어에 추가
            text_splitter = LawDocumentTextSplitter(
                chunk_size=CONFIG["chunk_size"],
                chunk_overlap=CONFIG["chunk_overlap"]
            )
            split_docs = text_splitter.split_documents(documents)
            vectorstore.add_documents(split_docs)

            # 처리 결과 기록
            document_hashes[doc_hash] = {
                "filename": uploaded_file.name,
                "pages": len(documents),
                "total_chunks": len(split_docs),
                "processed_date": datetime.now().isoformat()
            }
            processed_files.append(uploaded_file.name)
            
            # 성공 메시지 추가
            status_messages.append(f"✅ {uploaded_file.name}: 처리 완료 ({len(split_docs)} 청크)")
            status_container.markdown("\n".join(status_messages))

        except Exception as e:
            status_messages.append(f"❌ {uploaded_file.name}: 처리 실패")
            status_container.markdown("\n".join(status_messages))
            logger.error(f"❌ {uploaded_file.name} 처리 중 오류: {e}")
        
        finally:
            # 리소스 정리
            if 'pdf_document' in locals():
                pdf_document.close()
            if temp_path and temp_path.exists():
                temp_path.unlink()

    # 처리 완료 후
    if processed_files:
        vectorstore.save_local(CONFIG["vectorstore_path"])
        save_document_hashes(hash_file_path, document_hashes)
        with upload_status.container():
            st.success(f"""
            ✅ 파일 처리 완료
            - 새로 추가된 파일: {len(processed_files)}개
            - 건너뛴 파일: {len(skipped_files)}개
            """)
        
        # 3초 후 업로드 상태 메시지 제거
        time.sleep(3)
        upload_status.empty()

    logger.info("=== 업로드된 파일 처리 종료 ===")
    return processed_files, skipped_files

# =========================
# 10. 토큰 수 계산 및 로그 기록 함수
# =========================

def log_document_tokens(docs):
    """
    각 문서의 토큰 수를 계산하여 로그에 기록합니다.
    """
    total_tokens = 0
    for i, doc in enumerate(docs, 1):
        tokens = len(tokenizer.encode(doc.page_content))
        total_tokens += tokens
        logger.debug(f"[문서 {i}] 토큰 수: {tokens}")

    logger.debug(f"총 문서 토큰 수: {total_tokens}")
    return total_tokens

# =========================
# 11. ChatPromptTemplate 정의
# =========================

chat_prompt = ChatPromptTemplate.from_messages([
     (
      "system",
         """당신은 한국의 최고 법률 전문가입니다. 다음 지침을 반드시 준수하여 답변하세요:

        1. [사전검토] 질문내용이 법률과 관련이 없으면 짧게 답변하세요. 
        
        2. [필수] 답변은 다음 내용을 포함하되, 자연스러운 문장으로 작성하세요:
        - 질문에 대한 직접적인 답변
        - 관련 법적 개념 설명
        - 구체적인 법적 근거와 해석
        - 적용 가능한 조항과 **단서조항**을 구체적으로 설명 (특히 제한사항이나 예외사항이 있는 경우 이를 강조)

        3. 참고 법령을 명시할때는 문서의 제목에 있는 법령과 조항을 명시하세요.         

        4. [중요] 검색된 문서에 같은 내용이 있는 경우, 법적 효력에 따라 다음 순서로 우선순위를 두고 답변하세요:
        - 법률
        - 시행령
        - 업무편람 등 비법령 자료
        * 여러 문서에서 관련 규정이 발견될 경우, 법적 효력이 높은 문서를 우선으로 인용하고, 필요 시 다른 자료는 보충 설명에 활용하세요.             

        5. [형식] 법률적인 질문에 대한 답변 마지막에는 다음 두 가지를 추가하세요:
        ⚖️ 결론: 핵심 내용을 간단히 정리
        📌 참고 법령: 인용된 법령 목록

        6. 답변이 불확실한 경우 "제공된 문서에서 해당 내용을 찾을 수 없습니다."라고 명시하세요.

        7. [참고] 모든 조항 인용 시 "「법률명」 제X조 제X항"과 같이 정확한 출처를 표시하세요.
                """
    ),
    ("user", "{chat_history}\n\n[제공된 문서]\n{context}\n\n[질문]\n{question}")
])

# =========================
# 12. 커스텀 콜백 핸들러 정의
# =========================

class StreamlitCallbackHandler(BaseCallbackHandler): 
    """Streamlit을 사용하여 실시간으로 답변을 표시하는 콜백 핸들러."""

    def __init__(self, message_placeholder):
        self.message_placeholder = message_placeholder
        self.answer_text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        """새로운 토큰이 생성될 때마다 호출됩니다."""
        skip_patterns = ["Human:", "Assistant:", "질문: ", "답변: "]
        if any(pattern in token for pattern in skip_patterns):
            return
            
        self.answer_text += token
        self.message_placeholder.markdown(self.answer_text + "▌")

    def on_llm_end(self, response, **kwargs):
        """LLM 응답 완료되면 호출됩니다."""
        final_answer = self.answer_text.strip()
        self.message_placeholder.markdown(final_answer)

# =========================
# 12-1. qa_chain 초기화 함수 추가
# =========================

def initialize_qa_chain(vectorstore):
    """QA Chain을 초기화하고 세션 상태에 저장"""
    if 'qa_chain' not in st.session_state or st.session_state['qa_chain'] is None:
        # 초기화 시 단일 콜백 핸들러 생성
        message_placeholder = st.empty()
        callback_handler = StreamlitCallbackHandler(message_placeholder)
        callback_manager = CallbackManager([callback_handler])

        llm = ChatOpenAI(
            model_name=CONFIG["model_name"],
            temperature=0,
            streaming=True,
            openai_api_key=CONFIG["api_keys"]["openai"],
            callback_manager=callback_manager
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=st.session_state['memory'],
            return_source_documents=True,
            chain_type="stuff",
            combine_docs_chain_kwargs={
                "prompt": chat_prompt,
                "document_variable_name": "context",
            },
            verbose=False,
            output_key='answer'
        )
        st.session_state['qa_chain'] = qa_chain
        st.session_state['callback_handler'] = callback_handler  # 현재 핸들러 저장

    return st.session_state['qa_chain'], st.session_state['callback_handler']


def is_relevant_question(question, vectorstore, threshold=0.5):
    """질문이 벡터스토어의 문서와 관련이 있는지 확인하는 함수"""
    # 질문 임베딩 생성
    question_embedding = model.encode(question, convert_to_tensor=True)

    # 벡터스토어의 문서 임베딩과 유사성 계산
    cosine_similarities = util.pytorch_cos_sim(question_embedding, vectorstore.embeddings)

    # 유사성 임계값을 기준으로 관련성 판단
    return cosine_similarities.max().item() > threshold


# =========================
# 13. 메인 함수 정의
# =========================

def main():
    # ====================
    # 1. 세션 상태 초기화
    # ====================
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'qa_chain' not in st.session_state:
        st.session_state['qa_chain'] = None
    if 'callback_handler' not in st.session_state:
        st.session_state['callback_handler'] = None

    # 벡터스토어와 해시 파일, 메모리 초기화
    vectorstore, hash_file_path = initialize_vectorstore_and_memory()

    # 토크나이저 초기화 (전역)
    global tokenizer
    tokenizer = tiktoken.encoding_for_model(CONFIG["model_name"])

    with st.sidebar:
        st.header("📄 문서 관리")
        
        # 리셋 버튼
        if st.button("🔄 벡터스토어 초기화", help="청크 사이즈 변경 등을 위해 벡터스토어를 초기화합니다"):
            if reset_vectorstore():
                st.success("벡터스토어가 초기화되었습니다. 페이지를 새로고침하세요.")
                st.info("새로운 설정으로 문서를 다시 업로드해주세요.")
            else:
                st.error("벡터스토어 초기화 중 오류가 발생했습니다.")

        # 구분선 추가
        st.markdown("---")
        
        # 파일 업로더
        st.subheader("📤 문서 업로드")
        uploaded_files = st.file_uploader(
            "PDF 파일을 업로드하세요",
            type=["pdf"],
            accept_multiple_files=True,
            key="uploader",
            help="최대 업로드 크기: 20MB",
        )

        if uploaded_files:
            with st.spinner("문서 처리 중..."):
                processed_files, skipped_files = process_uploaded_files(
                    uploaded_files, vectorstore, hash_file_path
                )

        # 구분선 추가
        st.markdown("---")
        
        # 문서 통계
        st.subheader("📊 문서 통계")
        document_hashes = load_document_hashes(hash_file_path)
        if document_hashes:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 문서 수", f"{len(document_hashes)}개")
            with col2:
                total_pages = sum(doc_info.get('pages', 0) for doc_info in document_hashes.values())
                st.metric("총 페이지 수", f"{total_pages}개")
            with col3:
                total_chunks = sum(doc_info.get('total_chunks', 0) for doc_info in document_hashes.values())
                st.metric("총 청크 수", f"{total_chunks}개")

    st.markdown("<h3 style='text-align: center;'>💬 청렴법률 상담챗봇</h3>", unsafe_allow_html=True)

    # 과거 대화 표시
    for i in range(len(st.session_state['generated'])):
        st.chat_message("user").markdown(st.session_state['past'][i])
        st.chat_message("assistant").markdown(st.session_state['generated'][i])

    # QA Chain 초기화 (한 번만 실행됨) 및 콜백 핸들러 가져오기
    qa_chain, callback_handler = initialize_qa_chain(vectorstore)

    # 사용자 입력 처리
    if question := st.chat_input("법률 관련 질문을 입력하세요"):
        st.chat_message("user").markdown(question)
        st.session_state['past'].append(question)

        try:
            # 질문과 관련된 문서 검색
            retrieved_docs = vectorstore.similarity_search(question, k=5)
            log_document_tokens(retrieved_docs)

            logger.info("\n=== 질문 처리 시작 ===")
            logger.info(f"질문: {question}")
            logger.info(f"검색된 문서 수: {len(retrieved_docs)}개")

            # context 생성 코드
            context = ""
            for i, doc in enumerate(retrieved_docs, 1):
                sources = doc.metadata.get("law_name", "출처 정보 없음")
                page = doc.metadata.get("page", "페이지 정보 없음")
                context += f"\n\n 📜관련법령 {i}] {sources}, \n📄 페이지 {page}: \n💡 내용:\n{doc.page_content}\n"
                        
                logger.info(f"\n[문서 {i}]")
                logger.info(f"[입력 전체 정보 {context}]")
                logger.info(f"출처: {doc.metadata.get('law_name', '알 수 없음')}")
                logger.info(f"페이지: {doc.metadata.get('page', '알 수 없음')}")
                logger.info(f"내용 요약: {doc.page_content[:200]}...")

            logger.info("=== 검색 결과 종료 ===\n")
            
            # 콜백 핸들러의 message_placeholder 업데이트
            callback_handler.message_placeholder = st.empty()
            callback_handler.answer_text = ""  # 이전 응답 초기화

            # 질문에 대한 응답 생성
            response = qa_chain({"question": question})
            source_docs = response.get('source_documents', [])
            
            # 생성된 답변을 세션에 저장
            st.session_state['generated'].append(callback_handler.answer_text)
            logger.info("=== 질문 처리 완료 ===\n")

        except Exception as e:
            logger.error(f"질문 처리 중 오류 발생 - {question}: {str(e)}")
            st.error(f"❌ 질문 처리 실패: {str(e)}")
            st.session_state['generated'].append("죄송합니다. 답변을 생성하는 데 실패했습니다.")
            traceback.print_exc()

# =========================
# 14. 실행 시작
# =========================

if __name__ == "__main__":
    main()
