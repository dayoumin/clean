# app.py

import os
import sys
import openai
import logging
import warnings
import json
import re
import traceback
from pathlib import Path
from datetime import datetime
import faiss
import streamlit as st
from dotenv import load_dotenv
import tiktoken

from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.docstore.document import Document


# =========================
# 1. CONFIG 딕셔너리 정의
# =========================

CONFIG = {
    "model_name": "gpt-4o-mini",  # LLM 모델명 (오타 수정: "gpt-4o" → "gpt-4")
    "embedding_model": "text-embedding-3-large",  # 임베딩 모델명 (실제 사용 가능한 모델명으로 수정)
    "vectorstore_path": "faiss_vectorstore",  # 벡터스토어 경로
    "hash_file_path": "faiss_vectorstore/document_hashes.json",  # 문서 해시 파일 경로
    "api_keys": {
        "openai": None,  # OpenAI API 키
    }
}

# =========================
# 2. 로깅 설정
# =========================

class CustomFormatter(logging.Formatter):
    def format(self, record):
        return f"\n--- {record.levelname} ---\n{super().format(record)}\n"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('chat_server.log', encoding='utf-8')
    ]
)

for handler in logging.root.handlers:
    handler.setFormatter(CustomFormatter())

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================
# 3. API 키 로드
# =========================

load_dotenv()

CONFIG["api_keys"]["openai"] = os.getenv("OPENAI_API_KEY")

if not CONFIG["api_keys"]["openai"]:
    logger.error("OpenAI API 키가 설정되지 않았습니다.")
    st.error("OPENAI_API_KEY가 설정되지 않았습니다.")
    st.stop()

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

def load_document_hashes(hash_file_path: Path) -> dict:
    """저장된 문서 해시 로드"""
    try:
        return json.loads(hash_file_path.read_text())
    except Exception as e:
        logger.error(f"해시 파일 로드 실패: {e}")
        return {}

# =========================
# 5. 벡터스토어 초기화 함수
# =========================

def initialize_vectorstore(embedding_model):
    VECTORSTORE_PATH = Path(CONFIG["vectorstore_path"])
    HASH_FILE_PATH = Path(CONFIG["hash_file_path"])

    embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=CONFIG["api_keys"]["openai"])

    if 'vectorstore' in st.session_state:
        logger.info("세션에서 벡터스토어를 재사용합니다.")
        return st.session_state['vectorstore'], HASH_FILE_PATH

    try:
        vectorstore = LangChainFAISS.load_local(
            str(VECTORSTORE_PATH), 
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("기존 벡터스토어를 로드했습니다.")
    except Exception as e:
        logger.error(f"벡터스토어 로드 실패: {e}")
        st.error("벡터스토어를 로드하는 데 실패했습니다.")
        st.stop()

    st.session_state['vectorstore'] = vectorstore
    return vectorstore, HASH_FILE_PATH

# =========================
# 6. 커스텀 콜백 핸들러 정의
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
        self.answer_text = ""

# =========================
# 7. 토큰 수 계산 및 로그 기록 함수
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
# 8. ChatPromptTemplate 정의
# =========================

chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",

        """당신은 한국의 최고 법률 전문가입니다. 국립수산과학원의 근무자들을 위해, **청렴**(반부패, 윤리 등) 관련 법률 상담만을 제공합니다. 다음 지침을 반드시 준수하여 답변하세요:

        1. **[중요]** 질문이 청렴, 복무 등과 관련된 질문**인 경우에만 아래 지침에 따라 답변하세요.

        2. **[필수]** 답변은 자연스러운 문장으로 작성하며, 다음 내용을 포함하세요:
        - 질문에 대한 직접적인 답변
        - 관련 법적 개념 설명
        - 구체적인 법적 근거와 해석
        - 적용 가능한 조항 설명
        - **단서조항**, **예외사항** 등은 명확히 명시

        3. **[참고 법령 명시]** 참고 법령을 명시할 때는 **법령의 제목**과 **조항**을 정확히 명시하세요.
        - 예시: "이는 「공직자윤리법」 제5조 제2항에 근거합니다."

        4. **[법적 효력 우선순위]** 제공된 문서에 같은 내용이 여러 번 언급된 경우, 법적 효력에 따라 다음 순서로 우선순위를 두고 답변하세요:
        - **법률**
        - **시행령**
        - **업무편람 등 비법령 자료**
        - 여러 문서에서 관련 규정이 발견될 경우, 법적 효력이 높은 문서를 우선으로 인용하고, 필요 시 다른 자료는 보충 설명에 활용하세요.

        5. **[형식]** 청렴 관련 법률적인 질문에 답변할 경우에만, 답변 마지막에 다음 두 가지를 추가하세요:
        - ⚖️ **결론**: 핵심 내용을 간단히 정리
        - 📌 **참고 법령**: 인용된 법령 목록

        6. **[참고 법령 형식]** 결론과 '참고 법령' 목록 사이에 반드시 줄바꿈을 하세요.

        7. **[질문 분류]** 질문이 청렴과 관련된 법률적인 질문이 아닌 경우, 다음과 같이 답변하세요:
        - **청렴과 관련된 일반적인 질문인 경우**: "죄송하지만, 본 챗봇은 청렴 관련 법률 상담에만 답변드립니다."
        - **인사말 등 small talk은 **답변하지만, 짧게 인사말만 하고 "본 챗봇은 청렴 관련 법률 상담에만 답변드립니다."라고 하세요.
        - **청렴과 무관한 질문인 경우**: "죄송하지만, 본 챗봇은 청렴 관련 법률 상담을 위한 챗봇입니다."

        8. **[출처 표기]** 모든 조항 인용 시 "「법률명」 제X조 제X항"과 같이 정확한 출처를 표시하세요.
        
        """
    ),
    ("user", "\n\n[제공된 문서]\n{context}\n\n[질문]\n{question}")
])

# =========================
# 9. 메인 함수 정의
# =========================

# 페이지 설정을 메인 함수 밖으로 이동
import streamlit as st
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 테마 설정
current_hour = datetime.now().hour
is_dark_mode = current_hour < 6 or current_hour >= 18

if is_dark_mode:
    bg_color = "#1a1a1a"
    text_color = "#ffffff"
    header_bg = "#2d2d2d"
    chat_bg = "#1a1a1a"
    user_msg_bg = "#4a4a4a"
    assistant_msg_bg = "#2d2d2d"
    input_bg = "#2d2d2d"
else:
    bg_color = "#f5f6f7"
    text_color = "#1a1a1a"
    header_bg = "#ffffff"
    chat_bg = "#f5f6f7"
    user_msg_bg = "#007AFF"
    assistant_msg_bg = "#ffffff"
    input_bg = "#ffffff"

# CSS 스타일
st.markdown(f"""
    <style>
    /* 기본 Streamlit 요소 숨기기 */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    [data-testid="stToolbar"] {{display: none !important;}}
    
    /* 전체 배경색 설정 */
    .stApp {{
        background-color: {bg_color};
        max-width: 100vw;
        overflow-x: hidden;
    }}
    
    /* 상단 헤더 스타일 */
    .fixed-header {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 50px;
        background: {header_bg};
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        z-index: 1000;
    }}
    
    .header-title {{
        font-size: 1rem;
        color: {text_color};
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    
    /* 채팅 컨테이너 스타일 */
    .stChatFloatingInputContainer {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 8px;
        background: {input_bg};
        border-top: 1px solid rgba(0,0,0,0.1);
        z-index: 999;
    }}
    
    /* 메시지 컨테이너 스타일 */
    .stChatMessageContent {{
        padding: 10px 12px;
        border-radius: 18px;
        max-width: 85%;
        margin: 4px 8px;
        font-size: 0.95rem;
        word-break: break-word;
    }}
    
    /* 사용자 메시지 스타일 */
    [data-testid="StChatMessage"][data-testid="user"] {{
        justify-content: flex-end;
        padding-left: 10%;
    }}
    
    [data-testid="StChatMessage"][data-testid="user"] .stChatMessageContent {{
        background: {user_msg_bg};
        color: #ffffff;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }}
    
    /* 어시스턴트 메시지 스타일 */
    [data-testid="StChatMessage"]:not([data-testid="user"]) {{
        padding-right: 10%;
    }}

    [data-testid="StChatMessage"]:not([data-testid="user"]) .stChatMessageContent {{
        background: {assistant_msg_bg};
        color: {text_color};
        border-bottom-left-radius: 4px;
        margin-right: auto;
    }}
    
    /* 어시스턴트 아이콘 스타일 */
    [data-testid="StChatMessage"]:not([data-testid="user"]) .stChatMessageAvatar {{
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="%23666666"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/></svg>');
        background-size: 65%;
        background-position: center;
        background-repeat: no-repeat;
        background-color: transparent;
        width: 28px;
        height: 28px;
        margin-right: 6px;
    }}
    
    /* 사용자 아이콘 숨기기 */
    [data-testid="StChatMessage"][data-testid="user"] .stChatMessageAvatar {{
        display: none;
    }}
    
    /* 채팅 입력창 스타일 */
    .stChatInputContainer {{
        background: {input_bg};
        border-radius: 20px;
        margin: 0 8px;
        padding: 6px 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    
    textarea {{
        border: none !important;
        background: transparent !important;
        padding: 8px !important;
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
        max-height: 100px !important;
    }}
    
    /* 스크롤바 스타일 */
    ::-webkit-scrollbar {{
        width: 4px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: transparent;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: rgba(0,0,0,0.2);
        border-radius: 2px;
    }}
    
    /* 채팅 영역 여백 조정 */
    .main .block-container {{
        padding-top: 60px;
        padding-bottom: 80px;
        padding-left: 0;
        padding-right: 0;
    }}

    /* 모바일 최적화 */
    @media (max-width: 768px) {{
        .stChatMessageContent {{
            max-width: 90%;
            font-size: 0.9rem;
            padding: 8px 12px;
        }}
        
        .header-title {{
            font-size: 0.95rem;
        }}
        
        .stChatInputContainer {{
            margin: 0 4px;
        }}
        
        textarea {{
            font-size: 0.9rem !important;
        }}
    }}
    </style>
    
    <div class="fixed-header">
        <div class="header-title">
            <span>⚖️</span>
            <span>청렴법률 상담챗봇</span>
        </div>
    </div>
""", unsafe_allow_html=True)


def main():
    
    
    # 스타일 설정 적용
    
    
    # 벡터스토어와 해시 파일 초기화
    vectorstore, hash_file_path = initialize_vectorstore(CONFIG["embedding_model"])

    # 토크나이저 초기화 (전역)
    global tokenizer
    try:
        tokenizer = tiktoken.encoding_for_model(CONFIG["model_name"])
    except KeyError:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    # 메모리 초기화 - 대화 기록 유지를 위해 필요
    if 'memory' not in st.session_state:
        st.session_state['memory'] = ConversationBufferMemory(
            memory_key="chat_history",
            human_prefix="질문",
            ai_prefix="답변",
            return_messages=True,
            output_key='answer'
        )

    # QA 체인 초기화 (한 번만 생성)
    if 'qa_chain' not in st.session_state:
        llm = ChatOpenAI(
            model_name=CONFIG["model_name"],
            temperature=0,
            streaming=True,
            openai_api_key=CONFIG["api_keys"]["openai"]
        )

        st.session_state['qa_chain'] = ConversationalRetrievalChain.from_llm(
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
        logger.info("QA 체인을 초기화하고 세션 상태에 저장했습니다.")
    else:
        callback_handler = st.session_state.get('callback_handler')

    # 메인 컨테이너 시작
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # 채팅 영역 시작
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    
    # 메시지 표시
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_msg = {
            "role": "assistant",
            "content": "안녕하세요! 청렴법률 상담챗봇입니다. 궁금하신 점이 있으면 알려주세요. 😊"
        }
        st.session_state.messages.append(welcome_msg)

    # 모든 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 입력창
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    if question := st.chat_input("💭 법률 관련 질문을 입력하세요"):
        # 사용자 메시지 추가 및 표시
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            # 질문과 관련된 문서 검색
            retrieved_docs = vectorstore.similarity_search(question, k=5)
            log_document_tokens(retrieved_docs)

            logger.info("\n=== 질문 처리 시작 ===")
            logger.info(f"질문: {question}")
            logger.info(f"검색된 문서 수: {len(retrieved_docs)}개")

            # context 생성
            context = ""
            for i, doc in enumerate(retrieved_docs, 1):
                law_name = doc.metadata.get("law_name", "출처 정보 없음")
                page = doc.metadata.get("page", "페이지 정보 없음")
                content = doc.page_content

                context += f"\n\n 관련법령 {i}] {law_name}, \n📄 페이지 {page}: \n💡 내용:\n{content}\n"

                logger.info(f"\n[문서 {i}]")
                logger.info(f"출처: {law_name}")
                logger.info(f"페이지: {page}")
                logger.info(f"내용 요약: {content[:200]}...")

            logger.info("=== 검색 결과 종료 ===\n")
            
            # 응답 생성 및 표시
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                callback_handler = StreamlitCallbackHandler(message_placeholder)
                callback_manager = CallbackManager([callback_handler])
                
                # 응답 생성을 위한 임시 LLM 인스턴스 생성
                temp_llm = ChatOpenAI(
                    model_name=CONFIG["model_name"],
                    temperature=0,
                    streaming=True,
                    openai_api_key=CONFIG["api_keys"]["openai"],
                    callback_manager=callback_manager
                )
                
                # 임시 QA 체인 생성
                temp_qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=temp_llm,
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
                
                response = temp_qa_chain({"question": question})
                
                # 응답을 메시지 기록에 추가
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})

            logger.info("=== 질문 처리 완료 ===\n")

            # 응답 생성 후 스크롤
            st.markdown("""
                <script>
                    setTimeout(scrollToBottom, 100);
                    setTimeout(scrollToBottom, 500);
                </script>
            """, unsafe_allow_html=True)

        except Exception as e:
            logger.error(f"질문 처리 중 오류 발생 - {question}: {str(e)}")
            st.error(f"❌ 질문 처리 실패: {str(e)}")
            traceback.print_exc()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()  # main() 한 번만 호출