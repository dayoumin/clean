import os
import logging
import warnings
from pathlib import Path
from threading import Lock
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler

# 페이지 설정
st.set_page_config(
    page_title="⚖️ 청렴 법률 챗봇",
    page_icon="⚖️",
    layout="wide"
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# DeprecationWarning 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)

# OpenAI API 키 설정

load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다.")
    st.stop()

# 세션 Lock 생성 (스레드 안전성 확보)
session_lock = Lock()

# 벡터스토어 경로 설정
VECTORSTORE_PATH = Path("chroma_vectorstore")

# 채팅 프롬프트 정의
chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """당신은 한국의 최고 법률 전문가입니다. 다음 지침에 따라 명확하고 신뢰성 있는 답변을 제공하세요:
        
        1. **법률 전문가로서 법적 논리에 근거한 설명**을 제공합니다.
        - 필요 시 법률명과 조항을 자연스럽게 포함합니다. (예: "공직자 윤리법 제28조에 따르면...")

        2. 예외 사항이 있는 경우 명확히 구분해 설명합니다.
        - 중요한 정보는 **이모지(🔍, 📄)** 등을 사용해 강조합니다.
        
        3. 설명은 **500자 이내**로 간결하게 작성하고, 중요한 부분은 **줄바꿈**과 **강조**를 사용합니다.
        """
    ),
    ("user", "{chat_history}\n\n{context}\n\n질문: {question}")
])

# Streamlit 콜백 핸들러 정의
class StreamlitCallbackHandler(BaseCallbackHandler):
    """Streamlit을 사용해 실시간으로 답변을 표시하는 핸들러."""
    def __init__(self, message_placeholder):
        self.message_placeholder = message_placeholder
        self.answer_text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        """LLM이 새 토큰을 생성할 때마다 호출됩니다."""
        if token.strip() not in ["Human:", "Assistant:", "질문:", "답변:"]:
            self.answer_text += token
            self.message_placeholder.markdown(self.answer_text + "▌")

    def on_llm_end(self, response, **kwargs):
        """LLM 응답 완료 후 마지막 커서를 제거합니다."""
        self.message_placeholder.markdown(self.answer_text.strip())

# 벡터스토어 초기화 함수
def initialize_vectorstore():
    """로컬 벡터스토어를 초기화합니다."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)

    if VECTORSTORE_PATH.exists():
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_PATH),
            embedding_function=embeddings,
            collection_name="law_documents"
        )
        logging.info("벡터스토어 로드 완료.")
    else:
        st.error("벡터스토어가 존재하지 않습니다. 관리자에게 문의하세요.")
        st.stop()

    if 'memory' not in st.session_state:
        st.session_state['memory'] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    return vectorstore

# 벡터스토어 업데이트 함수
def update_vectorstore(documents):
    """새로운 문서를 벡터스토어에 추가합니다."""
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_PATH)
    )
    vectorstore.persist()
    logging.info("벡터스토어 업데이트 완료.")

# 메인 함수
def main():
    # 사이드바 제거
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .css-1d391kg {display: none;} /* 사이드바 숨김 */
        .block-container {padding-left: 0rem;} /* 페이지 정렬 조정 */
        </style>
    """, unsafe_allow_html=True)

    # 벡터스토어 초기화
    vectorstore = initialize_vectorstore()

    # 채팅 인터페이스 구현
    st.markdown("<h3 style='text-align: center;'>💬 청렴 법률 상담챗봇</h3>", unsafe_allow_html=True)

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    # 이전 채팅 히스토리 표시
    for i in range(len(st.session_state['generated'])):
        st.chat_message("user").markdown(st.session_state['past'][i])
        st.chat_message("assistant").markdown(st.session_state['generated'][i])

    # 사용자 입력 받기
    if question := st.chat_input("법률 관련 질문을 입력하세요"):
        st.chat_message("user").markdown(question)
        st.session_state['past'].append(question)

        # 답변 생성 및 표시
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            callback_handler = StreamlitCallbackHandler(message_placeholder)
            callback_manager = CallbackManager([callback_handler])

            # LLM 생성
            llm = ChatOpenAI(
                model_name="gpt-4o",
                temperature=0,
                streaming=True,
                openai_api_key=openai_api_key,
                callback_manager=callback_manager
            )

            # ConversationalRetrievalChain 생성
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                memory=st.session_state['memory'],
                return_source_documents=False,
                chain_type="stuff",
                combine_docs_chain_kwargs={"prompt": chat_prompt},
                verbose=False
            )

            try:
                response = qa_chain({"question": question})
                st.session_state['generated'].append(callback_handler.answer_text)
            except Exception as e:
                logger.error(f"질문 처리 중 오류 발생: {str(e)}")
                st.error(f"❌ 답변 생성 실패: {str(e)}")
                st.session_state['generated'].append("죄송합니다. 답변을 생성할 수 없습니다.")

if __name__ == "__main__":
    main()
