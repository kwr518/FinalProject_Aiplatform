# import os
# from typing import Optional

# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

# class LLMService:
#     def __init__(self):
#         """
#         에러 수정 완료: Pydantic 및 ChatGroq 호환성 최적화
#         """
#         # 1. API 키 설정 (직접 입력보다 환경변수 우선 권장하지만 일단 코드에 유지)
#         os.environ["GROQ_API_KEY"] = "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"
        
#         # 2. 경로 설정 (app/models/ 하위)
#         base_dir = os.path.dirname(os.path.dirname(__file__))
#         self.db_path = os.path.join(base_dir, "models", "chroma_db_combined10")

#         # 3. 임베딩 로드
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="BAAI/bge-m3",
#             model_kwargs={"device": "cpu"}
#         )
# #llama-3.3-70b-versatile
#         # 4. LLM 로드 (💡 에러 해결을 위해 인자 간소화)
#         # proxies 에러가 발생할 경우 아래와 같이 모델명만 정확히 기입합니다.
#         self.llm = ChatGroq(
#             model_name="llama-3.1-8b-instant",
#             temperature=0
#         )

#         # 5. VectorStore 로드
#         self.vectorstore = Chroma(
#             persist_directory=self.db_path,
#             embedding_function=self.embeddings
#         )

#         # 6. Prompt 설정
#         system_prompt = (
#             "당신은 대한민국 교통법규 전문가입니다.\n"
#             "아래 제공된 [데이터]에 근거해서만 답변하세요.\n"
#             "데이터에 없는 내용은 추측하지 말고 "
#             "'제공된 자료에서 해당 정보를 찾을 수 없습니다.'라고 답하세요.\n\n"
#             "{context}"
#         )

#         self.prompt = ChatPromptTemplate.from_messages([
#             ("system", system_prompt),
#             ("human", "{input}"),
#         ])

#         # 7. RAG 체인 구성
#         combine_docs_chain = create_stuff_documents_chain(
#             self.llm,
#             self.prompt
#         )

#         # 💡 에러 방지를 위해 retriever 설정을 좀 더 유연하게 가져갑니다.
#         retriever = self.vectorstore.as_retriever(
#             search_kwargs={"k": 5}
#         )

#         self.rag_chain = create_retrieval_chain(
#             retriever,
#             combine_docs_chain
#         )

#         print("✅ LLM RAG 서비스 초기화 완료")

#     def get_answer(self, question: str) -> str:
#         try:
#             response = self.rag_chain.invoke({"input": question})
#             return response.get("answer", "답변을 생성할 수 없습니다.")
#         except Exception as e:
#             print(f"❌ LLM 호출 에러: {e}")
#             return f"에러 발생: {str(e)}"

# # 싱글톤 인스턴스 관리
# _llm_manager: Optional[LLMService] = None

# def get_llm_manager() -> LLMService:
#     global _llm_manager
#     if _llm_manager is None:
#         _llm_manager = LLMService()
#     return _llm_manager

# llm_manager = get_llm_manager()

import os
from dotenv import load_dotenv
from typing import Optional

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class LLMService:
    def __init__(self):
        """
        교통법규 전문가 및 신고 초안 생성기 통합 서비스
        """
        # 0. .env 파일 로드 (가장 먼저 실행)
        load_dotenv()

        # 1. API 키 확인 (디버깅용 안전장치)
        if not os.getenv("GROQ_API_KEY"):
            print("🚨 에러: .env 파일에서 GROQ_API_KEY를 찾을 수 없습니다.")
        
        # 2. 경로 및 리소스 로드
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.db_path = os.path.join(base_dir, "models", "chroma_db_combined10")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"}
        )

        # 3. 듀얼 모델 설정
        # ChatGroq는 os.environ["GROQ_API_KEY"]를 자동으로 감지하므로 
        # 별도로 api_key 인자를 넘겨주지 않아도 됩니다.
        self.llm_70b = ChatGroq(
            model_name="llama-3.3-70b-versatile", 
            temperature=0
        )

        # 4. VectorStore 로드
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # ---------------------------------------------------------
        # 5. [프롬프트 1] 법률 전문가 상담 템플릿
        # ---------------------------------------------------------
        law_template = """당신은 대한민국 교통법규 전문가입니다. 
제공된 [데이터]를 바탕으로 답변하되, CSV와 PDF의 정보를 하나도 빠짐없이 정리하세요.
다음 형식을 엄격히 지켜 답변하세요.
반드시 [데이터]에 근거가 있을 때만 답변하고, 데이터에 없으면 "관련 정보를 찾을 수 없습니다"라고 안내 해주세요.

[답변 규칙]
1. **질문의 본질 파악**: 질문이 '용어 정의'나 '차이점'을 묻는 것이라면 PDF 설명을 최우선으로 정리하세요.
2. 위반 행위의 **정확한 명칭**을 명시하세요.
3. 금액 정보는 **마크다운 표(Table)**를 사용하여 차종별로 비교하세요.
4. **PDF 데이터 우선 활용**: 감속 기준, 도로 차이점 등을 상세히 설명하세요.
5. 두가지 이상의 위반 발생 시 각각 나누어 분석하고 합계표를 작성하세요.
6. 근거 없는 추론 금지. 7. '노란선'은 '중앙선'으로 해석.
12. 정의/형사처벌은 PDF, 단순 범칙금은 CSV 우선.
14. **음주운전 질문 시**: PDF의 '징역 및 벌금' 수치를 반드시 기재하세요.
20. 벌점 수치가 명확하지 않으면 '벌점 수치가 명시되지 않음'이라고 답변하세요.

### 1. 주요 개념 및 위반 항목 분석
- 분석 대상: (질문에서 언급된 주요 개념 또는 위반 행위 이름) 
- 법적 근거: (데이터에 명시된 법규 명칭)
- 상세 내용 : (용어 정의, 행동 요령 등 상세 서술)

### 2. 관련 수치 정보 (속도 또는 범칙금)
| 구분 | 차종 | 관련 수치 (속도/금액) |
| :--- | :--- | :--- |

[데이터]:
{context}

질문: {input}
답변:"""

        # ---------------------------------------------------------
        # 6. [프롬프트 2] 안전신문고 신고 초안 생성 템플릿
        # ---------------------------------------------------------
        report_template = """당신은 대한민국 안전신문고 신고 초안 생성기 입니다. 
제공된 [데이터]를 바탕으로 답변하되, CSV와 PDF의 정보를 하나도 빠짐없이 정리하세요.
다음 형식을 엄격히 지켜 답변하세요.

[답변 규칙]
1. 사용자가 제공한 내용을 전부 포함해서 출력하세요.
2. 위치, 시각, 위반 행위를 반드시 포함하세요.
3. '노란선'은 '중앙선'으로 해석하세요.
4. 법적 근거나 금액은 출력하지 마세요.
5. 사용자가 안전 신문고에 바로 신고할 수 있도록 사용자의 입장에서 상세 내용을 작성하세요.

### 1. 위반 일시 
- 일시: (사용자가 제공한 날짜 및 시간)
### 2. 위반 위치
- 위치: (사용자가 제공한 위치)
### 3. 위반 항목 분석
- 분석 대상: (위반 행위 이름)
- 상세 내용 : (신고용 상세 설명 문장)

[데이터]:
{context}

질문: {input}
답변:"""

        # 7. 각각의 RAG 체인 생성
        law_doc_chain = create_stuff_documents_chain(self.llm_70b, ChatPromptTemplate.from_template(law_template))
        self.law_chain = create_retrieval_chain(self.retriever, law_doc_chain)

        report_doc_chain = create_stuff_documents_chain(self.llm_70b, ChatPromptTemplate.from_template(report_template))
        self.report_chain = create_retrieval_chain(self.retriever, report_doc_chain)

        print("✅ 교통법규 AI 전문가 및 신고 초안 시스템 로드 완료")

    # 💡 기능 1: 법률 상담 답변
    def get_law_answer(self, question: str) -> str:
        try:
            response = self.law_chain.invoke({"input": question})
            return response.get("answer", "답변을 생성할 수 없습니다.")
        except Exception as e:
            return f"법률 상담 에러: {str(e)}"

    # 💡 기능 2: 신고 초안 작성
    def get_report_draft(self, question: str) -> str:
        try:
            response = self.report_chain.invoke({"input": question})
            return response.get("answer", "초안을 생성할 수 없습니다.")
        except Exception as e:
            return f"초안 생성 에러: {str(e)}"

# 싱글톤 인스턴스 관리
_llm_manager: Optional[LLMService] = None

def get_llm_manager() -> LLMService:
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMService()
    return _llm_manager