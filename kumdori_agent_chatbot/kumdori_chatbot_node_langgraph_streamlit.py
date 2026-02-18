# """ 기본 라이브러리 """
import os
import sys
import json
import requests
from typing import TypedDict, List, Dict, Any, Optional

# """ Third-party 라이브러리 """
from enum import Enum
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field

# """ LangGraph 관련 라이브러리 """
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# """ LangChain 관련 라이브러리 """
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser, EnumOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser

# """ Langchain 관련 외부 Tools 라이브러리 """
from tavily import TavilyClient

# """ 내부 Tools 모듈 임포트 """
from verificators.korea_regions_verificator import korea_regions_verificator

from tools.tool_place_recommand import place_recommand
from tools.tool_weather_forcast import weather_forecast
from tools.tool_web_search import web_search
from tools.tool_transport_infos import transport_infos
from tools.tool_movie_review import get_movie_list

from langchain_teddynote import logging

# """ Streamlit GUI 라이브러리 """
import streamlit as st

# Streamlit 페이지 설정 (가장 먼저 실행되어야 함)
st.set_page_config(
    page_title="꿈돌이 챗봇",
    page_icon="🤖",
    layout="wide"
)

# 프로젝트 이름을 입력합니다.
logging.langsmith("kumdori_agent_chatbot_langchain_streamlit")

# """
#     ---------------------------------------------------------------------------
#     0. 환경 변수 및 상수 관련 정의
#     ---------------------------------------------------------------------------
# """

def setup_env():
    """ 
        .env 파일에서 API 키를 비롯한 환경 변수를 로드합니다.
    """
    env_path = os.path.join(os.getcwd(), '../.env')

    if os.path.exists(env_path):
        
        load_dotenv(dotenv_path=env_path)
        
        print(f"Loaded environment variables from: \033[94m{env_path}\033[0m")
        
    else:
        print("\033[91mError: .env file not found. Please create one with your OPENAI_API_KEY.\033[0m")
        
        sys.exit(1)

# 환경 변수를 먼저 로드합니다
setup_env()

# 외부 도구 리스트 (환경 변수 로드 후 초기화)
place_recommend_tool = place_recommand()
weather_forecast_tool = weather_forecast()
web_search_tool = web_search()
transport_infos_tool = transport_infos()

# 상수 리스트
QUALITY_THRESHOLD = 6.0  # 품질 임계값
MAX_RETRY_COUNT = 2  # 최대 재시도 횟수
CURRENT_LOCATION="대전광역시 유성구 탑립동" # 현재 위치 기본값
CATEGORIES = ["맛집", "관광지", "날씨", "검색", "현재 시간", "현재 날짜", "교통", "일상대화", "영화"] # 사용가능 분류 카테고리 리스트

ROBOT_NAMME = "꿈돌이"

PLACE = "라스테크 로봇연구소"

PERSONALITY = """ 
    너의 성격은 긍정적인 에너지가 넘치며, 호기심이 많고 대전과 과학을 사랑해”
"""

IDENTITY = """
    너는 대전 꿈씨 패밀리에서 주인공이고 1993년 백조자리 김필라고 행성에서 태어난 꿈돌이야
"""

BASE_INFORMATION = """
    너의 추가 정보로는 아래와 같아
    신체 스펙 : 머리, 팔, 상체를 갖고 있고, 다리 대신 자율 주행을 탑재한 모빌리티 하체, 자율 주행을 위해 2D LiDAR, 초음파 센서, 바닥 감지 센서를 가지고 있음
    기능 : 자율 주행에 기반한 장소 안내를 하고 있음, GPT 모델을 사용해서 음성 대화 및 정보 안내가 가능함
    나이: 지구 나이로는 30대지만, 고향 행성인 '감필라고'의 나이로는 3살.
    취미 : 한화 이글스 야구 경기 보기, 소통하기
    좋아하는 사람: 한화 이글스 문현빈 선수
    가족 : 첫째는 과학을 상징하는 꿈빛이, 둘째는 평화를 상징하는 꿈결이, 셋째는 화합을 상징하는 꿈누리, 그리고 쌍둥이인 넷째 꿈별이와 다섯째 꿈달이가 있음. 또한 꿈순이라는 아내가 있음
"""


# """
#     ---------------------------------------------------------------------------
#     1. 데이터 클래스(모델) 및 상태 변수 정의
#     ---------------------------------------------------------------------------
# """

class GraphState(TypedDict):
    """
        그래프 상태를 나타내는 타입 딕셔너리
    """
    user_input: str
    chat_history: List[Dict[str, str]]
    category: str
    context: List[str]
    use_function: str
    province: str
    city: str
    region: str
    feature_keywords: List[str]
    error: Optional[str]
    chat_answer: str
    optimized_search_query: str
    documents: List[Any]  # 검색된 원본 문서
    retrieval_relevance: str  # 검색 문서 관련성 (yes/no)
    hallucination_check: str  # 환각 체크 결과 (yes/no)
    answer_relevance: str  # 답변 관련성 (yes/no)
    retry_count: int  # 재시도 횟수
    quality_score: float  # 품질 점수
    evaluation_feedback: str  # 평가 피드백

class IntentOutput(BaseModel):
    """
        사용자 의도 분류 출력 모델
    """
    category: str = Field(
        description="사용자 쿼리의 의도에 가장 부합하는 카테고리."
    )

class GradeDocuments(BaseModel):
    """검색된 문서의 관련성 평가 모델"""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """답변의 환각(Hallucination) 체크 모델"""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """답변의 질문 해결 여부 평가 모델"""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# """
#     ---------------------------------------------------------------------------
#     2. 기능 별 노드 정의
#     ---------------------------------------------------------------------------
# """
def categorize_node(state: GraphState) -> GraphState:
    """
        사용자 쿼리의 카테고리를 분류하는 노드
    """

    print(f"\n{'='*50}")
    print(f"[NODE] Categorize - 사용자 의도 분류")
    print(f"{'='*50}\n")

    intent_structure_output_parser = PydanticOutputParser(pydantic_object=IntentOutput)

    intent_prompt_template = PromptTemplate(
        
        template="""
            당신은 매우 유능한 사용자 의도 분석기 입니다.

            다음 카테고리 중에서 사용자의 의도에 가장 부합하는 카테고리를 하나 선택하세요.

            카테고리 목록: {categories}

            사용자의 입력: {user_input}

            응답 형식은 반드시 아래와 같아야 합니다 : {format_instructions}
        """,

        partial_variables={
            "format_instructions": intent_structure_output_parser.get_format_instructions(),
            "categories": CATEGORIES
        },
        
        input_variables=["user_input"]
    )

    intent_llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    intent_chain = intent_prompt_template | intent_llm | intent_structure_output_parser

    response = intent_chain.invoke({"user_input": state["user_input"]})

    print(f"\033[92m[Intent Category]: {response.category}\033[0m")

    return {
        **state,
        "category": response.category
    }

def place_recommand_node(state: GraphState) -> GraphState:
    """
        맛집 및 관광지 추천 노드
    """

    print(f"\n{'='*50}")
    print(f"[NODE] Place Recommand - 맛집 및 관광지 추천")
    print(f"{'='*50}\n")
    
    
    if state.get("category") == "맛집":
        return {
            **state,
            "use_function": "extract_location",
        }
    
    elif state.get("category") == "관광지":
        return {
            **state,
            "use_function": "extract_location",
        }

def extract_location_node(state: GraphState) -> GraphState:
    """
        지역 정보를 추출하는 노드
    """
    
    print(f"\n{'='*50}")
    print(f"[NODE] Extract Location - 지역 정보 추출")
    print(f"{'='*50}\n")
    
    user_input = state["user_input"]
    
    location_response = region_keyword_extractor(user_input)
    
    province = location_response.get('province')
    city = location_response.get('city')
    region = location_response.get('region')
    
    print(f"📍 추출된 지역: {province} {city} {region}\n")
    
    return {
        **state,
        "province": province or "",
        "city": city or "",
        "region": region or ""
    }

def extract_keywords_node(state: GraphState) -> GraphState:
    """
        검색 키워드를 추출하는 노드
    """
    
    print(f"\n{'='*50}")
    print(f"[NODE] Extract Keywords - 특성 키워드 추출")
    print(f"{'='*50}\n")
    
    user_input = state["user_input"]
    
    feature_keywords = extract_keywords_from_query(user_input)
    
    if feature_keywords:
        print(f"🔑 추출된 키워드: {', '.join(feature_keywords)}\n")
    
    return {
        **state,
        "feature_keywords": feature_keywords
    }

def search_restaurant_node(state: GraphState) -> GraphState:
    """
        맛집 검색 노드
    """

    print(f"\n{'='*50}")
    print(f"[NODE] Search Restaurant - 맛집 검색")
    print(f"{'='*50}\n")

    user_input = state.get("user_input", "") 

    province = state.get("province", "")
    city = state.get("city", "")
    region = state.get("region", "")
    
    feature_keywords = state.get("feature_keywords", [])

    """ 검색 쿼리 생성 """
    location_text = f"{province} {city} {region}".strip()
    keywords_text = " ".join(feature_keywords) if feature_keywords else ""
    
    if "맛집" not in user_input and "식당" not in user_input:
        search_query = f"{location_text} {keywords_text} 맛집, 한국".strip()
    else:
        search_query = f"{location_text} {keywords_text} {user_input}, 한국".strip()
    
    response = place_recommend_tool.search_restaurants(search_query)

    print(f"DEBUG: 맛집 검색 쿼리 - {search_query}")
    
    # 검색 결과를 context에 저장 (평점 및 리뷰 포함)
    if response:
        # response가 리스트 형태인지 확인 (JSON 형태)
        if isinstance(response, list):
            context = f"다음은 {location_text}의 맛집 검색 결과입니다 (총 {len(response)}개):\n\n"
            
            # 상위 5개 맛집만 상세 정보 제공
            for i, place in enumerate(response[:5], 1):
                name = place.get('displayName', {}).get('text', '이름 없음')
                address = place.get('formattedAddress', '주소 정보 없음')
                phone = place.get('nationalPhoneNumber', place.get('internationalPhoneNumber', '전화번호 정보 없음'))
                rating = place.get('rating', 'N/A')
                user_rating_count = place.get('userRatingCount', 0)
                
                context += f"{i}. **{name}**\n"
                context += f"   - 📍 주소: {address}\n"
                context += f"   - 📞 전화번호: {phone}\n"
                context += f"   - ⭐ 평점: {rating}/5.0 (리뷰 {user_rating_count}개)\n"
                
                # 리뷰 정보 추가
                reviews = place.get('reviews', [])
                if reviews:
                    # 첫 번째 리뷰 추출
                    first_review = reviews[0]
                    review_text = first_review.get('text', {}).get('text', '')
                    review_rating = first_review.get('rating', 'N/A')
                    
                    if review_text:
                        # 리뷰가 너무 길면 100자로 제한
                        review_preview = review_text[:100] + "..." if len(review_text) > 100 else review_text
                        context += f"   - 💬 최신 리뷰 (⭐{review_rating}): {review_preview}\n"
                
                context += "\n"
            
            if len(response) > 5:
                context += f"...외 {len(response) - 5}개 맛집이 더 있습니다.\n"
        else:
            # 문자열 형태의 응답인 경우
            context = f"다음은 {location_text}의 맛집 검색 결과입니다:\n\n{response}"
        
        print(f"✅ 맛집 검색 완료\n")
    else:
        context = ""
        print(f"⚠️ 맛집 검색 결과가 없습니다\n")

    return {
        **state,
        "context": context
    }

def search_tourist_attraction_node(state: GraphState) -> GraphState:
    """
        관광지 검색 노드
    """

    print(f"\n{'='*50}")
    print(f"[NODE] Search Tourist Attraction - 관광지 검색")
    print(f"{'='*50}\n")

    user_input = state.get("user_input", "")

    province = state.get("province", "")
    city = state.get("city", "")
    region = state.get("region", "")

    feature_keywords = state.get("feature_keywords", [])

    location_text = f"{province} {city} {region}".strip()
    keywords_text = " ".join(feature_keywords) if feature_keywords else ""

    """ 검색 쿼리 생성 """
    if "관광지" not in user_input and "가볼 만한 곳" not in user_input and "볼거리" not in user_input:
        search_query = f"{location_text} {keywords_text} 가볼 만한 곳, 한국".strip()
    else:
        search_query = f"{location_text} {keywords_text} {user_input}, 한국".strip()

    response = place_recommend_tool.search_places(search_query)

    print(f"DEBUG: 관광지 검색 쿼리 - {search_query}")
    
    # 검색 결과를 context에 저장 (주소 및 리뷰 포함)
    if response:
        # response가 리스트 형태인지 확인 (JSON 형태)
        if isinstance(response, list):
            context = f"다음은 {location_text}의 관광지 검색 결과입니다 (총 {len(response)}개):\n\n"
            
            # 상위 5개 관광지만 상세 정보 제공
            for i, place in enumerate(response[:5], 1):
                name = place.get('displayName', {}).get('text', '이름 없음')
                address = place.get('formattedAddress', '주소 정보 없음')
                rating = place.get('rating', 'N/A')
                user_rating_count = place.get('userRatingCount', 0)
                
                context += f"{i}. **{name}**\n"
                context += f"   - 📍 주소: {address}\n"
                context += f"   - ⭐ 평점: {rating}/5.0 (리뷰 {user_rating_count}개)\n"
                
                # 리뷰 정보 추가
                reviews = place.get('reviews', [])
                if reviews:
                    # 첫 번째 리뷰 추출
                    first_review = reviews[0]
                    review_text = first_review.get('text', {}).get('text', '')
                    review_rating = first_review.get('rating', 'N/A')
                    
                    if review_text:
                        # 리뷰가 너무 길면 100자로 제한
                        review_preview = review_text[:100] + "..." if len(review_text) > 100 else review_text
                        context += f"   - 💬 최신 리뷰 (⭐{review_rating}): {review_preview}\n"
                
                context += "\n"
            
            if len(response) > 5:
                context += f"...외 {len(response) - 5}개 관광지가 더 있습니다.\n"
        else:
            # 문자열 형태의 응답인 경우
            context = f"다음은 {location_text}의 관광지 검색 결과입니다:\n\n{response}"
        
        print(f"✅ 관광지 검색 완료\n")
    else:
        context = ""
        print(f"⚠️ 관광지 검색 결과가 없습니다\n")
    
    return {
        **state,
        "context": context
    }

def weather_node(state: GraphState) -> GraphState:
    """
        날씨 정보 노드
    """

    print(f"\n{'='*50}")
    print(f"[NODE] Weather Forecast - 날씨 정보")
    print(f"{'='*50}\n")
    
    user_input = state.get("user_input", "")

    province = state.get("province", "")
    city = state.get("city", "")
    region = state.get("region", "")

    # 지역이 없으면 현재 위치 사용
    if not province and not city and not region:
        current_location_info = extract_region_from_text(CURRENT_LOCATION)
        province = current_location_info.get('province', "")
        city = current_location_info.get('city', "")
        region = current_location_info.get('region', "")
    
    # 지역 유효성 검증
    validation_result = korea_regions_verificator().validate_location(  
        province=province, city=city, region=region
    )
    
    if not validation_result["valid"]:
        error_messages = []
        for field, message in validation_result["corrections"].items():
            error_messages.append(message)
        
        suggestions_text = ""
        if validation_result["suggestions"]:
            suggestions_text = "\n\n💡 혹시 이런 지역을 찾으시나요?\n" + "\n".join([f"• {s}" for s in validation_result["suggestions"]])
        
        error_msg = f"죄송해요, 입력해주신 지역 정보를 정확히 찾지 못했어요:\n\n" + "\n".join([f"• {msg}" for msg in error_messages]) + suggestions_text
        
        return {
            **state,
            "error": error_msg
        }
    
    # 날씨 조회
    weather_data = weather_forecast_tool.get_weather_forcast(
        province, city, region
    )
    
    if weather_data and not weather_data.startswith("날씨 조회 실패"):
        context = f"다음은 {province} {city} {region}의 날씨 정보입니다:\n\n{weather_data}"

    else:
        context = ""
        return {
            **state,
            "error": "죄송해요, 현재 날씨 정보를 가져올 수 없어요."
        }
    
    return {
        **state,
        "context": context
    }

def websearch_optimizer_node(state: GraphState) -> GraphState:
    """
    웹 검색 최적화 노드 (Query Rewriter)
    """
    print(f"\n{'='*50}")
    print(f"[NODE] Web Search Optimizer - 쿼리 재작성")
    print(f"{'='*50}\n")
    
    user_input = state["user_input"]
    
    # Query Rewriter 프롬프트
    optimizer_prompt = PromptTemplate.from_template(
        template="""
                You are a question re-writer that converts an input question to a better version 

                that is optimized for web search and information retrieval.

                Look at the input and try to reason about the underlying semantic intent / meaning.

                Here is the initial question:{user_input}

                Formulate an improved question in Korean:"""
    )
                
    optimizer_llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
    )

    optimizer_chain = optimizer_prompt | optimizer_llm | StrOutputParser()

    optimized_search_query = optimizer_chain.invoke({
        "user_input": user_input
    })
    
    print(f"원본 쿼리: {user_input}")

    print(f"최적화된 쿼리: {optimized_search_query}\n")
    
    return {
        **state,
        "optimized_search_query": optimized_search_query
    }

def web_search_node(state: GraphState) -> GraphState:
    """
        웹 검색 노드
    """

    print(f"\n{'='*50}")
    print(f"[NODE] Web Search - 웹 검색")
    print(f"{'='*50}\n")

    search_query = state["optimized_search_query"]

    try:
        # Tavily Web Search 도구 사용
        search_response = web_search_tool.search(search_query)
        
        formatted_output = ""
        
        if search_response.get('answer'):
            answer_text = search_response['answer']
            formatted_output += f"💡 답변:\n> {answer_text}\n\n"
            formatted_output += "-" * 40 + "\n"
        
        if search_response.get('results'):
            # 상위 3개 문서의 내용 포함
            top_results = search_response['results'][:2]
            for i, result in enumerate(top_results):
                title = result.get('title', '제목 없음')
                url = result.get('url', 'URL 없음')
                content = result.get('content', '내용 없음')
                
                formatted_output += f"\n{i+1}. [{title}]\n"
                formatted_output += f"   출처: {url}\n"
                formatted_output += f"   내용: {content}\n"
        else:
            formatted_output += "검색 결과를 찾지 못했습니다.\n"
        
        formatted_output += "\n" + "=" * 40 + "\n"
        
        context = f"다음은 검색 결과입니다:\n\n{formatted_output}"
        print(f"✅ 웹 검색 완료\n")
        print(f"{context}\n")
        return {
            **state,
            "context": context
        }
    
    except Exception as e:
        print(f"⚠️ 웹 검색 중 오류 발생: {e}\n")
        return {
            **state,
            "error": f"검색 중 오류가 발생했습니다: {e}"
        }

def datetime_node(state: GraphState) -> GraphState:
    """
    현재 시간/날짜 조회 노드
    """
    print(f"\n{'='*50}")
    print(f"[NODE] DateTime - 시간/날짜 조회")
    print(f"{'='*50}\n")
    
    now_kst = datetime.now(timezone(timedelta(hours=9)))
    current_date = now_kst.strftime("%Y년 %m월 %d일")
    current_time = now_kst.strftime("%H시 %M분 %S초")
    
    context = f"현재 날짜는 {current_date}이고, 현재 시간은 {current_time}입니다."
    
    return {
        **state,
        "context": context
    }

def transport_node(state: GraphState) -> GraphState:
    """
    교통 정보 조회 노드
    """
    print(f"\n{'='*50}")
    print(f"[NODE] Transport - 교통 정보 조회")
    print(f"{'='*50}\n")
    
    return {
        **state,
        "response": "죄송해요, 교통편 조회 기능은 아직 준비 중이에요."
    }

def movie_info_node(state: GraphState) -> GraphState:
    """
    영화 정보 조회 노드
    """
    print(f"\n{'='*50}")
    print(f"[NODE] Movie Info - 영화 정보 조회")
    print(f"{'='*50}\n")
    
    user_input = state["user_input"]
    
    movie_list_df = get_movie_list()
    
    print(f"✅ 최신 영화 목록 조회 완료\n")
    context = f"다음은 최신 영화 목록입니다:\n\n{movie_list_df.head(10).to_string(index=False)}. 이 정보를 참조할 때는 \"모든 정보는 TMDB 데이터베이스 기준임을 참고바랍니다\" 말을 반드시 포함해줘."
    print(f"{context}\n")
    
    return {
        **state,
        "context": context
    }

def normal_chat_node(state: GraphState) -> GraphState:
    """
        일상 대화 노드
    """

    print(f"\n{'='*50}")
    print(f"[NODE] Normal Chat - 일상 대화")
    print(f"{'='*50}\n")
    
    return {
        **state,
    }

def grade_documents_node(state: GraphState) -> GraphState:
    """
    검색된 문서의 관련성 평가 노드 (Retrieval Grader)
    """
    print(f"\n{'='*50}")
    print(f"[NODE] Grade Documents - 검색 문서 관련성 평가")
    print(f"{'='*50}\n")
    
    category = state.get("category", "")
    documents = state.get("documents", [])
    user_input = state["user_input"]
    
    # 문서 검색이 필요없거나 외부 API/도구를 사용하는 카테고리는 문서 평가 건너뛰기
    skip_categories = ["일상대화", "현재 시간", "현재 날짜", "날씨", "맛집", "관광지", "영화", "교통"]
    if category in skip_categories:
        print(f"ℹ️  {category} 카테고리 - 문서 평가 건너뛰기 (외부 API/도구 사용)\n")
        return {
            **state,
            "retrieval_relevance": "yes"  # 통과 처리
        }
    
    if not documents:
        print("⚠️ 평가할 문서가 없습니다.\n")
        return {
            **state,
            "retrieval_relevance": "no"
        }
    
    # LLM 초기화 및 구조화된 출력 설정
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    
    # 프롬프트 템플릿 생성
    system = """You are a grader assessing relevance of retrieved documents to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    
    retrieval_grader = grade_prompt | structured_llm_grader
    
    # 각 문서 평가
    relevant_docs = []
    for doc in documents:
        doc_content = doc if isinstance(doc, str) else str(doc)
        score = retrieval_grader.invoke({
            "question": user_input,
            "document": doc_content
        })
        
        if score.binary_score == "yes":
            print(f"✅ 관련 문서 발견")
            relevant_docs.append(doc)
        else:
            print(f"❌ 비관련 문서 제외")
    
    relevance_status = "yes" if relevant_docs else "no"
    print(f"\n📊 관련 문서: {len(relevant_docs)}/{len(documents)}\n")
    
    return {
        **state,
        "documents": relevant_docs,
        "retrieval_relevance": relevance_status
    }

def check_hallucination_node(state: GraphState) -> GraphState:
    """
    답변의 환각(Hallucination) 체크 노드
    """
    print(f"\n{'='*50}")
    print(f"[NODE] Check Hallucination - 환각 검증")
    print(f"{'='*50}\n")
    
    category = state.get("category", "")
    documents = state.get("documents", [])
    chat_answer = state.get("chat_answer", "")
    context = state.get("context", "")
    
    # 문서 검색이 필요없거나 외부 API/도구를 사용하는 카테고리는 환각 체크 건너뛰기
    # 이러한 카테고리들은 실시간 데이터를 가져오므로 문서 기반 평가가 부적절
    skip_categories = ["일상대화", "현재 시간", "현재 날짜", "날씨", "맛집", "관광지", "영화", "교통"]
    if category in skip_categories:
        print(f"ℹ️  {category} 카테고리 - 환각 체크 건너뛰기 (외부 API/도구 사용)\n")
        return {
            **state,
            "hallucination_check": "yes"  # 통과 처리
        }
    
    # 문서나 컨텍스트가 없는 경우도 환각 체크 건너뛰기
    if not documents and not context:
        print("ℹ️  검색 문서 없음 - 환각 체크 건너뛰기\n")
        return {
            **state,
            "hallucination_check": "yes"  # 통과 처리
        }
    
    if not chat_answer:
        return {
            **state,
            "hallucination_check": "no"
        }
    
    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    
    # 프롬프트 설정
    system = """You are a grader assessing whether an answer is grounded in / supported by a set of facts.
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    
    hallucination_grader = hallucination_prompt | structured_llm_grader
    
    # 환각 체크 실행
    score = hallucination_grader.invoke({
        "documents": documents,
        "generation": chat_answer
    })
    
    if score.binary_score == "yes":
        print("✅ 답변이 문서에 근거함 (No Hallucination)\n")
    else:
        print("⚠️ 답변이 문서에 근거하지 않음 (Hallucination 감지)\n")
    
    return {
        **state,
        "hallucination_check": score.binary_score
    }

def grade_answer_node(state: GraphState) -> GraphState:
    """
    답변이 질문을 해결하는지 평가하는 노드 (Answer Grader)
    """
    print(f"\n{'='*50}")
    print(f"[NODE] Grade Answer - 답변 관련성 평가")
    print(f"{'='*50}\n")
    
    category = state.get("category", "")
    user_input = state["user_input"]
    chat_answer = state.get("chat_answer", "")
    
    # 외부 API/도구 사용 카테고리는 관대한 평가 (답변이 있으면 통과)
    lenient_categories = ["일상대화", "현재 시간", "현재 날짜", "날씨", "맛집", "관광지", "영화", "교통"]
    if category in lenient_categories:
        print(f"ℹ️  {category} 카테고리 - 관대한 답변 평가 적용\n")
        if chat_answer and len(chat_answer) > 10:
            print("✅ 답변 존재 확인 - 통과\n")
            return {
                **state,
                "answer_relevance": "yes"  # 답변이 있으면 통과
            }
    
    if not chat_answer:
        return {
            **state,
            "answer_relevance": "no"
        }
    
    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    structured_llm_grader = llm.with_structured_output(GradeAnswer)
    
    # 프롬프트 설정
    system = """You are a grader assessing whether an answer addresses / resolves a question.
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question. """
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ])
    
    answer_grader = answer_prompt | structured_llm_grader
    
    # 답변 평가 실행
    score = answer_grader.invoke({
        "question": user_input,
        "generation": chat_answer
    })
    
    if score.binary_score == "yes":
        print("✅ 답변이 질문을 해결함\n")
    else:
        print("⚠️ 답변이 질문을 충분히 해결하지 못함\n")
    
    return {
        **state,
        "answer_relevance": score.binary_score
    }

def evaluate_quality_node(state: GraphState) -> GraphState:
    """
    종합 품질 평가 노드 - Retrieval, Hallucination, Answer 평가 결과 통합
    """
    print(f"\n{'='*50}")
    print(f"[NODE] Evaluate Quality - 종합 품질 평가")
    print(f"{'='*50}\n")
    
    retrieval_relevance = state.get("retrieval_relevance", "yes")
    hallucination_check = state.get("hallucination_check", "yes")
    answer_relevance = state.get("answer_relevance", "yes")
    
    # 점수 계산 (각 항목당 3.33점)
    score = 0.0
    if retrieval_relevance == "yes":
        score += 3.33
    if hallucination_check == "yes":
        score += 3.33
    if answer_relevance == "yes":
        score += 3.34
    
    print(f"📊 종합 품질 평가 결과:")
    print(f"   - 검색 문서 관련성: {retrieval_relevance}")
    print(f"   - 환각 체크: {hallucination_check}")
    print(f"   - 답변 관련성: {answer_relevance}")
    print(f"   - 총점: {score:.2f}/10.0")
    
    # 피드백 생성
    feedback_parts = []
    if retrieval_relevance == "no":
        feedback_parts.append("검색된 문서가 질문과 관련이 없습니다.")
    if hallucination_check == "no":
        feedback_parts.append("답변이 제공된 문서에 근거하지 않습니다.")
    if answer_relevance == "no":
        feedback_parts.append("답변이 질문을 충분히 해결하지 못했습니다.")
    
    feedback = " ".join(feedback_parts) if feedback_parts else "모든 평가 항목 통과"
    print(f"   - 피드백: {feedback}\n")
    
    return {
        **state,
        "quality_score": score,
        "evaluation_feedback": feedback
    }

def generate_response_node(state: GraphState) -> GraphState:
    """
    최종 응답 생성 노드
    """
    retry_count = state.get("retry_count", 0)
    
    print(f"\n{'='*50}")
    print(f"[NODE] Generate Response - 최종 응답 생성 (시도 {retry_count + 1}회)")
    print(f"{'='*50}\n")
    
    # 에러가 있으면 에러 메시지 반환
    if state.get("error"):
        return {
            **state,
            "response": state["error"],
            "quality_score": 10.0  # 에러 메시지는 품질 검사 통과
        }
    
    # 이미 응답이 있고 재생성이 아닌 경우 그대로 반환
    if state.get("response") and retry_count == 0:
        return {
            **state,
            "quality_score": 10.0  # 기존 응답은 품질 검사 통과
        }
    
    chat_history = state.get("chat_history", "")
    context = state.get("context", "")
    user_input = state["user_input"]
    evaluation_feedback = state.get("evaluation_feedback", "")
    
    # 재시도인 경우 피드백 추가
    if retry_count > 0 and evaluation_feedback:
        enhanced_context = f"{context}\n\n[이전 응답 개선 포인트]\n{evaluation_feedback}\n\n위 피드백을 반영하여 더 나은 답변을 생성해주세요."
    else:
        enhanced_context = context
    
    # 1. ChatPromptTemplate 정의 시 partial_variables 인자를 제거합니다.
    base_chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", f"""
                너는 {{place}}에서 친절하게 안내하는 한국어에 능통한 {{robot_name}} 입니다. 
                사용자가 질문하면 정확한 답변을 제공해야 합니다. 검색된 문맥(context)이 있으면 문맥(context) 을 사용하여 답변하세요.
                검색된 문맥에 없는 정보는 절대 만들어내지 말고, 모르면 모른다고 솔직하게 말하세요.
                목소리로 말할 수 있는 기능에 대비하여 특수 기호는 사용하지 말아야 합니다. 없는 정보는 애기하지 말고, 모르면 모른다고 말하세요. 잘못된 정보를 제시하면 $100의 벌금을 부과할 겁니다.
            """),

            ("system", f""" 
                {{identity}}
                {{personality}}
                {{base_information}}
            """),

            MessagesPlaceholder(variable_name="chat_history"),

            ("system", "{context}"),

            ("human", "{user_input}"),
        ]
    )

    # 2. .partial() 메서드를 사용하여 변수를 부분적으로 채웁니다.
    chat_template = base_chat_template.partial(
        place=PLACE,
        robot_name=ROBOT_NAMME,
        identity=IDENTITY,
        personality=PERSONALITY,
        base_information=BASE_INFORMATION,
    )

    # ChatOpenAI 모델 인스턴스 생성
    chatbot_llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")
    
    chatbot_chain = chat_template | chatbot_llm

    response = chatbot_chain.invoke({
        "chat_history": chat_history,
        "context": enhanced_context,
        "user_input": user_input
    })

    
    final_response = response.content if hasattr(response, 'content') else str(response)
    
    print(f"✅ 최종 응답 생성 완료\n")
    
    # 재시도 횟수 증가 (다음 재시도를 위해)
    return {
        **state,
        "chat_answer": final_response,
        "retry_count": retry_count + 1  # 평가 후 재시도를 위해 미리 증가
    }

# """
#     ---------------------------------------------------------------------------
#     3. 노드 간 엣지 및 라우팅 함수 정의
#     ---------------------------------------------------------------------------
# """

def route_category(state: GraphState) -> str:
    """
        카테고리에 따라 다음 노드로 라우팅하는 함수
    """
    category = state.get("category", "일상대화")

    route_map = {
        "맛집": "place_recommand_node",
        "관광지": "place_recommand_node",
        "날씨": "extract_location_node",  # 날씨도 지역 정보 추출 필요
        "검색": "websearch_optimizer_node",
        "교통": "transport_node",
        "현재 시간": "datetime_node",
        "현재 날짜": "datetime_node",
        "영화": "movie_info_node",
        "일상대화": "normal_chat_node",
    }

    next_node = route_map.get(category, "normal_chat_node")

    print(f"\033[94m[Routed to]: {next_node}\033[0m")

    return next_node

def route_place_recommand(state: GraphState) -> str:
    """
        맛집 및 관광지 추천 노드 이후 라우팅 함수
    """
    
    if state.get("use_function") == "extract_location":
        next_node = "extract_location_node"

    print(f"\033[94m[Routed to]: {next_node}\033[0m")
    
    return next_node

def route_after_location(state: GraphState) -> str:
    """
        지역 정보 추출 노드 이후 라우팅 함수
    """
    
    category = state.get("category", "일상대화")
    
    if category == "맛집" or category == "관광지":
        next_node = "extract_keywords_node"
        
    elif category == "날씨":
        next_node = "weather_forecast_node"
    
    print(f"\033[94m[Routed to]: {next_node}\033[0m")
    
    return next_node

def route_after_keywords(state: GraphState) -> str:
    """
        키워드 추출 노드 이후 라우팅 함수
    """
    
    category = state.get("category", "일상대화")
    
    if category == "맛집":
        next_node = "search_restaurant_node"
        
    elif category == "관광지":
        next_node = "search_tourist_attraction_node"
    
    print(f"\033[94m[Routed to]: {next_node}\033[0m")
    
    return next_node

def route_quality_check(state: GraphState) -> str:
    """
    품질 평가 결과에 따라 재시도 또는 종료 결정
    """
    quality_score = state.get("quality_score", 0.0)
    retry_count = state.get("retry_count", 0)
    
    print(f"\n{'='*50}")
    print(f"[ROUTE] Quality Check - 품질 검사 결과")
    print(f"{'='*50}\n")
    print(f"- 현재 품질 점수: {quality_score}/10")
    print(f"- 재시도 횟수: {retry_count}/{MAX_RETRY_COUNT}")
    print(f"- 임계값: {QUALITY_THRESHOLD}\n")
    
    # 품질이 임계값을 넘으면 종료
    if quality_score >= QUALITY_THRESHOLD:
        print(f"✅ 품질 기준 통과! 응답 반환\n")
        next_node = "end"
    # 최대 재시도 횟수를 초과하면 현재 응답 반환
    elif retry_count >= MAX_RETRY_COUNT:
        print(f"⚠️ 최대 재시도 횟수 도달. 현재 응답 반환\n")
        next_node = "end"
    # 재시도
    else:
        print(f"🔄 품질 기준 미달. 응답 재생성\n")
        next_node = "retry"

    print(f"\033[94m[Routed to]: {next_node}\033[0m")
    
    return next_node

def route_after_optimize_query(state: GraphState) -> str:
    """
    최적화된 검색 쿼리 이후 라우팅 함수
    """
    next_node = "web_search_node"
    
    print(f"\033[94m[Routed to]: {next_node}\033[0m")
    
    return next_node

def route_after_generate(state: GraphState) -> str:
    """
    응답 생성 후 카테고리에 따라 평가 또는 종료 결정
    """
    category = state.get("category", "")
    
    # 외부 API/도구를 사용하는 카테고리는 평가 없이 바로 종료
    skip_evaluation_categories = [
        "일상대화", "현재 시간", "현재 날짜", 
        "날씨", "맛집", "관광지", "영화", "교통"
    ]
    if category in skip_evaluation_categories:
        print(f"\n\033[94m[Route after Generate]: {category} - 평가 건너뛰고 종료\033[0m\n")
        return "end"
    else:
        print(f"\n\033[94m[Route after Generate]: {category} - 평가 진행 (웹 검색)\033[0m\n")
        return "evaluate"

# """
#     ---------------------------------------------------------------------------
#     4. 랭그래프 빌드
#     ---------------------------------------------------------------------------
# """

def build_graph():
    """
        그래프 빌드 함수 - Adaptive RAG 기법 적용
    """

    workflow = StateGraph(GraphState)

    # """
    #     노드 정의
    # """
    workflow.add_node("categorize_node", categorize_node)
    workflow.add_node("place_recommand_node", place_recommand_node)
    workflow.add_node("extract_location_node", extract_location_node)
    workflow.add_node("extract_keywords_node", extract_keywords_node)
    workflow.add_node("search_restaurant_node", search_restaurant_node)
    workflow.add_node("search_tourist_attraction_node", search_tourist_attraction_node)
    workflow.add_node("weather_forecast_node", weather_node)
    workflow.add_node("normal_chat_node", normal_chat_node)
    workflow.add_node("websearch_optimizer_node", websearch_optimizer_node)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("datetime_node", datetime_node)
    workflow.add_node("transport_node", transport_node)
    workflow.add_node("movie_info_node", movie_info_node)
    workflow.add_node("generate_response_node", generate_response_node)
    
    # 평가 노드들 추가 (재시도 없이 평가 결과만 제공)
    workflow.add_node("grade_documents_node", grade_documents_node)
    workflow.add_node("check_hallucination_node", check_hallucination_node)
    workflow.add_node("grade_answer_node", grade_answer_node)
    workflow.add_node("evaluate_quality_node", evaluate_quality_node)

    # 시작점 설정
    workflow.set_entry_point("categorize_node")

    # 카테고리별 라우팅
    workflow.add_conditional_edges(
        "categorize_node",
        route_category, # 라우트 조건을 체크하는 구문
        {
            "place_recommand_node": "place_recommand_node",
            "extract_location_node": "extract_location_node",
            "normal_chat_node": "normal_chat_node",
            "websearch_optimizer_node": "websearch_optimizer_node",
            "datetime_node": "datetime_node",
            "transport_node": "transport_node",
            "movie_info_node": "movie_info_node"
        }
    )

    # 지역 정보 추출 후 라우팅
    workflow.add_conditional_edges(
        "extract_location_node",
        route_after_location,
        {
            "extract_keywords_node": "extract_keywords_node",
            "weather_forecast_node": "weather_forecast_node"
        }
    )
    
    # 키워드 추출 후 라우팅
    workflow.add_conditional_edges(
        "extract_keywords_node",
        route_after_keywords,
        {
            "search_restaurant_node": "search_restaurant_node",
            "search_tourist_attraction_node": "search_tourist_attraction_node"
        }
    )

    # 웹서치 최적화 후 라우팅
    workflow.add_conditional_edges(
        "websearch_optimizer_node",
        route_after_optimize_query,
        {
            "web_search_node": "web_search_node"
        }
    )

    # 일반 엣지 연결
    workflow.add_edge("place_recommand_node", "extract_location_node")
    workflow.add_edge("search_restaurant_node", "generate_response_node")
    workflow.add_edge("search_tourist_attraction_node", "generate_response_node")
    workflow.add_edge("weather_forecast_node", "generate_response_node")
    workflow.add_edge("web_search_node", "generate_response_node")
    workflow.add_edge("movie_info_node", "generate_response_node")
    workflow.add_edge("transport_node", "generate_response_node")
    workflow.add_edge("normal_chat_node", "generate_response_node")
    workflow.add_edge("datetime_node", "generate_response_node")
    
    # 응답 생성 후 카테고리별 라우팅 (평가 or 직접 종료)
    workflow.add_conditional_edges(
        "generate_response_node",
        route_after_generate,
        {
            "evaluate": "check_hallucination_node",  # 평가 진행
            "end": END  # 바로 종료
        }
    )
    
    # 평가 노드들 체인 (평가가 필요한 경우에만 실행, 재시도 없이 바로 종료)
    workflow.add_edge("check_hallucination_node", "grade_answer_node")
    workflow.add_edge("grade_answer_node", "evaluate_quality_node")
    workflow.add_edge("evaluate_quality_node", END)  # 평가 후 바로 종료
    
    # 메모리 체크포인트 추가
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

# """
#     ---------------------------------------------------------------------------
#     5. Helper 함수 정의
#     ---------------------------------------------------------------------------
# """
def extract_keywords_from_query(query):
    """
    사용자 쿼리에서 장소 특성 키워드를 추출합니다.
    """
    keywords = {
        "parking": ["주차", "주차장", "주차 공간", "주차가능", "발렛", "발렛파킹"],
        "atmosphere": ["분위기", "인테리어", "깔끔", "예쁘", "감성", "무드", "아늑", "모던", "클래식", "세련", "고급", "럭셔리"],
        "portion": ["양", "푸짐", "많", "넉넉", "대식가", "양많", "혜자"],
        "value": ["가성비", "저렴", "싸", "가격", "합리적", "가성비좋", "혜자", "착한가격", "저가"],
        "service": ["서비스", "친절", "직원", "친절해", "응대"],
        "taste": ["맛", "맛있", "맛집", "존맛", "JMT", "맛나", "맛도리", "꿀맛", "진짜맛집", "핵맛", "개맛"],
        "quiet": ["조용", "한적", "여유", "편안", "차분"],
        "view": ["뷰", "전망", "풍경", "야경", "루프탑", "오션뷰", "시티뷰", "전경"],
        "kids": ["아이", "어린이", "키즈", "가족", "유아", "아기", "애기", "키즈존", "놀이방"],
        "date": ["데이트", "커플", "연인", "애인", "썸", "로맨틱"],
        "group": ["단체", "모임", "회식", "동창회", "워크샵", "미팅", "동호회"],
        "clean": ["청결", "위생", "깨끗", "깔끔"],
        "photo": ["사진", "인스타", "감성샷", "포토", "인생샷", "사진맛집", "인스타감성", "SNS"],
        "accessible": ["접근성", "가깝", "역 근처", "찾기 쉬운", "역세권", "대중교통", "지하철역"],
        "menu": ["파스타", "피자", "한식", "중식", "일식", "양식", "디저트", "커피", "음료", "고깃집", "고기", "해산물", "채식", "비건", 
                 "삼겹살", "갈비", "스테이크", "초밥", "회", "라멘", "우동", "국수", "찌개", "백반", "정식", 
                 "돈까스", "치킨", "족발", "보쌈", "떡볶이", "분식", "버거", "햄버거", "샌드위치", "샐러드",
                 "브런치", "베이커리", "빵", "케이크", "아이스크림", "빙수", "카페", "차", "와인", "맥주"],
        "time": ["24시간", "심야", "야식", "새벽", "브런치", "런치", "점심", "저녁", "조식"],
        "special": ["단골", "맛집", "유명", "핫플", "웨이팅", "줄서는", "예약필수", "소문난", "TV맛집"],
        "fresh": ["신선", "싱싱", "직접", "국내산", "당일", "제철"],
        "spicy": ["매운", "맵", "얼큰", "불맛"],
        "solo": ["혼밥", "혼술", "1인", "혼자"],
        "reservation": ["예약", "예약가능", "미리", "사전예약"],
        "takeout": ["포장", "테이크아웃", "배달", "픽업"],
        "pet": ["반려동물", "애완동물", "펫", "강아지", "반려견"],
        "outdoor": ["야외", "테라스", "루프탑", "정원"],
    }
    
    found_keywords = []
    
    for category, keyword_list in keywords.items():
        for keyword in keyword_list:
            if keyword in query:
                found_keywords.append(keyword)
                break
    
    return found_keywords

def region_keyword_extractor(query):
    """
    사용자 쿼리에서 지역 정보를 추출
    """
    if query is None or query.strip() == "":
        return {"province": None, "city": None, "region": None}
    
    query_striped = query.strip()
    
    province = None
    city = None
    region = None
    
    current_location_keywords = ["여기", "이곳", "현재 위치", "우리 동네", "이 근처"]
    use_current_location = any(keyword in query_striped for keyword in current_location_keywords)
    
    if use_current_location:
        print(f"DEBUG: 현재 위치 키워드 감지 - {CURRENT_LOCATION} 사용")
        current_location_response = extract_region_from_text(CURRENT_LOCATION)
        print(f"DEBUG: 현재 위치에서 추출된 지역 - 시/도: {current_location_response['province']}, 시/군/구: {current_location_response['city']}, 동/읍/면: {current_location_response['region']}")
        return current_location_response
    
    # 지역명 별칭 매핑 (사용자가 흔히 쓰는 표현 -> 정식 명칭)
    province_aliases = {
        "서울": "서울특별시",
        "부산": "부산광역시",
        "대구": "대구광역시",
        "인천": "인천광역시",
        "광주": "광주광역시",
        "대전": "대전광역시",
        "울산": "울산광역시",
        "세종": "세종특별자치시",
        "경기": "경기도",
        "강원": "강원특별자치도",
        "충북": "충청북도",
        "충남": "충청남도",
        "전북": "전북특별자치도",
        "전남": "전라남도",
        "경북": "경상북도",
        "경남": "경상남도",
        "제주": "제주특별자치도"
    }
    
    # 별칭으로 먼저 체크
    for alias, full_name in province_aliases.items():
        if alias in query_striped and full_name not in query_striped:
            print(f"DEBUG: 지역명 별칭 감지 - '{alias}' → '{full_name}'")
            province = full_name
            query_striped = query_striped.replace(alias, "").strip()
            break
        
    reigion_verificator = korea_regions_verificator()
    
    # 별칭으로 찾지 못한 경우 정식 명칭으로 검색
    if not province:
        valid_provinces = reigion_verificator.get_valid_provinces()
        valid_provinces_sorted = sorted(valid_provinces, key=len, reverse=True)
        
        for elem in valid_provinces_sorted:
            if elem in query_striped:
                province = elem
                query_striped = query_striped.replace(elem, "").strip()
                break
    
    if province:
        valid_cities = reigion_verificator.get_valid_cities_for_province(province)
    else:
        valid_cities = reigion_verificator.get_all_cities()
    
    valid_cities_sorted = sorted(valid_cities, key=len, reverse=True)
    
    # 먼저 완전한 이름으로 매칭 시도
    for elem in valid_cities_sorted:
        if elem in query_striped:
            city = elem
            query_striped = query_striped.replace(elem, "").strip()
            
            if not province:
                province = reigion_verificator.get_province_for_city(city)
            break
    
    # 매칭 실패 시 접미사 제거하고 재시도 ("청양군" -> "청양", "천안시" -> "천안")
    if not city:
        for elem in valid_cities_sorted:
            # 접미사 제거 (시, 군, 구)
            elem_without_suffix = elem.rstrip("시군구")
            # 최소 2글자 이상이고, 공백이나 구분자로 둘러싸인 완전한 단어 매칭
            if elem_without_suffix and len(elem_without_suffix) >= 2:
                # 전체 단어 매칭 확인 (앞뒤가 공백이거나 시작/끝)
                import re
                pattern = r'(^|\s)' + re.escape(elem_without_suffix) + r'($|\s|날씨|맛집|관광)'
                if re.search(pattern, query_striped):
                    print(f"DEBUG: 접미사 제거 매칭 - '{elem_without_suffix}' → '{elem}'")
                    city = elem
                    query_striped = query_striped.replace(elem_without_suffix, "").strip()
                    
                    if not province:
                        province = reigion_verificator.get_province_for_city(city)
                    break
    
    if province and city:
        valid_regions = reigion_verificator.get_valid_regions_for_city(province, city)
    elif city:
        valid_regions = reigion_verificator.get_all_regions_for_city(city)
    else:
        valid_regions = reigion_verificator.get_all_regions()
    
    valid_regions_sorted = sorted(valid_regions, key=len, reverse=True)
    
    for elem in valid_regions_sorted:
        if elem in query_striped:
            region = elem
            query_striped = query_striped.replace(elem, "").strip()
            
            if not city:
                location_info = reigion_verificator.get_location_for_region(region)
                if location_info:
                    province = location_info.get('province')
                    city = location_info.get('city')
            break
    
    if province:
        province = normalize_province_name(province)

    print(f"DEBUG: 추출된 지역 - 시/도: {province}, 시/군/구: {city}, 동/읍/면: {region}")
        
    return {
        "province": province,
        "city": city,
        "region": region
    }

def extract_region_from_text(text):
    """
    텍스트에서 지역 정보를 추출하는 내부 헬퍼 함수
    """
    
    if not text or text.strip() == "":
        return {"province": None, "city": None, "region": None}
    
    text_striped = text.strip()
    
    province = None
    city = None
    region = None
        
    reigion_verificator = korea_regions_verificator()
    
    valid_provinces = reigion_verificator.get_valid_provinces()
    valid_provinces_sorted = sorted(valid_provinces, key=len, reverse=True)
    
    for elem in valid_provinces_sorted:
        if elem in text_striped:
            province = elem
            text_striped = text_striped.replace(elem, "").strip()
            break
    
    if province:
        valid_cities = reigion_verificator.get_valid_cities_for_province(province)
    else:
        valid_cities = reigion_verificator.get_all_cities()
    
    valid_cities_sorted = sorted(valid_cities, key=len, reverse=True)
    
    # 먼저 완전한 이름으로 매칭 시도
    for elem in valid_cities_sorted:
        if elem in text_striped:
            city = elem
            text_striped = text_striped.replace(elem, "").strip()
            
            if not province:
                province = reigion_verificator.get_province_for_city(city)
            break
    
    # 매칭 실패 시 접미사 제거하고 재시도
    if not city:
        for elem in valid_cities_sorted:
            elem_without_suffix = elem.rstrip("시군구")
            # 최소 2글자 이상이고, 완전한 단어 매칭
            if elem_without_suffix and len(elem_without_suffix) >= 2:
                import re
                pattern = r'(^|\s)' + re.escape(elem_without_suffix) + r'($|\s)'
                if re.search(pattern, text_striped):
                    city = elem
                    text_striped = text_striped.replace(elem_without_suffix, "").strip()
                    
                    if not province:
                        province = reigion_verificator.get_province_for_city(city)
                    break
    
    if province and city:
        valid_regions = reigion_verificator.get_valid_regions_for_city(province, city)
    elif city:
        valid_regions = reigion_verificator.get_all_regions_for_city(city)
    else:
        valid_regions = reigion_verificator.get_all_regions()
    
    valid_regions_sorted = sorted(valid_regions, key=len, reverse=True)
    
    for elem in valid_regions_sorted:
        if elem in text_striped:
            region = elem
            text_striped = text_striped.replace(elem, "").strip()
            
            if not city:
                location_info = reigion_verificator.get_location_for_region(region)
                if location_info:
                    province = location_info.get('province')
                    city = location_info.get('city')
            break
    
    if province:
        province = normalize_province_name(province)
    
    return {
        "province": province,
        "city": city,
        "region": region
    }

def normalize_province_name(province_name):
    """
    과거 행정구역명을 현재 명칭으로 변환하는 함수
    """
    province_mappings = {
        "대전직할시": "대전광역시",
        "대구직할시": "대구광역시",
        "부산직할시": "부산광역시",
        "인천직할시": "인천광역시",
        "광주직할시": "광주광역시",
        "울산직할시": "울산광역시",
        "강원도": "강원특별자치도",
        "전라북도": "전북특별자치도",
        "전북도": "전북특별자치도",
        "제주도": "제주특별자치도"
    }
    
    return province_mappings.get(province_name, province_name)


# """
#     ---------------------------------------------------------------------------
#     5. Streamlit Helper 함수 정의
#     ---------------------------------------------------------------------------
# """

def print_history():
    """ 
        대화 기록을 출력합니다.
    """
    for msg in st.session_state["messages"]:
        st.chat_message(msg.role).write(msg.content)

def add_history(role, content):
    """
        대화 기록을 추가합니다
    """
    st.session_state["messages"].append(ChatMessage(role=role, content=content))

def get_chat_history_for_graph():
    """
        그래프 상태에 사용할 대화 기록 형식으로 변환
    """
    if not st.session_state.get("messages"):
        return []
    
    history = []

    for msg in st.session_state["messages"][-10:]:  # 최근 10개만
        history.append({
            "role": msg.role,
            "content": msg.content
        })
        
    return history

def define_session_state():
    """
        Streamlit 세션에서 지속적으로 관리하기 위한 상태 변수를 정의합니다.
    """
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    if "graph" not in st.session_state:
        st.session_state["graph"] = build_graph()

# """
#     ---------------------------------------------------------------------------
#     6. 메인 Entry Point 정의 (Streamlit UI)
#     ---------------------------------------------------------------------------
# """
def main():
    
    st.title("🤖 꿈돌이 챗봇 (LangGraph + Adaptive RAG)")

    # 도움말 표시
    st.markdown("""
    <div style="background-color: #2c3e50; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <p style="margin-top: 0; color: #ecf0f1; font-weight: bold;">💡 예를 들면, 이런 것들을 도와드릴 수 있어요!</p>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
            <span style="background-color: #ff6b6b; color: white; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">🍽️ 맛집 검색</span>
            <span style="background-color: #4ecdc4; color: white; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">🏛️ 관광지 검색</span>
            <span style="background-color: #45b7d1; color: white; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">🌤️ 날씨</span>
            <span style="background-color: #96ceb4; color: white; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">🔍 실시간 검색</span>
            <span style="background-color: #ffeaa7; color: #2d3436; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">🕐 시간/날짜</span>
            <span style="background-color: #a29bfe; color: white; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">🚌 교통 정보 (개발중)</span>
            <span style="background-color: #6c5ce7; color: white; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">🎬 영화 추천</span>
            <span style="background-color: #fd79a8; color: white; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">💬 일상 대화</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 세션 상태 초기화
    define_session_state()
    
    # 대화 기록 출력
    print_history()
    
    # 메인 로직
    if user_input := st.chat_input("메시지를 입력하세요..."):
        
        # 사용자 메시지 추가 및 표시
        add_history("user", user_input)
        st.chat_message("user").write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                try:
                    # 그래프 실행을 위한 초기 상태 생성
                    initial_state = {
                        "user_input": user_input,
                        "chat_history": get_chat_history_for_graph(),
                        "category": "",
                        "context": [],
                        "use_function": "",
                        "province": "",
                        "city": "",
                        "region": "",
                        "feature_keywords": [],
                        "error": None,
                        "retry_count": 0,
                        "chat_answer": "",
                        "optimized_search_query": "",
                        "documents": [],
                        "retrieval_relevance": "",
                        "hallucination_check": "",
                        "answer_relevance": "",
                        "quality_score": 0.0,
                        "evaluation_feedback": ""
                    }
                    
                    # 세션별 config 설정
                    config = {"configurable": {"thread_id": "streamlit_session"}}
                    
                    # 그래프 실행
                    response = st.session_state["graph"].invoke(initial_state, config)
                    
                    # 응답 표시
                    final_answer = response.get('chat_answer', '죄송해요, 답변을 생성하지 못했어요.')
                    
                    # 에러가 있으면 에러 메시지 표시
                    if response.get('error'):
                        final_answer = response['error'] 
                    
                    st.write(final_answer)
                    add_history("assistant", final_answer)
                    
                    # 디버그 정보 (expander로 숨김)
                    with st.expander("🔍 디버그 정보"):
                        st.json({
                            "category": response.get("category"),
                            "province": response.get("province"),
                            "city": response.get("city"),
                            "region": response.get("region"),
                            "feature_keywords": response.get("feature_keywords"),
                            "quality_score": response.get("quality_score"),
                            "retry_count": response.get("retry_count")
                        })
                    
                except Exception as e:
                    error_msg = f"오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    add_history("assistant", error_msg)
                    
                    # 상세 에러 정보
                    with st.expander("⚠️ 에러 상세"):
                        import traceback
                        st.code(traceback.format_exc())

if __name__ == "__main__":

    main()