"""
Kumdori Chatbot 테스트 스크립트
개선된 Adaptive RAG 기능 테스트
"""

import sys
import os

# 현재 디렉토리를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kumdori_agent_chatbot.kumdori_agent_chatbot.kumdori_chatbot_node_langgraph import build_graph
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def test_chatbot():
    """챗봇 기본 기능 테스트"""
    
    print("=" * 60)
    print("Kumdori Chatbot - Adaptive RAG 테스트")
    print("=" * 60)
    print()
    
    # 그래프 빌드
    print("🔧 그래프 빌드 중...")
    app = build_graph()
    print("✅ 그래프 빌드 완료\n")
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "맛집 추천 (지역 + 특성)",
            "input": "대전 유성구에서 주차 가능한 맛있는 떡볶이 집 추천해줘",
            "expected_category": "맛집"
        },
        {
            "name": "관광지 추천",
            "input": "서울에서 데이트하기 좋은 곳 알려줘",
            "expected_category": "관광지"
        },
        {
            "name": "날씨 조회",
            "input": "대전 날씨 어때?",
            "expected_category": "날씨"
        },
        {
            "name": "웹 검색",
            "input": "2024년 AI 트렌드에 대해 알려줘",
            "expected_category": "검색"
        },
        {
            "name": "일상 대화",
            "input": "안녕! 너는 누구야?",
            "expected_category": "일상대화"
        }
    ]
    
    for idx, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"테스트 {idx}: {test_case['name']}")
        print(f"{'='*60}\n")
        print(f"📝 질문: {test_case['input']}")
        print(f"🎯 예상 카테고리: {test_case['expected_category']}\n")
        
        # 설정
        config = {"configurable": {"thread_id": f"test_{idx}"}}
        
        # 입력 구성
        inputs = {
            "user_input": test_case['input'],
            "chat_history": [],
            "category": "",
            "context": [],
            "use_function": "",
            "province": "",
            "city": "",
            "region": "",
            "feature_keywords": [],
            "error": None,
            "chat_answer": "",
            "optimized_search_query": "",
            "documents": [],
            "retrieval_relevance": "",
            "hallucination_check": "",
            "answer_relevance": "",
            "retry_count": 0,
            "quality_score": 0.0,
            "evaluation_feedback": ""
        }
        
        try:
            # 그래프 실행 (스트리밍)
            print("🚀 챗봇 실행 중...\n")
            
            for output in app.stream(inputs, config):
                # 각 노드의 출력을 간략히 표시
                for node_name, node_output in output.items():
                    if node_name == "categorize_node":
                        print(f"✓ 카테고리 분류: {node_output.get('category', 'N/A')}")
                    elif node_name == "generate_response_node":
                        print(f"✓ 답변 생성 완료")
                    elif node_name == "check_hallucination_node":
                        print(f"✓ 환각 체크: {node_output.get('hallucination_check', 'N/A')}")
                    elif node_name == "grade_answer_node":
                        print(f"✓ 답변 관련성: {node_output.get('answer_relevance', 'N/A')}")
                    elif node_name == "evaluate_quality_node":
                        score = node_output.get('quality_score', 0)
                        print(f"✓ 품질 평가: {score:.2f}/10.0")
            
            # 최종 상태 확인
            final_state = app.get_state(config).values
            
            print(f"\n{'='*60}")
            print("📊 최종 결과")
            print(f"{'='*60}")
            print(f"카테고리: {final_state.get('category', 'N/A')}")
            print(f"품질 점수: {final_state.get('quality_score', 0):.2f}/10.0")
            print(f"재시도 횟수: {final_state.get('retry_count', 0)}")
            print(f"\n💬 최종 답변:\n{final_state.get('chat_answer', 'N/A')}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            import traceback
            traceback.print_exc()
        
        # 다음 테스트 전 대기
        if idx < len(test_cases):
            input("\n⏸️  다음 테스트를 진행하려면 Enter를 누르세요...")


def test_quality_retry():
    """품질 평가 및 재시도 메커니즘 테스트"""
    
    print("\n" + "=" * 60)
    print("품질 평가 및 재시도 메커니즘 테스트")
    print("=" * 60 + "\n")
    
    app = build_graph()
    
    # 복잡한 질문으로 재시도 유도
    test_input = "대전 유성구에서 가족과 함께 가기 좋고, 주차 가능하며, 분위기 좋은 한식당 추천해줘. 가격대도 알려줘."
    
    print(f"📝 테스트 질문: {test_input}\n")
    
    config = {"configurable": {"thread_id": "quality_test"}}
    
    inputs = {
        "user_input": test_input,
        "chat_history": [],
        "category": "",
        "context": [],
        "use_function": "",
        "province": "",
        "city": "",
        "region": "",
        "feature_keywords": [],
        "error": None,
        "chat_answer": "",
        "optimized_search_query": "",
        "documents": [],
        "retrieval_relevance": "",
        "hallucination_check": "",
        "answer_relevance": "",
        "retry_count": 0,
        "quality_score": 0.0,
        "evaluation_feedback": ""
    }
    
    retry_count = 0
    for output in app.stream(inputs, config):
        for node_name, node_output in output.items():
            if node_name == "generate_response_node":
                retry_count += 1
                print(f"🔄 답변 생성 시도 #{retry_count}")
            elif node_name == "evaluate_quality_node":
                score = node_output.get('quality_score', 0)
                feedback = node_output.get('evaluation_feedback', '')
                print(f"📊 품질 점수: {score:.2f}/10.0")
                if feedback:
                    print(f"💭 피드백: {feedback}")
    
    final_state = app.get_state(config).values
    print(f"\n✅ 최종 재시도 횟수: {final_state.get('retry_count', 0)}")
    print(f"✅ 최종 품질 점수: {final_state.get('quality_score', 0):.2f}/10.0")


def interactive_mode():
    """대화형 테스트 모드"""
    
    print("\n" + "=" * 60)
    print("Kumdori Chatbot - 대화형 모드")
    print("=" * 60)
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    app = build_graph()
    config = {"configurable": {"thread_id": "interactive"}}
    
    while True:
        user_input = input("질문: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("챗봇을 종료합니다.")
            break
        
        if not user_input:
            continue
        
        inputs = {
            "user_input": user_input,
            "chat_history": [],
            "category": "",
            "context": [],
            "use_function": "",
            "province": "",
            "city": "",
            "region": "",
            "feature_keywords": [],
            "error": None,
            "chat_answer": "",
            "optimized_search_query": "",
            "documents": [],
            "retrieval_relevance": "",
            "hallucination_check": "",
            "answer_relevance": "",
            "retry_count": 0,
            "quality_score": 0.0,
            "evaluation_feedback": ""
        }
        
        try:
            for output in app.stream(inputs, config):
                pass  # 내부 로그만 출력
            
            final_state = app.get_state(config).values
            print(f"\n💬 답변: {final_state.get('chat_answer', 'N/A')}")
            print(f"📊 품질: {final_state.get('quality_score', 0):.2f}/10.0\n")
            
        except Exception as e:
            print(f"❌ 에러: {e}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kumdori Chatbot 테스트")
    parser.add_argument(
        "--mode",
        choices=["basic", "quality", "interactive"],
        default="basic",
        help="테스트 모드 선택"
    )
    
    args = parser.parse_args()
    
    if args.mode == "basic":
        test_chatbot()
    elif args.mode == "quality":
        test_quality_retry()
    elif args.mode == "interactive":
        interactive_mode()
