# Kumdori Chatbot 개선 사항

## 적용된 Adaptive RAG 테크닉

### 1. 검색 문서 관련성 평가 (Retrieval Grader)
- **목적**: 검색된 문서가 사용자 질문과 실제로 관련이 있는지 평가
- **구현**: `grade_documents_node()`
- **효과**: 
  - 무관한 문서 필터링으로 답변 품질 향상
  - 관련성 있는 문서만 사용하여 더 정확한 답변 생성
  - Binary score ('yes'/'no')로 명확한 판단

```python
class GradeDocuments(BaseModel):
    """검색된 문서의 관련성 평가 모델"""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
```

### 2. 환각(Hallucination) 체크
- **목적**: 생성된 답변이 제공된 문서에 근거하는지 검증
- **구현**: `check_hallucination_node()`
- **효과**:
  - 잘못된 정보 생성 방지
  - 문서 기반 답변 보장
  - 신뢰성 있는 정보 제공

```python
class GradeHallucinations(BaseModel):
    """답변의 환각(Hallucination) 체크 모델"""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
```

### 3. 답변 관련성 평가 (Answer Grader)
- **목적**: 생성된 답변이 실제로 사용자 질문을 해결하는지 평가
- **구현**: `grade_answer_node()`
- **효과**:
  - 질문과 무관한 답변 방지
  - 사용자 의도에 부합하는 답변 보장
  - 답변 품질 향상

```python
class GradeAnswer(BaseModel):
    """답변의 질문 해결 여부 평가 모델"""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
```

### 4. 쿼리 재작성 (Query Rewriter) 개선
- **목적**: 검색 효율을 높이기 위해 사용자 질문을 최적화
- **구현**: `websearch_optimizer_node()` 개선
- **효과**:
  - 더 나은 검색 결과
  - 의미적 의도를 파악한 쿼리 변환
  - 웹 검색 정확도 향상

### 5. 종합 품질 평가 시스템
- **목적**: 다차원적 품질 평가로 최종 답변 검증
- **구현**: `evaluate_quality_node()`
- **평가 항목**:
  - 검색 문서 관련성 (3.33점)
  - 환각 체크 (3.33점)
  - 답변 관련성 (3.34점)
  - **총점: 10점 만점**

## 워크플로우 개선

### Before (기존)
```
User Query → Categorize → Tool Selection → Generate Response → END
```

### After (개선)
```
User Query 
  → Categorize 
  → Tool Selection 
  → Generate Response 
  → Hallucination Check 
  → Answer Grading 
  → Quality Evaluation 
  → (품질 기준 미달 시) Retry with Feedback
  → END
```

## 재시도 메커니즘

### 품질 기반 재시도 로직
- **임계값**: 7.0/10.0
- **최대 재시도**: 2회
- **피드백 통합**: 이전 평가 결과를 다음 생성에 반영

```python
def route_quality_check(state: GraphState) -> str:
    quality_score = state.get("quality_score", 0.0)
    retry_count = state.get("retry_count", 0)
    
    if quality_score >= QUALITY_THRESHOLD:
        return "end"  # 품질 통과
    elif retry_count >= MAX_RETRY_COUNT:
        return "end"  # 최대 재시도 도달
    else:
        return "retry"  # 재생성
```

## GraphState 확장

### 새로 추가된 필드
```python
class GraphState(TypedDict):
    # 기존 필드들...
    documents: List[Any]              # 검색된 원본 문서
    retrieval_relevance: str          # 검색 문서 관련성 (yes/no)
    hallucination_check: str          # 환각 체크 결과 (yes/no)
    answer_relevance: str             # 답변 관련성 (yes/no)
    retry_count: int                  # 재시도 횟수
    quality_score: float              # 품질 점수
    evaluation_feedback: str          # 평가 피드백
```

## 주요 이점

### 1. 답변 품질 향상
- 다단계 검증으로 정확도 증가
- 환각 방지로 신뢰성 향상
- 관련성 평가로 사용자 만족도 증가

### 2. 자동 품질 관리
- 품질 기준 미달 시 자동 재생성
- 피드백 기반 개선
- 일관된 품질 유지

### 3. 투명한 의사결정
- 각 단계별 평가 결과 출력
- 디버깅 및 모니터링 용이
- 성능 분석 가능

### 4. Adaptive RAG 구현
- 상황에 맞는 동적 처리
- 쿼리 복잡도에 따른 적응
- 효율적인 리소스 활용

## 실행 예시

```python
# 그래프 빌드
app = build_graph()

# 실행
config = {"configurable": {"thread_id": "test_001"}}
inputs = {
    "user_input": "대전에서 맛있는 떡볶이 집 추천해줘",
    "chat_history": [],
    "retry_count": 0
}

# 스트리밍 실행
for output in app.stream(inputs, config):
    for key, value in output.items():
        print(f"Node: {key}")
        print(f"Output: {value}")
```

## 성능 모니터링

### LangSmith 추적
- 모든 체인 실행 추적
- 평가 메트릭 시각화
- 병목 지점 분석

### 콘솔 출력
- 각 노드 실행 로그
- 평가 점수 실시간 출력
- 라우팅 결정 표시

## 향후 개선 방향

1. **검색 전략 다양화**
   - 하이브리드 검색 (키워드 + 의미 검색)
   - 멀티 소스 통합

2. **평가 기준 세분화**
   - 도메인별 특화 평가 기준
   - 사용자 피드백 반영

3. **동적 임계값 조정**
   - 쿼리 복잡도에 따른 임계값 변경
   - 학습 기반 최적화

4. **캐싱 전략**
   - 자주 묻는 질문 캐싱
   - 응답 시간 단축
