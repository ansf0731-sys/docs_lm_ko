# DocSLM 기반 한국어 문서 표/차트 추출 및 불확실성 시각화

본 프로젝트는 한국어 문서에서 표와 차트 정보를 추출하고, 추출 결과의 불확실성을 시각화하는 졸업 과제입니다.

DocSLM 공식 저장소(Tanveer81/DocSLM)는 현재 README만 공개되어 있고 모델 가중치가 공개되지 않았습니다. 따라서 본 프로젝트에서는 OCR-free 시각 언어 모델인 Qwen2-VL-2B-Instruct를 사용하여 문서 추론 파이프라인을 먼저 구축합니다.

## 현재 진행 상황

- PDF 페이지를 이미지로 변환하는 전처리 파이프라인 구현
- Qwen2-VL-2B-Instruct를 이용한 단일 페이지 문서 추론 완료
- 모델 출력 토큰의 엔트로피를 계산하여 불확실성을 측정하는 스크립트 구현

## 디렉토리 구조
## 실험 결과 (Baseline Evaluation)

파인튜닝 전 Qwen2-VL-2B-Instruct를 한국어 정부 보고서 이미지 기반으로 평가한 초기 베이스라인 결과입니다.

| 지표 | 점수 | 비고 |
| :--- | :--- | :--- |
| Exact Match (EM) | 0.00 | 정답 완전 일치율 |
| ANLS | 0.05 | 문자열 유사도 |
| Keyword Match (KM) | 0.70 | 핵심 키워드 포함율 |
| Avg Entropy | 0.00 | 엔트로피 기반 불확실성 (계산 로직 수정 예정) |

**분석**: EM/ANLS은 낮으나 KM이 70%로 높게 측정되어, 모델이 이미지 내 핵심 키워드를 시각적으로 인식·추출하는 능력은 갖추고 있음을 확인하였습니다. 다만 질문 의도에 맞는 '정답 형식' 생성 능력이 부족하므로, LoRA 파인튜닝을 통한 성능 개선이 다음 단계 목표입니다.

### 재현 방법 (Reproduction)
1. `pip install -r requirements.txt`
2. `python scriptspdf_to_img.py`  # PDF → 이미지 변환
3. `python scriptseval_metrics.py`  # 지표 스크립트 자기점검
4. `python scriptsrun_eval.py`  # 평가 실행