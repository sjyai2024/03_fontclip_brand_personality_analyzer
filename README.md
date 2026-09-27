# 03B FontCLIP Brand Personality Analyzer v1.0

## 역할
RQ1 본분석용 Streamlit 앱입니다.

`공식 텍스트 5D ↔ FontCLIP 영문 로고타입 5D`

03A는 방법개발 파일럿으로 보존하고, 03B에서 실제 RQ1용 서체 기반 시각적 브랜드 개성 프로파일을 계산합니다.

## 03A에서 유지하는 것
- 동일 FontCLIP source commit
- 동일 checkpoint
- 동일 1024×1024 흑백 로고 표준화
- 동일 image encoder

## 03A에서 본분석용으로 사용하지 않는 것
- `not X font`
- positive-negative contrast
- Primary Dimension 중심 해석
- maximin shortlist를 최종 계산표본으로 사용하는 방식

## 03B 핵심방법
1. Aaker(1997) 42 traits를 사용
2. 각 trait의 primary prompt를 `{trait} font`로 고정
3. FontCLIP cosine similarity 계산
4. 42 traits → 15 facets → 5 dimensions
5. raw score 보존
6. 적격 로고 reference mean을 이용한 reference-centered profile 산출
7. Text–FontCLIP RQ1 비교에서는 **동일한 overlapping eligible brand set**을 두 모델의 공통 centering reference로 사용
8. 5D cosine similarity를 RQ1 주지표로 사용
9. DejaVu Sans 동일문자 렌더링은 reference-font sensitivity로만 사용

## 왜 `{trait} font`인가
FontCLIP 공식 구현의 attribute prompt는 `bold font`와 같은 형식을 사용합니다. 본 연구는 Aaker trait를 FontCLIP에 직접 연결할 때 연구자 자유도를 줄이기 위해 하나의 positive prompt family를 primary rule로 고정합니다.

복수 prompt ensemble과 부정문은 03B primary analysis에 사용하지 않습니다. 필요할 경우 별도 민감도 분석으로만 추가할 수 있습니다.

## 입력
### 필수
- 표준화 영문 로고타입 ZIP
  - 권장: 기존 `logo_dataset_68_fixed.zip`

### 앱에 포함된 기준파일
- `03A_eligibility_review_final.csv`
  - 현재 03A 최종 검토: A 52 / B 14 / C 2
- `Aaker1997_42traits_15facets_5dimensions_mapping.csv`
- `02_text_5D_raw_FOR_03B.csv`
  - 현재 02 raw 5D 39개 브랜드

### 선택
- 새로운 A/B/C 검토 CSV
- 새로운 02 text raw 5D CSV

## Reference pool
본분석 기본값은 **A only**입니다.

B는 부가문구, 심볼, 다단배열 등이 정리된 뒤 A로 재분류하는 것이 권장됩니다.
C는 본분석에서 제외합니다.

## RQ1 공통 reference
Text와 FontCLIP은 서로 다른 모델이므로 raw cosine 크기를 직접 비교하지 않습니다.

RQ1에서는 다음 절차를 사용합니다.

1. Text raw 5D와 FontCLIP raw 5D의 브랜드 overlap 확인
2. 그중 현재 FontCLIP reference eligibility에 포함되는 브랜드만 common reference로 설정
3. 각 모델에서 같은 브랜드 집합의 차원별 평균을 각각 계산
4. 각 모델에서 그 평균을 차감
5. 두 common-reference-centered 5D profile의 cosine similarity 계산

이렇게 하면 서로 다른 브랜드 집합을 기준으로 center한 프로파일을 직접 비교하는 문제를 줄일 수 있습니다.

## Reference-font sensitivity
실제 로고와 같은 문자열을 DejaVu Sans로 다시 렌더링하여 actual-minus-reference 차이를 계산합니다.

DejaVu Sans를 심리적으로 완전히 중립적인 서체라고 가정하지 않습니다. 이 분석은 브랜드명 문자열 자체가 FontCLIP 결과에 영향을 줄 가능성을 점검하는 **민감도 분석**입니다.

## 결과파일
- `03B_01_logo_review.csv`
- `03B_02_prompt_definition.csv`
- `03B_03_trait_scores_raw.csv`
- `03B_04_trait_scores_reference_centered.csv`
- `03B_05_facet_profiles_raw.csv`
- `03B_06_facet_profiles_reference_centered.csv`
- `03B_07_5D_profiles_raw.csv`
- `03B_08_5D_profiles_reference_centered.csv`
- `03B_09_reference_font_sensitivity.csv`
- `03B_10_text_fontclip_congruence.csv`
- `03B_11_reference_parameters.csv`
- `03B_12_RQ1_common_reference_parameters.csv`
- `03B_13_image_embeddings.csv`
- `03B_14_run_metadata.csv`

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Community Cloud에서는 Python 3.12를 권장합니다.

## 재현성 고정값
- FontCLIP commit: `3d4c6af01f668800d8e4f9f4f753d29c74dad252`
- checkpoint SHA256: `c441277fbed4366d32d8fb65725189b97d3fe88bae5fe0648b969feea01bbb00`
- model: ViT-B/32 + LoRA-text checkpoint

## 해석 주의
FontCLIP은 Aaker 브랜드 개성 척도를 위해 개발된 모델이 아닙니다. 본 연구의 5D 점수는 Aaker 구조를 FontCLIP semantic typography space에 조작적으로 적용한 계산 프로파일입니다. 최종 외적 타당성은 05 인간평가에서 별도로 검증합니다.

## v1.1 Streamlit 배포 수정

v1.0에서 `Aaker1997_42traits_15facets_5dimensions_mapping.csv`가
`app.py`와 같은 배포 폴더에 없을 경우 앱 시작 시 `FileNotFoundError`가 발생할 수 있었습니다.

v1.1은 다음을 적용했습니다.

- `app.py` 위치뿐 아니라 현재 작업폴더와 저장소 상위폴더에서 리소스 자동 탐색
- Aaker 42-trait mapping을 `app.py` 내부 fallback으로 포함
- 03A 검토 CSV가 없으면 앱은 계속 실행되고 수동 검토 가능
- 02 Text 5D CSV가 없으면 앱 전체가 종료되지 않고 RQ1 결합만 건너뛰거나 업로드 요청
- 화면의 `배포 리소스 상태`에서 실제 탐색 경로 확인 가능

GitHub에는 가능하면 이 폴더의 파일을 **모두 같은 디렉터리**에 올리는 것을 권장합니다.
