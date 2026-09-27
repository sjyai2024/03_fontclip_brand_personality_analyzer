# 03B FontCLIP 본분석 방법 v1.0

## 연구목적
영문 로고타입에서 나타나는 서체 기반 시각적 브랜드 개성을 Aaker의 5차원 공통 좌표로 산출하고, 02 공식 커뮤니케이션 텍스트에서 계산한 표명된 지향 브랜드 개성 프로파일과 비교한다.

## 1. 분석대상
03A에서 구축한 표준화 영문 로고타입 후보군을 사용한다. 본분석 reference pool은 적격성 A를 기본으로 한다. B는 시각적 간섭요소를 제거한 뒤 재검토하며 C는 제외한다.

## 2. FontCLIP 고정환경
- commit: `3d4c6af01f668800d8e4f9f4f753d29c74dad252`
- model: ViT-B/32 / LoRA-text
- checkpoint SHA256: `c441277fbed4366d32d8fb65725189b97d3fe88bae5fe0648b969feea01bbb00`

## 3. Semantic prompt
Aaker(1997)의 42 traits를 사용한다.

Primary prompt:

`{trait} font`

예: `honest font`, `imaginative font`, `glamorous font`

03A에서 사용했던 `not X font`는 본분석에서 제외한다. Primary analysis에서는 prompt ensemble도 사용하지 않는다.

## 4. Trait score
각 로고 이미지 임베딩과 각 trait prompt 임베딩의 cosine similarity를 계산한다.

`S(i,t) = cosine(I_i, T_t)`

raw similarity는 모두 보존한다.

## 5. Aaker 계층집계
- 동일 facet의 traits를 동일가중 평균 → 15 facets
- 동일 dimension의 facets를 동일가중 평균 → 5 dimensions

trait 수가 많은 facet이 dimension에 더 큰 가중치를 갖지 않도록 facet을 동일가중한다.

## 6. FontCLIP 내부 reference centering
적격 reference logo set에서 trait별 평균을 구한다.

`S_centered(i,t) = S(i,t) - mean_reference[S(t)]`

이후 동일한 Aaker 계층으로 15 facets와 5 dimensions를 집계한다.

이 값은 절대적인 성격 강도가 아니라 현재 K-코스메틱 적격 로고 집합에서의 상대적 프로파일이다.

## 7. RQ1 Text–FontCLIP 비교의 공통 reference
텍스트와 FontCLIP의 embedding space가 다르므로 raw cosine을 직접 비교하지 않는다.

또한 두 모델이 서로 다른 브랜드 집합을 기준으로 center되는 문제를 피하기 위해 RQ1에서는 별도의 common-reference centering을 적용한다.

1. Text raw 5D와 FontCLIP raw 5D의 overlap을 확인한다.
2. FontCLIP 본분석 적격성에 포함되는 overlap 브랜드를 common reference로 고정한다.
3. 각 모델에서 같은 common reference 브랜드들의 차원별 평균을 계산한다.
4. 각 브랜드의 5D raw profile에서 해당 모델의 common reference mean을 차감한다.
5. 두 centered 5D profile의 cosine similarity를 계산한다.

### 주지표
5D cosine similarity

### 보조지표
- profile Pearson correlation
- L2-normalized Euclidean distance
- reference 대비 방향 일치 차원 수

Pearson correlation은 5개 차원만으로 계산되므로 보조적으로만 해석한다.

## 8. Reference-font sensitivity
동일 브랜드 문자열을 DejaVu Sans로 렌더링하고 실제 로고타입과 비교한다.

`Style sensitivity = Actual logo score - Reference-font score`

DejaVu Sans를 심리적으로 중립적인 서체라고 가정하지 않는다. 이 값은 브랜드명 문자열 및 표준 기준서체와 비교했을 때 실제 로고의 서체표현이 결과에 미치는 영향을 탐색하는 민감도 지표다.

## 9. 03A와의 관계
03A의 FontCLIP image embedding, 반복실행 안정성, Sophistication 편중, positive/negative 비대칭, maximin 이상치 민감성은 본분석 설계 근거로 보존한다.

03A의 `not X` contrast, Primary Dimension, maximin shortlist는 본분석 최종 측정값으로 사용하지 않는다.

## 10. 인간검증
03B 계산 후 대표 로고타입 24~30개를 선정해 인간평가를 수행하고, stimulus-level 5D profile의 FontCLIP–human correspondence를 검증한다.
