# 🧬 펩톤 효소 추천 시스템 (Peptone Enzyme Selector)

원료의 성분 분석 데이터를 기반으로 펩톤 생산에 최적화된 효소를 자동 추천하는 Tool입니다.

## 📋 목차

1. [개요](#개요)
2. [설치 방법](#설치-방법)
3. [사용 방법](#사용-방법)
4. [알고리즘 설명](#알고리즘-설명)
5. [효소 데이터베이스](#효소-데이터베이스)
6. [파일 구조](#파일-구조)
7. [확장 및 커스터마이징](#확장-및-커스터마이징)

---

## 개요

### 목적
- 신규 원료(미세조류, 동물성 소재 등) 도입 시 효소 선정 시간 단축
- 아미노산 프로파일 기반 과학적 효소 매칭
- 최적 반응 조건(온도, pH, E/S ratio) 제공

### 핵심 기능
- ✅ Excel 파일 기반 성분 분석 데이터 자동 처리
- ✅ 원료 유형 자동 감지 (식물성/동물성/미세조류/콜라겐 등)
- ✅ Top 2 효소 추천 및 점수 산출
- ✅ 최적 반응 조건 제공
- ✅ Streamlit 웹 UI 지원

### 지원 원료
- 식물성: 대두(SOY), 밀(Wheat), 완두(Pea), 쌀(Rice)
- 동물성: 돼지(Pork), 어류(Fish), 젤라틴/콜라겐
- 유제품: 카제인(Casein)
- 미생물: 효모(Yeast), 미세조류(Microalgae)

---

## 설치 방법

### 1. 환경 요구사항
- Python 3.10 이상
- pip 또는 conda

### 2. 의존성 설치

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. VS Code 설정 (권장)

`.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true
}
```

---

## 사용 방법

### 방법 1: Streamlit 웹 앱 (권장)

```bash
streamlit run app/streamlit_app.py
```

브라우저에서 `http://localhost:8501` 접속 후:
1. Excel 파일 업로드 또는 직접 입력
2. 분석할 샘플 선택
3. "효소 추천 받기" 클릭
4. 결과 확인 및 적용

### 방법 2: Python 코드에서 직접 사용

```python
import pandas as pd
from src.recommender import EnzymeRecommender, print_recommendation_report

# 1. 추천 엔진 초기화
recommender = EnzymeRecommender('data/enzyme_database.json')

# 2. 데이터 로드
df = pd.read_excel('composition_template.xlsx', sheet_name='data')

# 3. 특정 샘플에 대해 추천
results = recommender.recommend(df, sample_id='Sample_01', top_n=2)

# 4. 결과 출력
for sample_id, result in results.items():
    print_recommendation_report(result['analysis'], result['recommendations'])
```

### 방법 3: 단일 샘플 간편 분석

```python
from src.recommender import EnzymeRecommender

recommender = EnzymeRecommender('data/enzyme_database.json')

# 아미노산 프로파일 정의 (g/100g)
amino_acid_profile = {
    'Asp': 7.2, 'Glu': 12.1, 'Ser': 3.3, 'Gly': 2.5,
    'Ala': 2.5, 'Val': 2.6, 'Leu': 4.2, 'Ile': 2.6,
    'Pro': 3.2, 'Phe': 2.8, 'Tyr': 1.9, 'Trp': 0.5,
    'Lys': 4.6, 'Arg': 5.7, 'His': 2.1, 'Met': 0.2
}

# 추천 실행
analysis, recommendations = recommender.recommend_single(
    amino_acid_profile,
    raw_material='soy',
    total_nitrogen=9.9,
    top_n=2
)

# 결과 확인
for rec in recommendations:
    print(f"#{rec.rank} {rec.enzyme_name}: {rec.score}점")
    print(f"  온도: {rec.optimal_temp}, pH: {rec.optimal_pH}")
```

---

## 알고리즘 설명

### 1. 입력 데이터 처리

```
Excel 파일 → 아미노산 프로파일 추출 → 그룹 비율 계산
```

### 2. 원료 유형 감지

| 특성 | 판단 기준 |
|------|----------|
| 콜라겐 계열 | Gly + Pro + Hyp > 25% |
| 효모 계열 | Glu 비율 > 12%, 산성 AA > 15% |
| 동물성 | 염기성 AA (Lys, Arg, His) > 15% |
| 기본값 | 식물성 |

### 3. 스코어링 공식

```
Score = Σ(AA그룹비율 × 효소친화도 × 가중치) - 프롤린페널티
      + 원료적합성보너스 + 특수원료보너스
```

**가중치:**
- 소수성 AA: 30%
- 방향족 AA: 25%
- 염기성 AA: 20%
- 산성 AA: 15%
- 프롤린 페널티: 10%

**보너스:**
- 원료 유형 매칭: ×1.20
- 콜라겐 특화 효소: ×1.25
- 세포벽 처리 효소: ×1.30

### 4. 출력

```
Top N 효소 → 최적 조건 → 추천 근거 → 주의사항
```

---

## 효소 데이터베이스

### 등록 효소 목록

| 효소명 | 유형 | 최적 온도 | 최적 pH | 주요 용도 |
|--------|------|----------|---------|----------|
| Alcalase 2.4L | Endoprotease | 55-60°C | 7.5-8.5 | 범용 (식물성) |
| Flavourzyme 1000L | Endo/Exo 복합 | 45-55°C | 5.5-7.0 | FAN 극대화, 쓴맛 감소 |
| Protamex | Endoprotease 복합 | 45-55°C | 6.0-8.0 | 식물성, 저쓴맛 |
| Neutrase 0.8L | Metalloprotease | 45-55°C | 6.0-7.5 | 동물성, 젤라틴 |
| Papain | Cysteine protease | 60-70°C | 5.5-7.5 | 범용, 콜라겐 |
| Bromelain | Cysteine protease | 50-60°C | 5.0-8.0 | 동물성, 콜라겐 |
| Trypsin | Serine protease | 35-45°C | 7.5-8.5 | Lys/Arg 특이적 |
| Pepsin | Aspartic protease | 35-45°C | 1.5-2.5 | 산성조건, 동물성 |
| Pronase E | Protease mixture | 35-50°C | 7.0-8.0 | 미세조류, 완전가수분해 |
| Celluclast+Protease | 복합 | 45-55°C | 4.5-6.0 | 세포벽 분해 + 단백질 |

### 데이터베이스 확장

`data/enzyme_database.json` 파일의 `enzymes` 배열에 새 효소 추가:

```json
{
  "id": "new_enzyme",
  "name": "New Enzyme 1.0",
  "manufacturer": "Company",
  "type": "endoprotease",
  "optimal_conditions": {
    "temperature": {"min": 50, "max": 55, "unit": "°C"},
    "pH": {"min": 6.0, "max": 7.0},
    "ES_ratio": {"min": 0.5, "max": 1.5, "unit": "% (w/w)"},
    "reaction_time": {"min": 2, "max": 4, "unit": "hours"}
  },
  "specificity": {
    "affinity_scores": {
      "hydrophobic": 0.85,
      "aromatic": 0.80,
      "basic": 0.70,
      "acidic": 0.60,
      "proline_penalty": 0.50
    }
  },
  "suitable_substrates": ["soy", "wheat"],
  "characteristics": {
    "DH_range": "15-25%",
    "FAN_yield": "높음",
    "bitterness": "중간",
    "specificity_type": "광범위"
  }
}
```

---

## 파일 구조

```
peptone_enzyme_selector/
│
├── data/
│   └── enzyme_database.json    # 효소 특성 DB
│
├── src/
│   ├── __init__.py
│   └── recommender.py          # 핵심 추천 엔진
│
├── app/
│   └── streamlit_app.py        # 웹 UI
│
├── tests/                      # 테스트 코드
├── docs/                       # 문서
├── requirements.txt            # 의존성
└── README.md                   # 이 파일
```

---

## 확장 및 커스터마이징

### 1. 가중치 조정

`data/enzyme_database.json`의 `scoring_weights` 섹션 수정:

```json
"scoring_weights": {
  "hydrophobic_weight": 35,  // 소수성 비중 증가
  "aromatic_weight": 20,
  "basic_weight": 25,
  "acidic_weight": 15,
  "proline_penalty_weight": 5
}
```

### 2. 새 원료 유형 추가

`substrate_type_rules`에 새 유형 추가:

```json
"new_material": {
  "preferred_enzymes": ["alcalase", "flavourzyme"],
  "characteristics": ["특성 설명"],
  "typical_TN": "10-12%"
}
```

### 3. 머신러닝 확장 (고급)

기존 실험 데이터가 축적되면 `scikit-learn`을 활용한 모델 학습 가능:

```python
from sklearn.ensemble import RandomForestRegressor

# 특성: 아미노산 조성 + 원료 유형 (원핫인코딩)
# 타겟: 실제 수율 (%)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 예측
predicted_yield = model.predict(X_new)
```

---

## 라이선스

내부 사용 전용 (Internal Use Only)

---

## 문의

R&D Team - [email@company.com]
