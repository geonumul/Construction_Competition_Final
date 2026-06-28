# 건설업 산업재해 사고결과 예측 — 설명가능 머신러닝(XAI) 분석

> 제10차 산업안전보건 실태조사(2021, 한국산업안전보건공단) 건설업 자료를 이용해,
> **전통적 추론 회귀(로지스틱)** 와 **머신러닝 예측(RF·XGBoost·LightGBM)** 을 결합하고
> **SHAP** 으로 해석한 사고예측 분석 코드 저장소입니다.
> 확정 분석표본은 **n = 1,375** 사업장입니다.

---

## 1. 개요

- **연구 질문**: 건설 사업장의 내부 안전관리·현장 안전행동·외부기관 개입이 사고발생과
  어떻게 연관되며, 머신러닝/SHAP이 선형 회귀가 놓치는 패턴을 드러내는가?
- **종속변수**: `사고발생` (0/1, 양성 391건 = 28.4%, 불균형 구조)
- **독립변수 8개**: 내부관리 3(안전조직수준·위원회수준·인증보유) + 현장행동 5(위험성평가수준·교육훈련도움·정리정돈상태·작업중지권·작업반장기여)
- **조절변수 3개**: 전문지도·고용노동부감독·안전보건공단지원
- **통제변수 5개**: 공사규모·발주처·기성공정률·공사종류·외국인비율
- **방법**: 추론 LR(statsmodels, 전체표본, SMOTENC 미적용) + 예측 ML 4종(SMOTENC 학습셋 한정) + SHAP(RandomForest 기준)

---

## 2. 핵심 결과 (n = 1,375)

| 발견 | 수치 | 근거 |
|---|---|---|
| **정리정돈상태**: 유의한 보호효과 | β = −0.208, p = 0.021, OR = 0.812 [0.681, 0.969] | 표 7.5 (M5) |
| └ ≤3점 임계 패턴 | 3점 이하에서 SHAP 기여 급증(위험↑), 4~5점에서 음(보호) | 그림 7.4 dependence |
| └ 임계더미도 유의 | housekeeping_low(≤3): OR = 1.535, p = 0.025 | `src/validation/` robustness |
| **인증보유 × 고용노동부감독**: 위험방향 상호작용 | β = +0.207, p = 0.007 (24개 중 유일 유의) | 표 7.6 / 그림 7.5 |
| RandomForest: 4모델 중 F1 최고 | F1 = 0.524, AUC = 0.702, CV-F1(5fold) = 0.512 | 표 7.7 |
| 통제변수(공사규모·기성공정률)의 강한 영향 | 공사규모_1·기성공정률_1/_2 모두 p < 0.001 (보호), SHAP 상위 지배 | 표 7.5 / 7.8 |

> **해석 주의(본문 작성 시):**
> - 인증보유×고용노동부감독은 **회귀에서 유의**하며 SHAP 상호작용 히트맵에서도 **상위권**에 위치한다.
>   다만 SHAP interaction value는 *방향 없는 강도(절댓값)* 라 회귀의 유의 상호작용과 1:1로 대응하지 않는다
>   ("SHAP 1위"가 아니라 "상위권"이 정확한 표현). 또한 24개 상호작용 중 단일 유의이므로 **다중비교**를 함께 명시해야 한다.
> - 정리정돈상태의 임계 효과는 선형 항과 통계적으로 **중복(redundant)** 이다(둘 다 유의·동일 방향).
>   "선형이 놓친 비선형"이 아니라 "선형·임계·SHAP·단변량이 모두 일관"이라는 **수렴 증거(convergent evidence)** 로 기술하는 것이 정확하다.

---

## 3. 폴더 구조

```
construction_competition/
├── data/                  원자료(KOSHA raw CSV) + 전처리_최종.csv (n=1375, 17변수)
├── notebooks/             01_전처리.ipynb (제외로직·종속변수 구성) / 02_데이터분석.ipynb (원본 분석, 수정 안 함)
├── src/                   최종 산출 코드
│   ├── make_tables.py       표 7.1~7.8 생성 (n=1375)
│   ├── make_figures.py      그림 7.1~7.5 생성 (300dpi)
│   └── validation/          검증·재계산 스크립트 + 그 산출물(outputs/)
├── outputs/
│   ├── tables/            table71~78.csv  ← 논문용 최종 표 (n=1375)
│   └── figures/           fig71~75.png    ← 논문용 최종 그림 (300dpi)
├── results/
│   └── _archive_old/      옛 산출물 보존 (아래 §5 참조 — 최종본 아님)
└── docs/                  방법론 해설·매칭 근거 문서
```

각 폴더 한 줄 요약:
- **data/** — 분석 입력. `전처리_최종.csv`가 확정 표본(n=1375).
- **notebooks/** — 전처리(01)와 원본 탐색 분석(02). **원본 보존, 셀 미수정.**
- **src/** — 논문 표·그림을 재현하는 독립 스크립트.
- **src/validation/** — 논문값 대 n=1375 전수 대조, 임계더미 robustness, 위계표 재계산.
- **outputs/** — 논문에 들어가는 최종 표/그림.
- **results/_archive_old/** — 구버전 산출물(보존용).
- **docs/** — 방법·근거 서술.

---

## 4. 재현 방법

### 환경
```bash
pip install pandas numpy scipy statsmodels scikit-learn imbalanced-learn xgboost lightgbm shap matplotlib seaborn
```
한글 폰트는 Windows `Malgun Gothic` 기준(코드에 설정 포함).

### (A) 최종 표·그림 생성 — 권장
```bash
python src/make_tables.py     # → outputs/tables/table71.csv ~ table78.csv (콘솔에 논문값 대조 출력)
python src/make_figures.py    # → outputs/figures/fig71.png ~ fig75.png (300dpi)
```

### (B) 전처리·원본 분석 재현
```bash
jupyter notebook notebooks/01_전처리.ipynb   # 원자료 → 전처리_최종.csv 검증(제외로직·종속변수 일치)
jupyter notebook notebooks/02_데이터분석.ipynb # 원본 탐색 분석(A/B 분리 구조)
```

### (C) 논문값 검증 재현
```bash
python src/validation/verify_paper_tables.py          # 논문 7.1~7.8 전수 대조 → outputs/verify_paper_vs_repo.csv
python src/validation/regenerate_tables_357.py        # 위계 M1~M5 재계산 → outputs/regenerated_357.csv
python src/validation/robustness_housekeeping_threshold.py  # 정리정돈 임계더미 robustness
```

### 표/그림 ↔ 코드 매핑
| 산출물 | 생성 코드 | 비고 |
|---|---|---|
| 표 7.1 t검정 / 7.2 카이제곱 | `make_tables.py` | Welch t / chi-square |
| 표 7.3 위계 M1~M5 / 7.4 우도비 | `make_tables.py` | 더미코딩, R²/AIC/BIC |
| 표 7.5 통합 M5 주효과 / 7.6 상호작용 24 | `make_tables.py` | statsmodels Logit |
| 표 7.7 모델성능 | `make_tables.py` | 테스트셋 + **CV 5-fold f1** |
| 표 7.8 SHAP top10 | `make_tables.py` | **더미코딩 RF**(논문 변수명 일치) |
| 그림 7.1 ROC / 7.2 SHAP막대 | `make_figures.py` | |
| 그림 7.3(a)전체/(b)독립·조절 | `make_figures.py` | summary plot 2장 분리 |
| 그림 7.4 정리정돈 dependence | `make_figures.py` | ≤3 임계 가시화 |
| 그림 7.5 조절효과 SHAP 히트맵 | `make_figures.py` | 8독립 × 3조절 interaction |

---

## 5. ★ 표본 이력 (중요 — 반드시 읽을 것)

**논문 원고(0531본)의 결과표 일부 수치는 전체표본이 아니라 8:2 학습분할(n=1,100)에서 산출된 값이다.**

- **확정 표본은 n = 1,375** (`data/전처리_최종.csv`). 이는 원자료 1,502건에 6단계 제외로직
  (종속변수 결측·사망 이상치·무응답 5종)을 적용한 결과이며, `notebooks/01_전처리.ipynb`가
  실제 제외를 실행하고 `assert len == 1375` 및 종속변수 100% 일치로 검증한다.
- **원인 확정**: 추론 회귀(표 7.3~7.6)는 전체표본에 적합해야 하나, 원고 수치는
  **머신러닝 8:2 stratified 학습분할(n = 1,375 × 0.8 = 1,100, `random_state=42`)** 에서
  돌린 것으로 확인됐다. 검증 결과 학습분할에서:
  - 정리정돈상태 β = −0.1680, p = 0.1008 → 원고 −0.168 / 0.101 **정확 일치**
  - M5 McFadden R² = 0.1714, **AIC = 1192.57** → 원고 0.171 / **1192.57 소수점까지 일치**
  (재현: `src/validation/`)
- **결과적 차이(전체표본 n=1,375로 바로잡으면):** 표본이 커지며 검정력이 올라가
  경계선 변수가 **'경향 → 유의'** 로 전환된다 — 서사를 **강화**하는 방향이다.
  - 정리정돈상태: p = 0.101(비유의) → **p = 0.021(유의)**, OR = 0.812
  - 인증보유×고용노동부감독: p = 0.063(경향) → **p = 0.007(유의)**, 부호(+) 유지
  - (네 핵심 항목 모두 **부호는 그대로**, 유의성만 표본크기에 따라 이동)
- **옛 산출물 보존**: `results/_archive_old/`. 이들은 **(i) A/B 분리 구조**(논문 통합 M5와 다름),
  **(ii) 그림 dpi 120**(최종 300dpi 아님), **(iii) 일부 옛 서사 기준**이라 최종본이 아니다.
  데이터 자체는 n=1,375이므로 폐기하지 않고 보존한다.

---

## 6. 미해결 사항

- **발주처·공사종류 더미 기준범주(reference) 레이블 미확정.** 논문 표 7.5 주석은 발주처='사기업/개인',
  공사종류='전기/통신/기타'를 기준으로 명시하나, KOSHA 코드집을 확보하지 못해 해당 정수코드를 특정할 수 없다.
  현재 **임시값(발주처=2, 공사종류=7)** 으로 두고 표/콘솔에 ★플래그로 표시한다.
  → 코드집 확보 시 `src/make_tables.py`의 `REF` 딕셔너리 두 줄만 교체하면 표 7.5의 해당 행이 갱신된다.
  - **영향 범위**: 이 선택은 발주처/공사종류 **더미 계수 레이블에만** 영향을 준다.
    McFadden R²·AIC·BIC·우도비검정(7.3/7.4), 상호작용(7.6), 그리고 정리정돈상태·안전보건공단지원 등
    **모든 비-통제 계수와 p값은 기준범주 선택과 무관하게 불변**이다.

---

## 7. 데이터 출처

- **제10차 산업안전보건 실태조사 (2021)** — 한국산업안전보건공단(KOSHA)
- 원자료: `data/제10차 산업안전보건 실태조사_raw data_건설업_230824.CSV` (건설업, 1,502건)
- 제외 기준 및 변수 구성: `notebooks/01_전처리.ipynb`, `docs/전처리_근거_및_과정.md`

---

## 8. 사용 라이브러리

| 용도 | 라이브러리 |
|---|---|
| 추론 통계(회귀·p값·CI) | `statsmodels`, `scipy` |
| 머신러닝 | `scikit-learn`, `xgboost`, `lightgbm` |
| 불균형 처리 | `imbalanced-learn` (SMOTENC, 학습셋 한정) |
| 해석(XAI) | `shap` (TreeExplainer) |
| 데이터·시각화 | `pandas`, `numpy`, `matplotlib`, `seaborn` |

---

*분석 설계 원칙: 추론 LR은 전체표본·SMOTENC 미적용·Z표준화(statsmodels Logit). 예측 ML은 8:2 분할 후 학습 fold에만 SMOTENC 적용(데이터 누수 방지). 두 목적을 분리한다.*
