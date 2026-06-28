# 건설업 산업재해 분석 — XAI 기반 위계적 LR + ML + SHAP

> 「건설안전 사고결과 예측을 위한 설명가능 인공지능(XAI) 기반 분석: 회귀모형과 머신러닝 모형의 비교를 중심으로」

## 연구 질문 (RQ)

**RQ1**: 내부 안전관리체계(독립 A 3변수) + 현장 안전행동(독립 B 5변수) → 산업재해 발생  
**RQ2**: 외부기관 개입(조절 3변수)의 조절효과

## 데이터

- **원자료**: 제10차 산업안전보건 실태조사 (2021, 건설업) — 1,502 사업장
- **최종 분석 표본**: 1,375 × 17 (Listwise Deletion 6단계)
- **종속변수 분포**: 발생 391 (28.4%) / 미발생 984 (71.6%)

### 변수 구성

| 구분 | 변수 (수) | 비고 |
|---|---|---|
| 종속 | 사고발생 `accident` (1) | 0/1 |
| 독립 A — 내부 안전관리체계 | 안전조직수준·위원회수준·인증보유 (3) | 0/1 |
| 독립 B — 현장 안전행동 | 위험성평가수준·교육훈련도움·정리정돈상태·작업중지권·작업반장기여 (5) | 0~2 또는 1~5 리커트 |
| 조절 — 외부기관 개입 | 전문지도·고용노동부감독·안전보건공단지원 (3) | 0/1 |
| 통제 — 현장 특성 | 공사규모·발주처·기성공정률·공사종류·외국인비율 (5) | 더미 15 + 연속 1 |

## 분석 단계

1. **§3.1~3.4 전처리**: 더미 15, Z-score, SMOTENC (학습셋 한정·CV fold 내부). 8:2 stratified split은 **ML 예측(§4.2)·SHAP(§5) 전용**.
2. **§4.1 위계적 LR**: M1(통제 16) → M2(+A) → M3(+B) → M4(+조절) → M5(+상호작용 24). **★추론 회귀(M1~M5, VIF)는 모집단 효과 추정이 목적이므로 전체표본 n=1,375에 적합한다** (train/test 분할을 쓰지 않음 — 분할은 예측 평가 전용).
3. **§4.2 ML**: LR/RF/XGB/LGB × 5-fold StratifiedKFold + GridSearchCV(scoring='f1') — 그리드는 논문 §7.8 (RF F1=0.517 최고) 재현을 위한 사양으로 고정 (LR `C∈[0.1,1,10]`, RF `n_est∈[200,400], depth∈[None,8,12]`, XGB·LGB 동급 sweep)
4. **§5 SHAP**: TreeExplainer (or LinearExplainer), 단일 모델 학습 → 시각화에서만 통제변수 마스킹

> **표본 이력 (중요).** 이전 버전은 추론 회귀(표 7.3~7.7)를 전체표본이 아니라 8:2 **학습분할(n=1,100)** 에 적합하는 버그가 있었다(`sm.Logit(y_train, …)`). 학습분할 적합값이 옛 결과표와 소수점까지 일치(정리정돈 β=−0.168/p=0.101, M5 AIC=1192.57)함이 `src/validation/diagnose_trainsplit_bug.py`로 증명된다. 현재는 **전체표본 n=1,375** 로 바로잡았으며, 이로써 경계선 변수가 경향→유의로 전환된다: **정리정돈상태 p=0.101→0.021**, **인증보유×고용노동부감독 p=0.063→0.007**(부호 + 유지). ML·SHAP 출력은 분할이 동일하여 불변(캐시 분할 해시로 확인). 진단·검증 근거는 [`src/validation/`](src/validation/) 참조.
>
> **VIF (표 7.3).** 전체표본 기준 최대 5.10(공사종류_1), 나머지 26개 변수 < 4.2. 5를 근소 초과한 공사종류_1은 동일 범주형에서 파생된 더미로 구조적으로 VIF가 다소 높으며(O'Brien, 2007), 관용 기준(10)에 크게 못 미쳐 실질적 다중공선성 문제로 보기 어렵다.

## 폴더 구조

```
construction_competition_final/
├── data/
│   ├── raw_2021.csv                  원자료 (cp949 원본, UTF로 rename)
│   ├── processed_2021.csv            전처리 산출 (1375×17, UTF-8 BOM)
│   └── 제10차...230824.CSV             원본 보존
├── src/
│   ├── utils_font.py                 한글 폰트 + 흑백 호환 스타일
│   ├── 01_preprocessing.py           표본 선정 + 변수 구성
│   ├── 02_descriptive.py             기술통계 (table71, 72, figure71)
│   ├── 03_logistic_regression.py     위계적 LR (table73~77)
│   ├── 04_ml_models.py               ML 4모형 + ROC (table78, figure72)
│   ├── 05_shap_analysis.py           SHAP (table79, figures 73~76)
│   ├── 06_research_model_figure.py   연구 모형 다이어그램 (figure21)
│   └── validation/                   표본 버그 진단·검증 근거 (train분할 n1100 → 전체표본 n1375)
├── outputs/
│   ├── tables/                       table71~79 (9 CSVs)
│   ├── figures/                      figure21 + figure71~76 (8 PNGs, 300dpi)
│   └── _intermediate/                스크립트 간 캐시 (joblib)
├── notebooks/full_pipeline.ipynb     6 스크립트 오케스트레이션
├── docs/                             분석·전처리·참고논문 (3 md)
├── legacy/                           이전 노트북·산출물 백업
├── RESULTS_SUMMARY.md                자동 생성 결과 요약
└── requirements.txt
```

## 실행

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) 순차 실행
python src/01_preprocessing.py        # 1,375 × 17 검증
python src/02_descriptive.py          # ~10초
python src/03_logistic_regression.py  # ~20초
python src/04_ml_models.py            # 15~30분 (193 grid × 5 fold)
python src/05_shap_analysis.py        # 1~2분
python src/06_research_model_figure.py # ~5초

# 또는 notebook으로 일괄
jupyter nbconvert --to notebook --execute notebooks/full_pipeline.ipynb
```

## 산출물 명세

### 표 (outputs/tables/, UTF-8 BOM CSV)

| 파일 | 내용 |
|---|---|
| `table71_descriptive_continuous.csv` | 연속·순서형 6변수 평균(SD) + Welch t-test |
| `table72_descriptive_categorical.csv` | 범주형 10변수 χ² |
| `table73_vif.csv` | 27변수 VIF (z-scored 학습셋, 내림차순) |
| `table74_hierarchical_lr.csv` | M1~M5 Pseudo R²·AIC·BIC |
| `table75_lr_test.csv` | 단계별 LR test (ΔR²·χ²·df·p) |
| `table76_m5_coefficients.csv` | M5 주효과 β/SE/z/p/OR/95%CI |
| `table77_interactions.csv` | 24개 상호작용항 (p 오름차순) |
| `table78_ml_performance.csv` | 4모형 CV F1·Test {Acc, P, R, F1, AUC} |
| `table79_shap_importance.csv` | mean\|SHAP\| Top 10 + 변수 유형 |

### 그림 (outputs/figures/, 300 dpi PNG, 흑백 호환)

| 파일 | 내용 |
|---|---|
| `figure21_research_model.png` | 연구 모형 다이어그램 (4 박스 + H1~H3b 화살표) |
| `figure71_y_distribution.png` | 종속변수 분포 (해치 패턴) |
| `figure72_roc_curves.png` | 4모형 ROC (선 스타일 + 마커 구분) |
| `figure73_shap_bar.png` | SHAP 평균 절댓값 Top 10 (회색 단색) |
| `figure74a_shap_summary_all.png` | Summary Plot 전체 27변수 (Greys cmap) |
| `figure74b_shap_summary_rqonly.png` | Summary Plot RQ 11변수 (통제 마스킹, 재학습 X) |
| `figure75_shap_dependence_정리정돈.png` | 정리정돈 Dependence (interaction_index=None) |
| `figure76_shap_interaction_heatmap.png` | 8×3 상호작용 |SHAP| 히트맵 |

## 흑백 인쇄 호환 디자인

- 컬러 의존 금지: 회색조 + 선 스타일 + 마커 + 해치 패턴
- 선 1.5pt+, 마커 충분히 크게, 축선 검정 1.0pt
- 격자선 회색 alpha=0.3, 배경 흰색, 스파인 top/right 제거

## 한글 폰트

`src/utils_font.py`가 OS별 자동 감지 → `fm.fontManager.addfont(path)`로 강제 등록:
- macOS: AppleGothic / Apple SD Gothic Neo / NanumGothic
- Windows: Malgun Gothic / NanumGothic / Gulim
- Linux: NanumGothic / Noto Sans CJK KR

SHAP 라이브러리 plot에는 `show=False` 후 `reapply_font_to_current_axes()`로 폰트 재적용.

## 재현성

- 전역 `random_state = 42`
- Train/Test 8:2 stratified split (random_state=42)
- SMOTENC: imblearn.Pipeline으로 CV fold의 train fold에만 작동 (val·test 원분포 유지)
- GridSearchCV(scoring='f1', cv=StratifiedKFold(5, shuffle=True, random_state=42))

## 참고

- 원자료: 한국산업안전보건공단 산업안전보건연구원 (2021). 제10차 산업안전보건 실태조사.
- 박천수 (2024, 2025). 동일 원자료 선행연구 (전처리 절차 준용).
- Reason (1990, 2000). Swiss Cheese Model.
- Levine, Toffel, Johnson (2012). *Science* — 외부감독 선택편향 보정.
- Lundberg & Lee (2017). SHAP.
- Chawla et al. (2002). SMOTE.
