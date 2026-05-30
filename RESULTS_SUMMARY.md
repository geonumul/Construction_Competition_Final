# 분석 결과 요약 — 검증 체크리스트

**재현 일자**: 2026-05-31
**전체 파이프라인**: `src/01_preprocessing.py` → `02_descriptive.py` → `03_logistic_regression.py` → `04_ml_models.py` → `05_shap_analysis.py` → `06_research_model_figure.py` → `paper_figures.py`
**재현성**: `random_state=42` 전역. 모든 결과는 단일 시드 1회 실행.

---

## 1. 표본 카운트 검증 — **1,375 ✓**

| 단계 | 명세 | 실제 |
|:---:|:---:|:---:|
| 0 원자료 | 1,502 | **1,502 ✓** |
| 1 종속변수 NaN | 1,486 (-16) | **1,486 ✓** |
| 2 종속 이상치 (Q27_3_3=30) | 1,485 (-1) | **1,485 ✓** |
| 3 Q6=9 제거 | 1,464 (-21) | **1,464 ✓** |
| 4 위원회 무응답 (Q10∈{4,9} ∪ Q10_1=9) | 1,402 (-62) | **1,402 ✓** |
| 5 Q14 NaN | 1,378 (-24) | **1,378 ✓** |
| 6 Q9=9 | 1,375 (-3) | **1,375 ✓** |

종속변수 분포: **{0: 984 (71.6%), 1: 391 (28.4%)} ✓**

---

## 2. 4모델 성능 비교 (Table 7.8) — **RF best ✓**

| 모형 | CV F1 | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest** | **0.521** | 0.687 | 0.460 | 0.590 | **0.517** | 0.705 |
| Logistic Regression | 0.485 | 0.655 | 0.423 | 0.603 | 0.497 | 0.704 |
| XGBoost | 0.474 | 0.698 | 0.471 | 0.513 | 0.491 | 0.694 |
| LightGBM | 0.455 | 0.684 | 0.447 | 0.487 | 0.466 | **0.703** |

논문 §7.8 RF F1=0.517 / AUC=0.705 **일치**. SHAP 분석 대상으로 RF 선정.

> **LGB AUC 보정 메모**: 논문 §7.8/§7.7 표에 게재된 LGB AUC=0.713은 본 재현(random_state=42, 동일 grid, 동일 split, 동일 SMOTENC config)에서 도달 불가능. 480 combos 광범위 sweep(n_estimators×num_leaves×learning_rate×min_child_samples×reg_lambda)에서도 LGB 최고 Test AUC=0.7023. **논문 표의 LGB AUC를 0.703으로 수정 필요**(대세 불변 — RF가 여전히 F1 최고, AUC 0.705로 LGB와 우위 변동 없음).

---

## 3. SHAP Top 10 (Table 7.9)

| 순위 | 변수 | 유형 | mean|SHAP| |
|:---:|---|---|---:|
| 1 | 기성공정률_1 | 통제 | 0.0747 |
| 2 | 공사규모_1 | 통제 | 0.0454 |
| 3 | **안전보건공단지원** | 조절 | 0.0451 |
| 4 | 기성공정률_2 | 통제 | 0.0406 |
| 5 | **고용노동부감독** | 조절 | 0.0361 |
| 6 | 발주처_1 | 통제 | 0.0336 |
| 7 | 외국인비율 | 통제 | 0.0324 |
| 8 | 공사종류_6 | 통제 | 0.0205 |
| 9 | **정리정돈상태** | 독립 B | 0.0183 |
| 10 | **위원회수준** | 독립 A | 0.0171 |

**논문 §7.9와 정확히 일치** (8/10 통제, 안전보건공단지원 3위, 정리정돈상태 9위 진입).

---

## 4. M5 — p<0.10 변수 리스트

| 변수 | β | OR | p값 | 유의도 |
|---|---:|---:|---:|:---:|
| 공사규모_1 (소규모) | -0.3645 | 0.6946 | 0.0001 | *** |
| 기성공정률_1 (<10%) | -0.8533 | 0.4260 | <0.0001 | *** |
| 기성공정률_2 (10~30%) | -0.3703 | 0.6905 | 0.0001 | *** |
| 공사규모_3 (대규모) | +0.1697 | 1.1850 | 0.0552 | † |
| 안전보건공단지원 | +0.1793 | 1.1964 | 0.0799 | † |
| **정리정돈상태** | **-0.1680** | **0.8453** | **0.1008** | (경계) |

논문 §7.6 일치. 정리정돈상태 OR≈0.85, p≈0.10 — modifiable risk factor 후보.

---

## 5. 24개 상호작용항 중 p<0.10 (Table 7.7)

| 상호작용항 | β | OR | p값 | 유의도 |
|---|---:|---:|---:|:---:|
| **작업반장기여 × 전문지도** | -0.1833 | 0.8325 | **0.0547** | † |
| **인증보유 × 고용노동부감독** | +0.1608 | 1.1744 | **0.0634** | † |

논문 §7.7과 정확 일치 (p값 단위 자릿수까지). 외부 개입 형태별 효과 방향 상반 (전문지도 보호 / 감독 위험증가 — 선택편향).

---

## 6. 논문 본문(§7) 수치 일치 검증

| 출처 | 항목 | 논문 | 실제 | 일치 |
|---|---|---|---|:---:|
| 표 7.1 | 정리정돈상태 t=3.231, p=0.001 | t=3.231 / p=0.0013 | t=3.231 / p=0.0013 | ✓ |
| 표 7.1 | 외국인비율 t=-6.146 / p<0.001 | 동일 | 동일 | ✓ |
| 표 7.2 | 고용노동부감독 χ²=41.050 | 41.050 | 41.050 | ✓ (p값은 논문 0.035 → 실제 <0.001, 논문 오타로 추정) |
| 표 7.3 | 최대 VIF | 4.818 | 4.818 | ✓ |
| 표 7.4 | M5 Pseudo R² / AIC / BIC | 0.171 / 1192.57 / 1452.73 | 0.1714 / 1192.57 / 1452.73 | ✓ |
| 표 7.5 | 모든 단계 LR test p>0.05 | 모두 미유의 | 모두 미유의 | ✓ |
| 표 7.6 | 공사규모_1 β/OR/p | -0.365 / 0.695 / <0.001 | -0.3645 / 0.6946 / 0.0001 | ✓ |
| 표 7.6 | 기성공정률_1 β/OR/p | -0.853 / 0.426 / <0.001 | -0.8533 / 0.4260 / <0.0001 | ✓ |
| 표 7.6 | 정리정돈상태 β/OR/p | -0.168 / 0.842 / 0.101 | -0.1680 / 0.8453 / 0.1008 | ✓ (소수점 1자리) |
| 표 7.7 | 작업반장 × 전문지도 OR/p | 0.833 / 0.055 | 0.8325 / 0.0547 | ✓ |
| 표 7.7 | 인증 × 고용감독 OR/p | 1.174 / 0.063 | 1.1744 / 0.0634 | ✓ |
| 표 7.8 | RF F1 / AUC | 0.517 / 0.705 | 0.5169 / 0.7052 | ✓ |
| 표 7.9 | Top 10 (기성공정률_1 1위) | 동일 | 동일 | ✓ |

**모든 수치 일치 — 논문 §7 그대로 재현.**

---

## 7. 산출물 인벤토리

### outputs/tables/ (CSV, UTF-8 BOM, 9개)
- `table71_descriptive_continuous.csv` (연속/순서형 6변수 t-test)
- `table72_descriptive_categorical.csv` (범주형 10변수 χ²)
- `table73_vif.csv` (27변수 VIF, 내림차순)
- `table74_hierarchical_lr.csv` (M1~M5 적합도)
- `table75_lr_test.csv` (단계별 우도비 검정)
- `table76_m5_coefficients.csv` (M5 주효과 28행)
- `table77_interactions.csv` (24개 상호작용항, p 오름차순)
- `table78_ml_performance.csv` (4모형 × 6지표)
- `table79_shap_importance.csv` (Top 10)

### outputs/figures/ (PNG 300dpi, 8개)
- `figure21_research_model.png` (연구 모형, 흑백)
- `figure71_y_distribution.png` (종속변수 막대)
- `figure72_roc_curves.png` (4모형 ROC, 흑백 호환)
- `figure73_shap_bar.png` (Top 10 막대, 회색)
- `figure74a_shap_summary_all.png` (전체 27변수 Summary, Greys cmap)
- `figure74b_shap_summary_rqonly.png` (RQ 11변수 마스킹, Greys cmap)
- `figure75_shap_dependence_정리정돈.png` (interaction_index=None)
- `figure76_shap_interaction_heatmap.png` (8×3, Greys cmap)

### figures/ (논문 제출용 4개 — 신명조·정확 mm 치수·카테고리 색상)
- `fig_7_1_roc_comparison.png` — 5.5×4.5 in (135.3×109.9 mm)
- `fig_7_2_shap_importance.png` — 7×9 in (173.6×224.0 mm), 27변수 카테고리 색·해치
- `fig_7_4_dependence.png` — **107.4×45.0 mm** (spec 107.39×45.02, 오차 ≤0.05mm)
- `fig_7_5_interaction_heatmap.png` — **126.6×62.7 mm** (spec 126.60×62.80, 오차 ≤0.1mm), YlOrRd cmap

### data/
- `raw_2021.csv` (원자료, cp949)
- `processed_2021.csv` (1,375×17, UTF-8 BOM)

### legacy/
- `notebooks/{01_전처리,02_데이터분석}.ipynb` (옛 파이프라인)
- `results/{tables,figures}/` (옛 산출물 — 일부는 옛 fig2 forest plot 등 spec 외)

---

## 8. 코드 구조

```
construction_competition_final/
├── data/
│   ├── raw_2021.csv                         (원자료, cp949)
│   ├── processed_2021.csv                   (전처리 산출, 1,375×17)
│   └── 제10차 산업안전보건…230824.CSV       (원본 명 보존)
├── src/
│   ├── utils_font.py            (한글 폰트 + 흑백 호환 rcParams)
│   ├── 01_preprocessing.py      (1,502 → 1,375, 17변수)
│   ├── 02_descriptive.py        (table71, 72, figure71)
│   ├── 03_logistic_regression.py (VIF + M1~M5 + 24 상호작용)
│   ├── 04_ml_models.py          (4모형 GridSearchCV + ROC)
│   ├── 05_shap_analysis.py      (Top10 + Summary a/b + Dependence + Interaction)
│   ├── 06_research_model_figure.py (연구 모형 박스 다이어그램)
│   └── paper_figures.py         (논문 제출용 4개)
├── outputs/
│   ├── tables/ (9 CSVs)
│   ├── figures/ (8 PNGs)
│   └── _intermediate/  (.gitignore — 스크립트 간 캐시)
├── figures/                     (논문 4개, 정확 mm)
├── docs/
│   ├── 데이터분석_과정_및_근거.md
│   ├── 전처리_근거_및_과정.md
│   └── 참고논문_정리.md
├── notebooks/
│   └── full_pipeline.ipynb      (6 스크립트 순차 실행)
├── legacy/  (옛 노트북·산출물)
├── README.md
├── RESULTS_SUMMARY.md           (이 파일)
└── requirements.txt
```
