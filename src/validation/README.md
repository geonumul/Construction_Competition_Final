# src/validation — 표본 버그 진단·검증 근거

이 폴더의 스크립트들은 **수정 전 `src/03_logistic_regression.py`가 추론 회귀를
전체표본(n=1375)이 아니라 8:2 학습분할(n=1100)에 적합하던 버그**를 진단·증명하고,
전체표본 기준 결과를 독립적으로 교차검증한 근거다. (별도 분석 세션에서 작성)

> 데이터: 이 스크립트들은 `data/processed_2021.csv`(n=1375, TARGET=`accident`)를 사용한다.
> 모든 추론 회귀는 statsmodels Logit · SMOTENC 미적용 · Z표준화 · 더미 기준범주는
> 본 파이프라인과 동일(공사규모=2·발주처=2·기성공정률=3·공사종류=7).

## 스크립트

| 파일 | 역할 | 핵심 산출 |
|---|---|---|
| `diagnose_trainsplit_bug.py` | **결정적 증거.** 동일 M5를 전체표본 vs 학습분할에 적합해 비교 | 학습분할(n=1100)이 논문 원고를 **소수점까지 재현**: 정리정돈 β=−0.168/p=0.101, M5 AIC=**1192.57**. 전체표본(n=1375)은 정리정돈 p=**0.021**, AIC=1460.43 |
| `verify_paper_tables.py` | 논문 Table 7.1~7.8을 n=1375로 전수 재현·대조 | [논문값 / n1375 재현값 / 불일치 플래그] |
| `regenerate_tables_357.py` | 위계 M1~M5 (표 7.3/7.5/7.6) 전체표본 재계산 | 더미 기준범주 확정본 |
| `robustness_housekeeping_threshold.py` | 정리정돈 ≤3 임계더미 robustness | housekeeping_low(≤3): OR=1.535, p=0.025 |

## 결론 (이 근거가 뒷받침하는 것)

1. **버그 위치**: `src/03`의 `sm.Logit(y_train, …)` — 추론 회귀가 학습분할에 적합됨.
2. **증명**: 학습분할 적합값이 논문 원고 표(AIC 1192.57 등)와 정확히 일치.
3. **수정**: 추론 회귀를 전체표본(n=1375)에 적합하도록 변경(`src/03`). train/test 분할은
   ML 예측(04)·SHAP(05) 전용으로 유지(캐시 분할 해시 불변 확인).
4. **결과**: 표본이 커지며 경계선 변수가 경향→유의로 전환 —
   정리정돈상태 p=0.101→**0.021**, 인증보유×고용노동부감독 p=0.063→**0.007**(부호 + 유지).

## 실행

```bash
python src/validation/diagnose_trainsplit_bug.py     # 버그 증거 (가장 먼저 볼 것)
python src/validation/verify_paper_tables.py         # 논문 7.1~7.8 전수대조
python src/validation/regenerate_tables_357.py       # 위계 재계산
python src/validation/robustness_housekeeping_threshold.py
```
