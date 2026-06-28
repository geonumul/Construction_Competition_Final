# -*- coding: utf-8 -*-
"""
diagnose_trainsplit_bug.py — train분할(n=1100) 버그의 '결정적 증거'

배경
----
수정 전 src/03_logistic_regression.py 는 추론 회귀(M1~M5)를 전체표본이 아니라
8:2 stratified 학습분할(n=1375×0.8=1100, random_state=42)에 적합하고 있었다.
이 스크립트는 그 사실을 재현 가능하게 증명한다:

  · 전체표본(n=1375) 적합  → 정리정돈 p≈0.021, M5 AIC≈1460.43  (올바른 추론)
  · 학습분할(n=1100) 적합  → 정리정돈 p≈0.101, M5 AIC=1192.57   (= 논문 원고 표값)

AIC가 소수점(1192.57)까지 논문값과 일치하면, 원고 결과표가 학습분할에 적합됐다는
직접 증거가 된다. (추론 회귀는 모집단 효과 추정이 목적이므로 전체표본이 정본.)

데이터: data/processed_2021.csv (n=1375), TARGET='accident'. git 상태 변경 없음.
"""
import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data" / "processed_2021.csv"

VARS_A = ["안전조직수준", "위원회수준", "인증보유"]
VARS_B = ["위험성평가수준", "교육훈련도움", "정리정돈상태", "작업중지권", "작업반장기여"]
VARS_MOD = ["전문지도", "고용노동부감독", "안전보건공단지원"]
CAT_CTRL = ["공사규모", "발주처", "기성공정률", "공사종류"]
CONT_CTRL = ["외국인비율"]
TARGET = "accident"
HK = "정리정돈상태"
STD_VARS = VARS_A + VARS_B + VARS_MOD + CONT_CTRL
# src/03 과 동일 기준범주
REF = {"공사규모": 2, "발주처": 2, "기성공정률": 3, "공사종류": 7}
RANDOM_STATE = 42


def make_dummies(df):
    parts = []
    for c in CAT_CTRL:
        d = pd.get_dummies(df[c].astype(int), prefix=c).astype(float)
        d = d[[col for col in d.columns if col != f"{c}_{REF[c]}"]]
        parts.append(d)
    return pd.concat(parts, axis=1)


def fit_M5(sub):
    """주어진 부분표본(sub)에 M5(주효과+상호작용24)를 적합."""
    Z = pd.DataFrame(StandardScaler().fit_transform(sub[STD_VARS]),
                     columns=STD_VARS, index=sub.index)
    D = make_dummies(sub)
    X = pd.concat([D, Z[CONT_CTRL], Z[VARS_A], Z[VARS_B], Z[VARS_MOD]], axis=1).copy()
    for iv in VARS_A + VARS_B:
        for mo in VARS_MOD:
            X[f"{iv}×{mo}"] = Z[iv] * Z[mo]
    return sm.Logit(sub[TARGET], sm.add_constant(X.astype(float))).fit(disp=0, maxiter=200)


def main():
    df = pd.read_csv(DATA)
    y = df[TARGET]
    # src/03 과 동일한 분할 (random_state=42)
    idx_tr, _ = train_test_split(df.index, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    print(f"데이터: {DATA.name} | 전체 n={len(df)} | 학습분할 n={len(idx_tr)}\n")
    print(f"{'적합 표본':14s} {'정리정돈 β':>10s} {'p':>8s} {'M5 R²':>8s} {'M5 AIC':>10s}")
    print("-" * 56)
    for label, sub in [("전체 n=%d" % len(df), df),
                       ("학습분할 n=%d" % len(idx_tr), df.loc[idx_tr])]:
        m = fit_M5(sub)
        print(f"{label:14s} {m.params[HK]:>10.4f} {m.pvalues[HK]:>8.4f} "
              f"{m.prsquared:>8.4f} {m.aic:>10.2f}")
    print("-" * 56)
    print("논문 원고(M5):     -0.1680   0.1008   0.1714    1192.57")
    print("\n→ 학습분할 적합값이 논문 원고와 소수점까지 일치 = 원고 표가 train분할(n=1100)에"
          " 적합됐다는 직접 증거. 전체표본(n=1375)이 추론 정본.")


if __name__ == "__main__":
    main()
