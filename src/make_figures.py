# -*- coding: utf-8 -*-
"""
make_figures.py — 논문 그림 7.1~7.5 를 n=1375 + 300dpi 로 재생성/신규작성

그림
----
7.1 ROC 곡선 (4모델)                         ← nb02 셀22 방식 (정수코딩 16변수 + 학습fold SMOTENC)
7.2 SHAP 평균절댓값 막대                       ← nb02 셀24 방식 (RF+TreeExplainer)
7.3 SHAP summary  (a)전체16 / (b)독립·조절만   ← 두 장 분리 (fig73a/fig73b)
7.4 정리정돈상태 dependence plot  ★신규        ← x축 정리정돈 1~5, 3점 이하 임계 가시화
7.5 조절효과 SHAP 상호작용 히트맵 ★신규        ← 8독립 × 3조절, shap_interaction_values

설계
----
- SHAP: RF + TreeExplainer, SMOTENC 학습셋에 fit된 RF 재사용 (nb02 셀24 방식, 정수코딩 16변수).
  정수코딩이라 dependence plot x축이 정리정돈 원값(1~5)으로 자연스럽게 나오고,
  히트맵의 독립·조절 변수명이 깔끔하다. (표 7.8은 별도로 더미코딩 SHAP 사용 — 논문명 일치용)
- random_state=42 고정으로 재현성 확보. 한글 폰트 Malgun Gothic. 전부 300dpi.
- 기존 노트북(01,02) 셀은 수정하지 않는다.

출력: outputs/figures/fig71.png, fig72.png, fig73a.png, fig73b.png, fig74.png, fig75.png
"""

import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import platform

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

# 한글 폰트
if platform.system() == "Windows":
    mpl.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":
    mpl.rc("font", family="AppleGothic")
mpl.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "전처리_최종.csv"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

GROUP_A = ["안전조직수준", "위원회수준", "인증보유"]
GROUP_B = ["위험성평가수준", "교육훈련도움", "정리정돈상태", "작업중지권", "작업반장기여"]
MODERATORS = ["전문지도", "고용노동부감독", "안전보건공단지원"]
CATCOLS = GROUP_A + ["위험성평가수준"] + MODERATORS + ["공사규모", "발주처", "공사종류"]
TARGET = "사고발생"
DPI = 300


def savefig(name):
    path = FIG / name
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  저장: outputs/figures/{name}  ({DPI}dpi)")


def main():
    df = pd.read_csv(DATA)
    y = df[TARGET]
    FEATURES = [c for c in df.columns if c != TARGET]
    X = df[FEATURES].copy()
    cat_idx = [X.columns.get_loc(c) for c in CATCOLS]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    Xtr_r, ytr_r = SMOTENC(categorical_features=cat_idx, random_state=42).fit_resample(Xtr, ytr)
    scl = StandardScaler().fit(Xtr_r)
    Xtr_sc, Xte_sc = scl.transform(Xtr_r), scl.transform(Xte)
    print(f"데이터 n={len(df)} | train(SMOTENC)={Xtr_r.shape} | test={Xte.shape}")

    # ── 4개 모델 학습 (테스트셋 확률) ────────────────────────────────
    models = {
        "로지스틱회귀": LogisticRegression(max_iter=1000, random_state=42),
        "랜덤포레스트": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0, n_jobs=-1),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
    }
    probs = {}
    rf_model = None
    for name, m in models.items():
        if name == "로지스틱회귀":
            m.fit(Xtr_sc, ytr_r); probs[name] = m.predict_proba(Xte_sc)[:, 1]
        else:
            m.fit(Xtr_r, ytr_r); probs[name] = m.predict_proba(Xte)[:, 1]
        if name == "랜덤포레스트":
            rf_model = m

    # ── 그림 7.1 ROC ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    for name in models:
        fpr, tpr, _ = roc_curve(yte, probs[name])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(yte, probs[name]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("거짓 양성률 (1 − 특이도)", fontsize=12)
    ax.set_ylabel("참 양성률 (민감도)", fontsize=12)
    ax.set_title("그림 7.1  ROC 곡선 — 4개 모델 비교 (n=1375, SMOTENC)", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig("fig71.png")

    # ── SHAP (RF, 정수코딩 16변수) ──────────────────────────────────
    sv = shap.TreeExplainer(rf_model).shap_values(Xte)
    arr = sv[1] if isinstance(sv, list) else sv
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    print(f"SHAP values shape: {arr.shape}")

    # 그림 7.2 SHAP 막대
    plt.figure(figsize=(10, 8))
    shap.summary_plot(arr, Xte, plot_type="bar", show=False, max_display=16)
    plt.title("그림 7.2  SHAP 변수 중요도 (평균 |SHAP|, RandomForest, n=1375)", fontsize=12)
    plt.tight_layout()
    savefig("fig72.png")

    # 그림 7.3 (a) 전체 16변수 summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(arr, Xte, show=False, max_display=16)
    plt.title("그림 7.3(a)  SHAP 요약 — 전체 16변수 (RandomForest, n=1375)", fontsize=12)
    plt.tight_layout()
    savefig("fig73a.png")

    # 그림 7.3 (b) 독립·조절만 (통제변수 마스킹)
    iv_mod = GROUP_A + GROUP_B + MODERATORS  # 11개
    idx = [Xte.columns.get_loc(c) for c in iv_mod]
    plt.figure(figsize=(10, 7))
    shap.summary_plot(arr[:, idx], Xte[iv_mod], show=False, max_display=11)
    plt.title("그림 7.3(b)  SHAP 요약 — 독립·조절 11변수만 (통제변수 제외, n=1375)", fontsize=12)
    plt.tight_layout()
    savefig("fig73b.png")

    # 그림 7.4 정리정돈상태 dependence plot (★신규)
    plt.figure(figsize=(8, 6))
    hk_idx = Xte.columns.get_loc("정리정돈상태")
    shap.dependence_plot(hk_idx, arr, Xte, interaction_index=None, show=False,
                         ax=plt.gca(), dot_size=16)
    plt.axvline(3.5, color="red", ls="--", lw=1.2, alpha=0.7, label="임계 경계(≤3)")
    plt.title("그림 7.4  정리정돈상태 SHAP Dependence (n=1375)", fontsize=12)
    plt.xlabel("정리정돈상태 (1~5)", fontsize=11)
    plt.ylabel("SHAP value (사고 위험 기여)", fontsize=11)
    plt.legend(fontsize=10)
    plt.tight_layout()
    savefig("fig74.png")

    # 그림 7.5 조절효과 SHAP 상호작용 히트맵 (★신규, 8독립 × 3조절)
    print("  SHAP interaction values 계산 중...")
    iv8 = GROUP_A + GROUP_B
    siv = shap.TreeExplainer(rf_model).shap_interaction_values(Xte)
    sarr = siv[1] if isinstance(siv, list) else siv
    if sarr.ndim == 4:
        sarr = sarr[:, :, :, 1]
    inter = np.abs(sarr).mean(axis=0)  # (16,16)
    colmap = {c: Xte.columns.get_loc(c) for c in Xte.columns}
    H = np.zeros((len(iv8), len(MODERATORS)))
    for i, iv in enumerate(iv8):
        for j, mo in enumerate(MODERATORS):
            # 비대각 상호작용은 대칭이며 절반씩 분배되므로 두 항 합산
            H[i, j] = inter[colmap[iv], colmap[mo]] + inter[colmap[mo], colmap[iv]]
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(H, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(MODERATORS))); ax.set_xticklabels(MODERATORS, rotation=20, ha="right")
    ax.set_yticks(range(len(iv8))); ax.set_yticklabels(iv8)
    for i in range(len(iv8)):
        for j in range(len(MODERATORS)):
            ax.text(j, i, f"{H[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("그림 7.5  SHAP 상호작용 강도 — 독립8 × 조절3 (mean|interaction|, n=1375)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean |SHAP interaction|")
    plt.tight_layout()
    savefig("fig75.png")

    print("\n완료: outputs/figures/fig71.png, fig72.png, fig73a.png, fig73b.png, fig74.png, fig75.png")


if __name__ == "__main__":
    main()
