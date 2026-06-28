# -*- coding: utf-8 -*-
"""
robustness_housekeeping_threshold.py

목적
----
SHAP dependence plot에서 관찰된 '정리정돈상태(housekeeping, 1-5 Likert)'의
비선형 임계(threshold) 패턴 — 3점 이하에서 사고 기여도가 급증 — 이
선형 로지스틱 회귀로도 재현되는지 검증하는 robustness check.

이 스크립트는 기존 노트북(02_데이터분석.ipynb)의 어떤 셀도 수정하지 않으며,
동일한 추론 LR 설계(statsmodels Logit · 전체표본 n=1375 · Z-score 표준화 ·
SMOTENC 미적용)를 그대로 재사용한다.

세 모델 비교 (동일 표본·동일 통제):
    M-A : 연속형 정리정돈상태만 포함            (baseline 재현)
    M-B : 연속형을 housekeeping_low 더미로 교체   (임계만)
    M-C : 연속형 + 더미 동시 포함                (임계가 선형을 넘는 추가 설명력?)

임계 기준 민감도: <=3 (주분석) 및 <=2 (보조분석) 두 가지로 반복.

주의(프롬프트 명세 준수):
- SMOTENC 미적용: 효과 추정용 모델이므로 원본 불균형 분포 그대로 fit.
- 표준화: 연속형 공변량은 Z-score, housekeeping_low 더미는 표준화하지 않음(0/1 유지).
- statsmodels Logit 사용: p-value·신뢰구간이 정확히 산출됨(sklearn은 p-value 미제공).
"""

import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Windows 콘솔에서 한글이 깨지지 않도록 UTF-8 강제
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

# ── 경로 (스크립트 위치 기준으로 안전하게 해석) ─────────────────────────
HERE = Path(__file__).resolve().parent           # .../src/validation
ROOT = HERE.parent.parent                          # 프로젝트 루트
DATA = ROOT / "data" / "processed_2021.csv"
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTCSV = OUTDIR / "results_housekeeping_threshold.csv"

# ── 변수 그룹 (노트북 셀[2]와 동일) ───────────────────────────────────
GROUP_A = ["안전조직수준", "위원회수준", "인증보유"]
GROUP_B = ["위험성평가수준", "교육훈련도움", "정리정돈상태", "작업중지권", "작업반장기여"]
MODERATORS = ["전문지도", "고용노동부감독", "안전보건공단지원"]
CONTROLS = ["공사규모", "발주처", "기성공정률", "공사종류", "외국인비율"]
TARGET = "accident"
HK = "정리정돈상태"  # housekeeping condition

# M-A/B/C의 '나머지' 공변량: 8개 독립(A+B) + 3개 조절 + 5개 통제.
# 기존 추론 LR(표 4-B)은 B그룹만 썼으나, 프롬프트의 '독립변수 8개 + 조절 + 통제,
# 통제변수 포함' 명세에 맞춰 전체 주효과 모델을 base로 삼는다.
# (housekeeping은 아래에서 사양별로 별도 처리하므로 여기서는 제외한 목록을 만든다)
BASE_CONTINUOUS = [v for v in (GROUP_A + GROUP_B + MODERATORS + CONTROLS)]  # 표준화 대상 전체
OTHERS = [v for v in BASE_CONTINUOUS if v != HK]  # housekeeping 제외한 나머지 공변량


def fit_logit(X: pd.DataFrame, y: pd.Series):
    """상수항 추가 후 statsmodels Logit 적합."""
    return sm.Logit(y, sm.add_constant(X)).fit(disp=0)


def coef_row(model, term, label):
    """한 항(term)의 β, SE, z, p, OR, 95%CI를 dict로."""
    beta = model.params[term]
    se = model.bse[term]
    z = model.tvalues[term]
    p = model.pvalues[term]
    ci_lo, ci_hi = model.conf_int().loc[term]
    return {
        "항": label,
        "β": round(beta, 4),
        "SE": round(se, 4),
        "z": round(z, 4),
        "p값": round(p, 4),
        "OR": round(np.exp(beta), 4),
        "OR_95CI_하한": round(np.exp(ci_lo), 4),
        "OR_95CI_상한": round(np.exp(ci_hi), 4),
        "유의도": "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "",
    }


def mcfadden(model, y):
    """McFadden pseudo-R² (statsmodels .prsquared와 동일, 명시 계산)."""
    return model.prsquared


def lr_test(full, reduced):
    """우도비검정: full이 reduced를 nest할 때. (chi2, df, p)."""
    chi2 = 2 * (full.llf - reduced.llf)
    df_diff = int(full.df_model - reduced.df_model)
    p = stats.chi2.sf(chi2, df_diff) if df_diff > 0 else np.nan
    return chi2, df_diff, p


def run_threshold(df_s, y, raw_hk, thr, all_rows, summary_rows):
    """주어진 임계기준 thr에 대해 M-A/B/C 적합·비교."""
    label_thr = f"<={thr}"
    hk_low_name = f"housekeeping_low({label_thr})"

    # 더미: 원본(비표준화) housekeeping <= thr → 1, 그 외 → 0
    hk_low = (raw_hk <= thr).astype(int)
    hk_low.name = hk_low_name

    # 공통 공변량(표준화 완료) — housekeeping 제외
    X_others = df_s[OTHERS].copy()

    # ── M-A: 연속형 housekeeping만 (표준화된 값) ──────────────────
    X_A = X_others.copy()
    X_A[HK] = df_s[HK]
    m_A = fit_logit(X_A, y)

    # ── M-B: 연속형 대신 더미 (더미는 표준화 안 함) ───────────────
    X_B = X_others.copy()
    X_B[hk_low_name] = hk_low.values
    m_B = fit_logit(X_B, y)

    # ── M-C: 연속형 + 더미 동시 ──────────────────────────────────
    X_C = X_others.copy()
    X_C[HK] = df_s[HK]
    X_C[hk_low_name] = hk_low.values
    m_C = fit_logit(X_C, y)

    # ── housekeeping 관련 계수표 ─────────────────────────────────
    print(f"\n{'='*78}")
    print(f" 임계기준: housekeeping {label_thr}  →  housekeeping_low=1  "
          f"(해당 현장 {int(hk_low.sum())}개 / {len(hk_low)}, "
          f"{hk_low.mean()*100:.1f}%)")
    print(f"{'='*78}")

    coef_tbl = []
    rA = coef_row(m_A, HK, f"M-A: {HK}(연속·표준화)")
    coef_tbl.append(rA)
    rB = coef_row(m_B, hk_low_name, f"M-B: {hk_low_name}(더미)")
    coef_tbl.append(rB)
    rC1 = coef_row(m_C, HK, f"M-C: {HK}(연속·표준화)")
    rC2 = coef_row(m_C, hk_low_name, f"M-C: {hk_low_name}(더미)")
    coef_tbl.append(rC1)
    coef_tbl.append(rC2)

    coef_df = pd.DataFrame(coef_tbl)
    print("\n[housekeeping 관련 계수]")
    print(coef_df.to_string(index=False))

    # CSV 누적 (계수)
    for r in coef_tbl:
        all_rows.append({"임계기준": label_thr, "구분": "계수", **r})

    # ── 모델 적합도 비교 ─────────────────────────────────────────
    fit_tbl = []
    for nm, m in [("M-A", m_A), ("M-B", m_B), ("M-C", m_C)]:
        fit_tbl.append({
            "모델": nm,
            "df_model": int(m.df_model),
            "LogLik": round(m.llf, 3),
            "AIC": round(m.aic, 3),
            "BIC": round(m.bic, 3),
            "McFadden_R2": round(mcfadden(m, y), 5),
        })
    fit_df = pd.DataFrame(fit_tbl)
    print("\n[모델 적합도]")
    print(fit_df.to_string(index=False))
    for r in fit_tbl:
        all_rows.append({"임계기준": label_thr, "구분": "적합도", **r})

    # ── 우도비검정 ───────────────────────────────────────────────
    # M-C는 M-A와 M-B를 각각 nest (각 1 df 추가).
    # M-A vs M-B는 비-내포(non-nested, 동일 df) → LRT 불가, AIC/BIC로 비교.
    chi_CA, df_CA, p_CA = lr_test(m_C, m_A)  # 더미가 연속형 위에 추가 설명력?
    chi_CB, df_CB, p_CB = lr_test(m_C, m_B)  # 연속형이 더미 위에 추가 설명력?

    print("\n[우도비검정 (LRT)]")
    print(f"  M-A ⊂ M-C  (더미 추가 효과): chi2={chi_CA:.3f}, df={df_CA}, "
          f"p={p_CA:.4f}{'  *유의' if p_CA < .05 else ''}")
    print(f"  M-B ⊂ M-C  (연속형 추가 효과): chi2={chi_CB:.3f}, df={df_CB}, "
          f"p={p_CB:.4f}{'  *유의' if p_CB < .05 else ''}")
    aic_better = min([("M-A", m_A.aic), ("M-B", m_B.aic), ("M-C", m_C.aic)],
                     key=lambda t: t[1])
    print(f"  M-A vs M-B (비-내포): AIC {m_A.aic:.2f} vs {m_B.aic:.2f} "
          f"→ AIC 최소 모델 = {aic_better[0]}")

    for nm, chi2, dfd, p in [("M-A⊂M-C(더미추가)", chi_CA, df_CA, p_CA),
                             ("M-B⊂M-C(연속추가)", chi_CB, df_CB, p_CB)]:
        all_rows.append({"임계기준": label_thr, "구분": "LRT", "비교": nm,
                         "chi2": round(chi2, 3), "df": dfd, "p값": round(p, 4),
                         "유의도": "*" if p < .05 else ""})

    # ── 한 줄 요약 (M-B의 더미) ─────────────────────────────────
    or_b = np.exp(m_B.params[hk_low_name])
    p_b = m_B.pvalues[hk_low_name]
    sig = "유의(p<0.05)" if p_b < .05 else f"비유의(p={p_b:.3f})"
    direction = "위험↑(OR>1)" if or_b > 1 else "위험↓(OR<1)"
    one_liner = (f"[요약·{label_thr}] M-B housekeeping_low: {sig}, "
                 f"OR={or_b:.3f} → {direction}")
    print(f"\n{one_liner}")
    summary_rows.append(one_liner)


def main():
    df = pd.read_csv(DATA)
    n = len(df)
    print(f"데이터: {DATA.name}  | n = {n}  | 사고발생 양성비 = {df[TARGET].mean():.4f}")
    print(f"정리정돈상태 분포: "
          f"{df[HK].value_counts().sort_index().to_dict()}")

    # 노트북 셀[2]와 동일한 Z-score 표준화 (연속형 공변량 전체)
    scaler = StandardScaler()
    df_s = df.copy()
    df_s[BASE_CONTINUOUS] = scaler.fit_transform(df[BASE_CONTINUOUS])
    y = df_s[TARGET]
    raw_hk = df[HK]  # 더미 생성은 원본값 기준

    all_rows, summary_rows = [], []

    # 주분석(<=3) + 민감도(<=2)
    for thr in (3, 2):
        run_threshold(df_s, y, raw_hk, thr, all_rows, summary_rows)

    # ── CSV 저장 ─────────────────────────────────────────────────
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUTCSV, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {OUTCSV}")

    print("\n" + "=" * 78)
    print(" 최종 한 줄 요약")
    print("=" * 78)
    for s in summary_rows:
        print(" " + s)


if __name__ == "__main__":
    main()
