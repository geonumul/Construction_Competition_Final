# -*- coding: utf-8 -*-
"""
verify_paper_tables.py

목적
----
현재 data/전처리_최종.csv 가 '최종 데이터'라는 전제로, 논문 원고에 보고된
Table 7.1 ~ 7.8 의 통계 수치가 이 데이터로 재현되는지 전수 대조한다.

원칙
----
- 기존 노트북 셀은 일절 수정하지 않는다(이 스크립트는 독립 실행).
- 모든 추론 회귀는 statsmodels Logit (정확한 p값·CI). sklearn 회귀는 추론에 쓰지 않는다.
- 추론 LR: 전체표본 n=1375, SMOTENC 미적용, 연속/서열은 Z표준화(노트북 셀2 방식).
- 성능표(7.7)만 셀21 방식(train/test split + 학습 fold SMOTENC).
- "맞게 보이도록" 표본을 자르거나 코딩을 바꿔 끼워맞추지 않는다. 안 맞으면 안 맞는 대로 보고한다.

코딩 차이에 대한 핵심 주의
--------------------------
레포 문서(docs/KEY_PAPER_매칭_해설.md L71, 전처리_근거_및_과정.md L62)는 범주형 통제변수
(공사규모/발주처/공사종류 등)를 '정수 코드 그대로 입력'한다고 명시한다.
반면 논문 3.2.4 및 Table 7.5(공사규모_1, 기성공정률_1/_2)는 '더미변수'를 쓴다.
=> 이 둘은 데이터 불일치가 아니라 '코딩 방식 차이'이다. M1~M5(7.3~7.6)는 논문 묘사대로
   더미코딩으로 새로 구성하며, 그 사실을 명시 보고한다.

더미 기준범주(reference) 가정
-----------------------------
논문이 공사규모_1, 기성공정률_1/_2, 발주처_1, 공사종류_6 을 보고하므로 '범주 1'이 살아있다.
따라서 pandas 기본 drop_first(범주1 제거)와 반대로 '마지막 범주'를 기준으로 제거한다.
주의: 기준범주 선택은 개별 더미 계수(7.5/7.6 일부)에만 영향을 주며,
      McFadden R²/AIC/BIC/LR검정(7.3/7.4) 및 표준화 연속변수(정리정돈상태 등) 계수·p값에는
      전혀 영향을 주지 않는다(동일 모델의 재매개변수화).
"""

import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "processed_2021.csv"
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTCSV = OUTDIR / "verify_paper_vs_repo.csv"

# ── 변수 그룹 ────────────────────────────────────────────────────────
GROUP_A = ["안전조직수준", "위원회수준", "인증보유"]
GROUP_B = ["위험성평가수준", "교육훈련도움", "정리정돈상태", "작업중지권", "작업반장기여"]
MODERATORS = ["전문지도", "고용노동부감독", "안전보건공단지원"]
CAT_CTRL = ["공사규모", "발주처", "기성공정률", "공사종류"]   # 더미코딩 대상(논문 3.2.4)
CONT_CTRL = ["외국인비율"]                                    # 연속 통제(표준화)
CONTROLS = CAT_CTRL + CONT_CTRL
TARGET = "accident"
HK = "정리정돈상태"

# 표준화 대상: 8개 독립 + 3개 조절 + 연속통제(외국인비율)
STD_VARS = GROUP_A + GROUP_B + MODERATORS + CONT_CTRL

ROWS = []  # CSV 누적


def flag_num(paper, repro, tol):
    if paper is None or (isinstance(paper, float) and np.isnan(paper)):
        return ""
    return "OK" if abs(paper - repro) <= tol else "DIFF"


def flag_p(paper_p, repro_p, thr=0.05):
    """유의성 부호(유의↔비유의)가 바뀌면 SIGN-FLIP."""
    if paper_p is None:
        return ""
    a = paper_p < thr
    b = repro_p < thr
    return "OK" if a == b else "SIGN-FLIP"


def rec(table, item, metric, paper, repro, flag):
    ROWS.append({"표": table, "항목": item, "지표": metric,
                 "논문값": paper, "재현값": repro, "플래그": flag})


def hdr(t):
    print("\n" + "=" * 80)
    print(" " + t)
    print("=" * 80)


# ──────────────────────────────────────────────────────────────────────
def make_design(df):
    """추론 LR 설계행렬 구성 요소 반환."""
    sc = StandardScaler()
    Z = pd.DataFrame(sc.fit_transform(df[STD_VARS]), columns=STD_VARS, index=df.index)
    # 더미: 마지막 범주를 기준으로 제거(범주1 보존 → 논문 명명과 일치)
    dums = {}
    for c in CAT_CTRL:
        d = pd.get_dummies(df[c].astype(int).astype("category"), prefix=c).astype(float)
        last = sorted(df[c].unique())[-1]
        ref_col = f"{c}_{last}"
        if ref_col in d.columns:
            d = d.drop(columns=[ref_col])
        dums[c] = d
    D = pd.concat(dums.values(), axis=1)
    return Z, D


def fit_logit(X, y):
    return sm.Logit(y, sm.add_constant(X.astype(float))).fit(disp=0, maxiter=200)


def lr_test(full, reduced):
    chi2 = 2 * (full.llf - reduced.llf)
    df_diff = int(full.df_model - reduced.df_model)
    p = stats.chi2.sf(chi2, df_diff) if df_diff > 0 else np.nan
    return chi2, df_diff, p


# ── Table 7.1: 연속/서열 t검정 ────────────────────────────────────────
def table_71(df):
    hdr("Table 7.1  연속·서열 변수 집단비교 (t검정)  [사고발생군 n=391 vs 비발생군 n=984]")
    vars6 = ["위험성평가수준", "교육훈련도움", "정리정돈상태", "작업중지권", "작업반장기여", "외국인비율"]
    # 논문 보고값(부호 포함). 관례: t = (비발생군 평균) - (발생군 평균)
    paper = {
        "정리정돈상태": (3.231, 0.001), "외국인비율": (-6.146, 0.0001),
        "위험성평가수준": (-2.560, 0.011), "작업반장기여": (2.113, 0.035),
        "교육훈련도움": (None, None), "작업중지권": (None, None),
    }
    a = df[df[TARGET] == 0]   # 비발생군
    b = df[df[TARGET] == 1]   # 발생군
    print(f"{'변수':12s} {'논문t':>8s} {'재현t(Student)':>14s} {'재현t(Welch)':>13s} "
          f"{'논문p':>8s} {'재현p':>9s}  플래그")
    for v in vars6:
        t_s, p_s = stats.ttest_ind(a[v], b[v], equal_var=True)
        t_w, p_w = stats.ttest_ind(a[v], b[v], equal_var=False)
        pt, pp = paper[v]
        # 어느 쪽이 논문과 맞는지: Student 우선 비교
        f_t = flag_num(pt, t_s, 0.05) if pt is not None else ""
        f_p = flag_p(pp, p_s) if pp is not None else ("OK" if p_s >= .05 else "DIFF(논문:비유의)")
        ptxt = f"{pt:8.3f}" if pt is not None else "   (n.s.)"
        pptxt = f"{pp:8.3f}" if pp is not None else "       -"
        print(f"{v:12s} {ptxt} {t_s:14.3f} {t_w:13.3f} {pptxt} {p_s:9.4f}  {f_t}/{f_p}")
        rec("7.1", v, "t(Student)", pt, round(t_s, 3), f_t)
        rec("7.1", v, "p", pp, round(p_s, 4), f_p)
    print("주: t = (비발생군 평균 − 발생군 평균). Student(등분산) 기준 비교, Welch 병기.")


# ── Table 7.2: 범주형 카이제곱 ────────────────────────────────────────
def table_72(df):
    hdr("Table 7.2  범주형 변수 카이제곱 검정 (vs 사고발생)")
    cats10 = ["안전조직수준", "위원회수준", "인증보유", "전문지도", "고용노동부감독",
              "안전보건공단지원", "공사규모", "발주처", "기성공정률", "공사종류"]
    # 논문: 6개 유의(p<0.001), 4개 비유의
    paper_sig = {"고용노동부감독": True, "안전보건공단지원": True, "공사규모": True,
                 "발주처": True, "기성공정률": True, "공사종류": True,
                 "안전조직수준": False, "위원회수준": False, "인증보유": False, "전문지도": False}
    print(f"{'변수':12s} {'chi2':>10s} {'df':>4s} {'p값':>12s}  {'재현유의':>7s} {'논문유의':>7s}  플래그")
    for v in cats10:
        ct = pd.crosstab(df[v], df[TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        repro_sig = p < 0.001
        psig = paper_sig.get(v)
        # 플래그: 논문 유의성(p<0.001 기준)과 재현 유의성 부호 비교
        flag = "OK" if repro_sig == psig else "SIGN-FLIP"
        print(f"{v:12s} {chi2:10.3f} {dof:4d} {p:12.3e}  {str(repro_sig):>7s} "
              f"{str(psig):>7s}  {flag}")
        rec("7.2", v, "chi2", None, round(chi2, 3), "")
        rec("7.2", v, "p(<0.001 유의여부)", psig, repro_sig, flag)


# ── Table 7.3 / 7.4: 위계 M1~M5 + 우도비검정 ──────────────────────────
def table_73_74(df, Z, D, y):
    hdr("Table 7.3  위계적 로지스틱 M1~M5  (더미코딩, McFadden R² / AIC / BIC)")

    Xc = D                                  # 통제(더미)
    Xa = Z[GROUP_A]
    Xb = Z[GROUP_B]
    Xm = Z[MODERATORS]
    Xfn = Z[CONT_CTRL]                       # 외국인비율(연속통제)

    M1 = fit_logit(pd.concat([Xc, Xfn], axis=1), y)
    M2 = fit_logit(pd.concat([Xc, Xfn, Xa], axis=1), y)
    M3 = fit_logit(pd.concat([Xc, Xfn, Xa, Xb], axis=1), y)
    M4 = fit_logit(pd.concat([Xc, Xfn, Xa, Xb, Xm], axis=1), y)

    # M5: + 상호작용 24개 (독립8 × 조절3)
    X5 = pd.concat([Xc, Xfn, Xa, Xb, Xm], axis=1).copy()
    for iv in GROUP_A + GROUP_B:
        for mo in MODERATORS:
            X5[f"{iv}×{mo}"] = Z[iv] * Z[mo]
    M5 = fit_logit(X5, y)

    models = {"M1": M1, "M2": M2, "M3": M3, "M4": M4, "M5": M5}
    paper_r2 = {"M1": 0.150, "M5": 0.171}
    paper_aic = {"M1": 1151.41, "M5": 1192.57}

    print(f"{'모델':4s} {'k':>4s} {'McFadden_R2':>12s} {'(논문R2)':>9s} "
          f"{'AIC':>10s} {'(논문AIC)':>10s} {'BIC':>10s}  플래그")
    for nm, m in models.items():
        r2 = m.prsquared
        pr2 = paper_r2.get(nm)
        pa = paper_aic.get(nm)
        f_r2 = flag_num(pr2, r2, 0.01) if pr2 is not None else ""
        f_a = flag_num(pa, m.aic, 5.0) if pa is not None else ""
        pr2t = f"{pr2:9.3f}" if pr2 is not None else "        -"
        pat = f"{pa:10.2f}" if pa is not None else "         -"
        print(f"{nm:4s} {int(m.df_model):4d} {r2:12.5f} {pr2t} "
              f"{m.aic:10.2f} {pat} {m.bic:10.2f}  {f_r2}{f_a}")
        rec("7.3", nm, "McFadden_R2", pr2, round(r2, 5), f_r2)
        rec("7.3", nm, "AIC", pa, round(m.aic, 2), f_a)

    hdr("Table 7.4  위계 우도비검정 (M1→M2→M3→M4→M5)")
    print("논문: 전 단계 비유의(p>0.05).")
    steps = [("M1→M2", M2, M1), ("M2→M3", M3, M2),
             ("M3→M4", M4, M3), ("M4→M5", M5, M4)]
    print(f"{'단계':8s} {'ΔR2':>9s} {'LR_chi2':>10s} {'df':>4s} {'p값':>9s}  재현유의  플래그")
    for nm, full, red in steps:
        chi2, dfd, p = lr_test(full, red)
        dr2 = full.prsquared - red.prsquared
        repro_sig = p < 0.05
        flag = "OK" if not repro_sig else "SIGN-FLIP(논문:비유의)"
        print(f"{nm:8s} {dr2:9.4f} {chi2:10.3f} {dfd:4d} {p:9.4f}  "
              f"{str(repro_sig):>6s}   {flag}")
        rec("7.4", nm, "LR_p(비유의기대)", "p>0.05", round(p, 4), flag)
    return models, M5, X5


# ── Table 7.5: M5 주효과 ──────────────────────────────────────────────
def table_75(M5):
    hdr("Table 7.5  M5 주효과 (β, SE, z, p, OR, 95%CI)  — 논문 핵심 대조")
    paper = {
        "정리정돈상태": (-0.168, 0.101),
        "안전보건공단지원": (0.179, 0.080),
        "공사규모_1": (-0.365, 0.0005),
        "기성공정률_1": (-0.853, 0.0005),
        "기성공정률_2": (-0.370, 0.0005),
    }
    print(f"{'항':16s} {'논문β':>8s} {'재현β':>8s} {'SE':>7s} {'z':>7s} "
          f"{'논문p':>8s} {'재현p':>9s} {'OR':>7s}  플래그")
    for term, (pb, pp) in paper.items():
        if term not in M5.params.index:
            print(f"{term:16s}  -- 재현 모델에 해당 항 없음(기준범주/명명 차이) --")
            rec("7.5", term, "β", pb, None, "ABSENT")
            continue
        b = M5.params[term]; se = M5.bse[term]; z = M5.tvalues[term]
        p = M5.pvalues[term]; orr = np.exp(b)
        f_b = flag_num(pb, b, 0.05)
        f_p = flag_p(pp, p)
        print(f"{term:16s} {pb:8.3f} {b:8.3f} {se:7.3f} {z:7.3f} "
              f"{pp:8.3f} {p:9.4f} {orr:7.3f}  {f_b}/{f_p}")
        rec("7.5", term, "β", pb, round(b, 4), f_b)
        rec("7.5", term, "p", pp, round(p, 4), f_p)


# ── Table 7.6: 상호작용 ───────────────────────────────────────────────
def table_76(M5):
    hdr("Table 7.6  조절효과 상호작용 (M5 내)")
    paper = {
        "작업반장기여×전문지도": (-0.183, 0.055),
        "인증보유×고용노동부감독": (0.161, 0.063),
        "작업중지권×전문지도": (0.136, 0.151),
    }
    print(f"{'상호작용항':22s} {'논문β':>8s} {'재현β':>8s} {'논문p':>8s} {'재현p':>9s}  플래그")
    for term, (pb, pp) in paper.items():
        if term not in M5.params.index:
            print(f"{term:22s}  -- 재현 모델에 항 없음 --")
            rec("7.6", term, "β", pb, None, "ABSENT")
            continue
        b = M5.params[term]; p = M5.pvalues[term]
        f_b = flag_num(pb, b, 0.05); f_p = flag_p(pp, p)
        print(f"{term:22s} {pb:8.3f} {b:8.3f} {pp:8.3f} {p:9.4f}  {f_b}/{f_p}")
        rec("7.6", term, "β", pb, round(b, 4), f_b)
        rec("7.6", term, "p", pp, round(p, 4), f_p)


# ── Table 7.7: 모델 분류성능 ──────────────────────────────────────────
def table_77(df, y):
    hdr("Table 7.7  모델 분류성능  (셀21 방식: train/test 8:2 + 학습fold SMOTENC)")
    from sklearn.model_selection import (train_test_split, cross_val_score,
                                         RepeatedStratifiedKFold, StratifiedKFold)
    from sklearn.preprocessing import StandardScaler as SS
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score)
    from imblearn.over_sampling import SMOTENC
    from imblearn.pipeline import Pipeline as ImbPipeline
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    FEATURES = [c for c in df.columns if c != TARGET]
    CATEGORICAL_COLS = ["안전조직수준", "위원회수준", "인증보유", "위험성평가수준",
                        "전문지도", "고용노동부감독", "안전보건공단지원",
                        "공사규모", "발주처", "공사종류"]
    X_raw = df[FEATURES].copy()
    cat_idx = [X_raw.columns.get_loc(c) for c in CATEGORICAL_COLS]
    Xtr, Xte, ytr, yte = train_test_split(X_raw, y, test_size=0.2,
                                          random_state=42, stratify=y)
    sm_ = SMOTENC(categorical_features=cat_idx, random_state=42)
    Xtr_res, ytr_res = sm_.fit_resample(Xtr, ytr)
    scl = SS().fit(Xtr_res)
    Xtr_res_sc = scl.transform(Xtr_res); Xte_sc = scl.transform(Xte)

    def mk(name):
        if name == "LR":
            return LogisticRegression(max_iter=1000, random_state=42)
        if name == "RF":
            return RandomForestClassifier(n_estimators=200, max_depth=10,
                                          random_state=42, n_jobs=-1)
        if name == "XGB":
            return XGBClassifier(random_state=42, eval_metric="logloss",
                                 verbosity=0, n_jobs=-1)
        return LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)

    def pipe(name):
        steps = [("smote", SMOTENC(categorical_features=cat_idx, random_state=42))]
        if name == "LR":
            steps += [("sc", SS()), ("clf", mk(name))]
        else:
            steps += [("clf", mk(name))]
        return ImbPipeline(steps)

    cv_repo = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    cv_paper = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    paper = {  # (CV-F1, F1_test, AUC_test) 일부
        "RF": (0.521, 0.517, 0.705), "LR": (None, 0.466, 0.703),
    }
    print(f"{'모델':5s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUC':>6s} "
          f"{'CV-AUC(10x3)':>12s} {'CV-F1(5)':>9s} | 논문(CV-F1/F1/AUC)")
    for name in ["LR", "RF", "XGB", "LightGBM"]:
        key = "LightGBM" if name == "LightGBM" else name
        clf = mk("LightGBM" if name == "LightGBM" else name)
        if name == "LR":
            clf.fit(Xtr_res_sc, ytr_res)
            yp = clf.predict(Xte_sc); ypr = clf.predict_proba(Xte_sc)[:, 1]
        else:
            clf.fit(Xtr_res, ytr_res)
            yp = clf.predict(Xte); ypr = clf.predict_proba(Xte)[:, 1]
        acc = accuracy_score(yte, yp); pre = precision_score(yte, yp, zero_division=0)
        rec_ = recall_score(yte, yp, zero_division=0); f1 = f1_score(yte, yp, zero_division=0)
        auc = roc_auc_score(yte, ypr)
        pl = "LightGBM" if name == "LightGBM" else name
        cv_auc = cross_val_score(pipe("LightGBM" if name == "LightGBM" else name),
                                 X_raw, y, cv=cv_repo, scoring="roc_auc", n_jobs=-1).mean()
        cv_f1 = cross_val_score(pipe("LightGBM" if name == "LightGBM" else name),
                                X_raw, y, cv=cv_paper, scoring="f1", n_jobs=-1).mean()
        pk = paper.get("RF" if name == "RF" else ("LR" if name == "LR" else None))
        ptxt = ""
        flag = ""
        if pk:
            pcvf1, pf1, pauc = pk
            ptxt = f"{('%.3f'%pcvf1) if pcvf1 else '   -'}/{pf1:.3f}/{pauc:.3f}"
            flag = flag_num(pauc, auc, 0.02)
        print(f"{pl:5s} {acc:6.3f} {pre:6.3f} {rec_:6.3f} {f1:6.3f} {auc:6.3f} "
              f"{cv_auc:12.3f} {cv_f1:9.3f} | {ptxt} {flag}")
        rec("7.7", pl, "AUC_test", pk[2] if pk else None, round(auc, 3), flag)
        rec("7.7", pl, "CV-F1(5fold)", pk[0] if pk and pk[0] else None, round(cv_f1, 3), "")
        rec("7.7", pl, "F1_test", pk[1] if pk else None, round(f1, 3), "")
    print("주: CV-AUC=10-fold×3(레포 셀21), CV-F1=5-fold(논문 묘사) 별도 재계산.")


# ── Table 7.8: SHAP top10 ─────────────────────────────────────────────
def table_78(df, y):
    hdr("Table 7.8  SHAP 변수중요도 top10  (RF + TreeExplainer, 더미코딩 = 논문 명명 기준)")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTENC
    import shap

    # 논문은 더미명(기성공정률_1 등)을 보고 → 더미코딩 행렬로 SHAP 산출해 명명 일치시킴
    Z, D = make_design(df)
    # SHAP는 원본 스케일 의미가 중요하나, 여기선 논문 재현 목적상 추론설계와 동일 행렬 사용
    Xfull = pd.concat([Z, D], axis=1).astype(float)
    cat_like = [c for c in Xfull.columns if "_" in c]  # 더미열
    cat_idx = [Xfull.columns.get_loc(c) for c in cat_like]
    Xtr, Xte, ytr, yte = train_test_split(Xfull, y, test_size=0.2,
                                          random_state=42, stratify=y)
    try:
        sm_ = SMOTENC(categorical_features=cat_idx, random_state=42)
        Xtr_res, ytr_res = sm_.fit_resample(Xtr, ytr)
    except Exception:
        # 더미가 연속 표준화와 섞여 SMOTENC가 까다로우면 SMOTE 없이 fallback
        Xtr_res, ytr_res = Xtr, ytr
        print("주: SMOTENC 적용 곤란 → 원본 학습셋으로 RF 적합(주의).")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                random_state=42, n_jobs=-1).fit(Xtr_res, ytr_res)
    expl = shap.TreeExplainer(rf)
    sv = expl.shap_values(Xte)
    arr = sv[1] if isinstance(sv, list) else sv
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    imp = pd.Series(np.abs(arr).mean(axis=0), index=Xte.columns).sort_values(ascending=False)
    top10 = imp.head(10)
    paper_rank = ["기성공정률_1", "공사규모_1", "안전보건공단지원", "기성공정률_2",
                  "고용노동부감독", "발주처_1", "외국인비율", "공사종류_6",
                  "정리정돈상태", "위원회수준"]
    print(f"{'순위':>4s} {'재현 변수':18s} {'mean|SHAP|':>10s}   {'논문 순위 변수':18s}  일치")
    repro_rank = list(top10.index)
    for i in range(10):
        rv = repro_rank[i]; pv = paper_rank[i]
        match = "OK" if rv == pv else ("in10" if pv in repro_rank else "DIFF")
        print(f"{i+1:4d} {rv:18s} {top10.iloc[i]:10.4f}   {pv:18s}  {match}")
        rec("7.8", f"순위{i+1}", "변수", pv, rv, "OK" if rv == pv else "DIFF")
    # 정리정돈상태 순위 추적
    if "정리정돈상태" in repro_rank:
        print(f"주: 정리정돈상태 재현순위 = {repro_rank.index('정리정돈상태')+1}위 (논문 9위)")
    else:
        rk = list(imp.index).index("정리정돈상태") + 1 if "정리정돈상태" in imp.index else None
        print(f"주: 정리정돈상태 재현순위 = {rk}위 (top10 밖, 논문 9위)")


# ──────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(DATA)
    y = df[TARGET]
    print(f"데이터: {DATA.name} | n={len(df)} | 사고={int(y.sum())}({y.mean()*100:.1f}%) "
          f"비사고={int((y==0).sum())}")
    Z, D = make_design(df)
    print(f"더미열({len(D.columns)}개): {list(D.columns)}")

    table_71(df)
    table_72(df)
    models, M5, X5 = table_73_74(df, Z, D, y)
    table_75(M5)
    table_76(M5)
    try:
        table_77(df, y)
    except Exception as e:
        print(f"\n[Table 7.7 오류] {e}")
    try:
        table_78(df, y)
    except Exception as e:
        print(f"\n[Table 7.8 오류] {e}")

    # ── 종합 요약 ────────────────────────────────────────────────────
    hdr("종합 요약")
    out = pd.DataFrame(ROWS)
    out.to_csv(OUTCSV, index=False, encoding="utf-8-sig")
    diffs = out[out["플래그"].isin(["DIFF", "SIGN-FLIP", "ABSENT"]) |
                out["플래그"].astype(str).str.contains("FLIP|DIFF")]
    print(f"총 대조 항목: {len(out)} | 불일치 플래그: {len(diffs)}")
    print("\n[불일치 항목]")
    if len(diffs):
        for _, r in diffs.iterrows():
            print(f"  · {r['표']:4s} {str(r['항목'])[:18]:18s} {str(r['지표'])[:16]:16s} "
                  f"논문={r['논문값']} 재현={r['재현값']}  [{r['플래그']}]")
    else:
        print("  (없음)")
    print(f"\n결과 저장: {OUTCSV}")


if __name__ == "__main__":
    main()
