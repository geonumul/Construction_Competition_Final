# -*- coding: utf-8 -*-
"""
make_tables.py — 논문 표 7.1~7.8 을 '논문 표 구조' + n=1375 기준으로 재생성

설계
----
- 추론 LR(7.3~7.6): statsmodels Logit, 전체표본 n=1375, SMOTENC 미적용, Z표준화.
- 통제변수는 더미코딩(논문 3.2.4). 기준범주(reference):
    공사규모=2(중)·기성공정률=3(30-50%)  → 확정
    발주처=2·공사종류=7                  → ★코드북 미확보, 임시(플래그)
  주의: 발주처/공사종류 기준범주는 '그 변수의 더미 계수'에만 영향. R²/AIC/LR/타 계수 불변.
- 7.7 성능: 02_데이터분석.ipynb 셀21 방식(train/test 8:2 + 학습fold SMOTENC, 정수코딩 16변수).
  CV는 논문 본문대로 5-fold f1 로 통일해 재계산.
- 7.8 SHAP: 더미코딩 RF + TreeExplainer 로 재산출 → 논문 변수명(기성공정률_1 등)과 일치.
- 각 표 콘솔에 [논문원고값(n≈1100) / n=1375 재생성값] 대조 출력. 끼워맞춤 없음.

출력: outputs/tables/table71.csv ~ table78.csv
기존 노트북(01,02) 셀은 일절 수정하지 않는다(독립 스크립트).
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
ROOT = HERE.parent
DATA = ROOT / "data" / "전처리_최종.csv"
OUT = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

# ── 변수 그룹 ──────────────────────────────────────────────────────────
GROUP_A = ["안전조직수준", "위원회수준", "인증보유"]
GROUP_B = ["위험성평가수준", "교육훈련도움", "정리정돈상태", "작업중지권", "작업반장기여"]
MODERATORS = ["전문지도", "고용노동부감독", "안전보건공단지원"]
CAT_CTRL = ["공사규모", "발주처", "기성공정률", "공사종류"]
CONT_CTRL = ["외국인비율"]
TARGET = "사고발생"
HK = "정리정돈상태"
STD_VARS = GROUP_A + GROUP_B + MODERATORS + CONT_CTRL

REF = {"공사규모": 2, "기성공정률": 3, "발주처": 2, "공사종류": 7}
PROVISIONAL = {"발주처", "공사종류"}   # 기준범주 레이블 미확정


def fit_logit(X, y):
    return sm.Logit(y, sm.add_constant(X.astype(float))).fit(disp=0, maxiter=200)


def lr_test(full, reduced):
    chi2 = 2 * (full.llf - reduced.llf)
    dfd = int(full.df_model - reduced.df_model)
    p = stats.chi2.sf(chi2, dfd) if dfd > 0 else np.nan
    return chi2, dfd, p


def make_dummies(df):
    parts = []
    for c in CAT_CTRL:
        d = pd.get_dummies(df[c].astype(int), prefix=c).astype(float)
        ref_col = f"{c}_{REF[c]}"
        d = d[[col for col in d.columns if col != ref_col]]
        parts.append(d)
    return pd.concat(parts, axis=1)


def sig(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def hdr(t):
    print("\n" + "=" * 80 + f"\n {t}\n" + "=" * 80)


def save(df, name):
    path = OUT / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  저장: outputs/tables/{name}")


# ──────────────────────────────────────────────────────────────────────
def table71(df):
    hdr("표 7.1  연속·서열 변수 집단비교 (Welch t검정)")
    vars6 = ["위험성평가수준", "교육훈련도움", "정리정돈상태", "작업중지권", "작업반장기여", "외국인비율"]
    paper = {"정리정돈상태": (3.231, 0.001), "외국인비율": (-6.146, 0.0001),
             "위험성평가수준": (-2.560, 0.011), "작업반장기여": (2.113, 0.035),
             "교육훈련도움": (None, None), "작업중지권": (None, None)}
    a, b = df[df[TARGET] == 0], df[df[TARGET] == 1]
    rows = []
    print(f"{'변수':12s} {'사고군 M(SD)':>14s} {'비사고군 M(SD)':>14s} {'t(Welch)':>9s} "
          f"{'p':>8s} {'논문t/p':>14s}")
    for v in vars6:
        t, p = stats.ttest_ind(a[v], b[v], equal_var=False)
        pt, pp = paper[v]
        ptxt = f"{pt:+.3f}/{pp:.3f}" if pt is not None else "  (n.s.)"
        print(f"{v:12s} {b[v].mean():6.2f}({b[v].std():.2f}) {a[v].mean():6.2f}({a[v].std():.2f}) "
              f"{t:9.3f} {p:8.4f} {ptxt:>14s}")
        rows.append({"변수": v, "사고군_평균": round(b[v].mean(), 3), "사고군_SD": round(b[v].std(), 3),
                     "비사고군_평균": round(a[v].mean(), 3), "비사고군_SD": round(a[v].std(), 3),
                     "t_Welch": round(t, 3), "p값": round(p, 4), "유의도": sig(p),
                     "논문t_n1100": pt, "논문p_n1100": pp})
    print("주: t = (비발생군 − 발생군). Welch 등분산 미가정.")
    save(pd.DataFrame(rows), "table71.csv")


def table72(df):
    hdr("표 7.2  범주형 변수 카이제곱 검정")
    cats = ["안전조직수준", "위원회수준", "인증보유", "전문지도", "고용노동부감독",
            "안전보건공단지원", "공사규모", "발주처", "기성공정률", "공사종류"]
    paper_sig = {"고용노동부감독": True, "안전보건공단지원": True, "공사규모": True, "발주처": True,
                 "기성공정률": True, "공사종류": True, "안전조직수준": False, "위원회수준": False,
                 "인증보유": False, "전문지도": False}
    rows = []
    print(f"{'변수':12s} {'chi2':>10s} {'df':>3s} {'p값':>11s} {'재현유의':>7s} {'논문유의':>7s}")
    for v in cats:
        ct = pd.crosstab(df[v], df[TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        rsig = p < 0.001
        print(f"{v:12s} {chi2:10.3f} {dof:3d} {p:11.2e} {str(rsig):>7s} {str(paper_sig[v]):>7s}")
        rows.append({"변수": v, "chi2": round(chi2, 3), "df": dof, "p값": p,
                     "유의(p<.001)": rsig, "유의도": sig(p), "논문유의_n1100": paper_sig[v]})
    save(pd.DataFrame(rows), "table72.csv")


def build_hier(df):
    """위계 M1~M5 적합 후 반환."""
    Z = pd.DataFrame(StandardScaler().fit_transform(df[STD_VARS]), columns=STD_VARS, index=df.index)
    D = make_dummies(df)
    y = df[TARGET]
    Xa, Xb, Xm, Xfn = Z[GROUP_A], Z[GROUP_B], Z[MODERATORS], Z[CONT_CTRL]
    M1 = fit_logit(pd.concat([D, Xfn], axis=1), y)
    M2 = fit_logit(pd.concat([D, Xfn, Xa], axis=1), y)
    M3 = fit_logit(pd.concat([D, Xfn, Xa, Xb], axis=1), y)
    M4 = fit_logit(pd.concat([D, Xfn, Xa, Xb, Xm], axis=1), y)
    X5 = pd.concat([D, Xfn, Xa, Xb, Xm], axis=1).copy()
    for iv in GROUP_A + GROUP_B:
        for mo in MODERATORS:
            X5[f"{iv}×{mo}"] = Z[iv] * Z[mo]
    M5 = fit_logit(X5, y)
    return {"M1": M1, "M2": M2, "M3": M3, "M4": M4, "M5": M5}, D


def table73_74(models):
    hdr("표 7.3  위계적 로지스틱 M1~M5 (McFadden R²/AIC/BIC)")
    paper = {"M1": (0.150, 1151.41), "M5": (0.171, 1192.57)}
    rows = []
    print(f"{'모델':4s} {'k':>3s} {'R2':>8s} {'(논문R2)':>9s} {'AIC':>10s} {'(논문AIC)':>10s} {'BIC':>10s}")
    for nm, m in models.items():
        pr = paper.get(nm)
        pr2 = f"{pr[0]:9.3f}" if pr else "        -"
        pai = f"{pr[1]:10.2f}" if pr else "         -"
        print(f"{nm:4s} {int(m.df_model):3d} {m.prsquared:8.4f} {pr2} {m.aic:10.2f} {pai} {m.bic:10.2f}")
        rows.append({"모델": nm, "k": int(m.df_model), "McFadden_R2": round(m.prsquared, 4),
                     "AIC": round(m.aic, 2), "BIC": round(m.bic, 2),
                     "논문R2_n1100": pr[0] if pr else None, "논문AIC_n1100": pr[1] if pr else None})
    print("주: AIC는 n=1375에 비례→논문(n≈1100)보다 큼. R²는 표본크기에 둔감해 근접.")
    save(pd.DataFrame(rows), "table73.csv")

    hdr("표 7.4  위계 우도비검정 (논문: 전단계 비유의)")
    rows = []
    print(f"{'단계':8s} {'ΔR2':>9s} {'LR_chi2':>10s} {'df':>3s} {'p값':>9s} 재현유의")
    for nm, full, red in [("M1→M2", models["M2"], models["M1"]), ("M2→M3", models["M3"], models["M2"]),
                          ("M3→M4", models["M4"], models["M3"]), ("M4→M5", models["M5"], models["M4"])]:
        chi2, dfd, p = lr_test(full, red)
        dr2 = full.prsquared - red.prsquared
        print(f"{nm:8s} {dr2:+9.4f} {chi2:10.3f} {dfd:3d} {p:9.4f} {'유의' if p<.05 else '비유의'}")
        rows.append({"단계": nm, "ΔR2": round(dr2, 4), "LR_chi2": round(chi2, 3), "df": dfd,
                     "p값": round(p, 4), "유의": p < .05, "논문_n1100": "비유의"})
    save(pd.DataFrame(rows), "table74.csv")


def table75(M5):
    hdr("표 7.5  통합 M5 주효과 전체 [β, SE, z, p, OR, 95%CI]")
    paper = {"정리정돈상태": (-0.168, 0.101), "안전보건공단지원": (0.179, 0.080),
             "공사규모_1": (-0.365, 0.001), "기성공정률_1": (-0.853, 0.001), "기성공정률_2": (-0.370, 0.001)}
    terms = CONT_CTRL + GROUP_A + GROUP_B + MODERATORS + \
        [c for c in M5.params.index if any(c.startswith(f"{k}_") for k in CAT_CTRL)]
    rows = []
    print(f"{'항':16s} {'β':>8s} {'SE':>7s} {'z':>7s} {'p':>8s} {'OR':>7s} {'논문β/p':>13s}")
    for t in terms:
        if t not in M5.params.index:
            continue
        b, se, z, p = M5.params[t], M5.bse[t], M5.tvalues[t], M5.pvalues[t]
        lo, hi = np.exp(M5.conf_int().loc[t])
        prov = t.startswith(("발주처_", "공사종류_"))
        pp = paper.get(t)
        ptxt = f"{pp[0]:+.3f}/{pp[1]:.3f}" if pp else ""
        print(f"{'★' if prov else ' '}{t:15s} {b:8.3f} {se:7.3f} {z:7.3f} {p:8.4f} {np.exp(b):7.3f} "
              f"{sig(p):3s} {ptxt:>13s}")
        rows.append({"항": t, "β": round(b, 4), "SE": round(se, 4), "z": round(z, 4),
                     "p값": round(p, 4), "OR": round(np.exp(b), 4), "CI_low": round(lo, 4),
                     "CI_high": round(hi, 4), "유의도": sig(p),
                     "기준범주_임시": prov, "논문β_n1100": pp[0] if pp else None,
                     "논문p_n1100": pp[1] if pp else None})
    print("주: ★=발주처/공사종류 더미(기준범주 미확정·임시). 그 외 항은 기준범주와 무관하게 확정.")
    save(pd.DataFrame(rows), "table75.csv")


def table76(M5):
    hdr("표 7.6  상호작용 24개 (p<0.20 강조)")
    paper = {"작업반장기여×전문지도": (-0.183, 0.055), "인증보유×고용노동부감독": (0.161, 0.063),
             "작업중지권×전문지도": (0.136, 0.151)}
    terms = [f"{iv}×{mo}" for iv in GROUP_A + GROUP_B for mo in MODERATORS]
    rows = []
    for t in terms:
        b, p = M5.params[t], M5.pvalues[t]
        lo, hi = np.exp(M5.conf_int().loc[t])
        rows.append({"상호작용항": t, "β": round(b, 4), "SE": round(M5.bse[t], 4),
                     "p값": round(p, 4), "OR": round(np.exp(b), 4), "CI_low": round(lo, 4),
                     "CI_high": round(hi, 4), "유의도": sig(p),
                     "논문β_n1100": paper.get(t, (None,))[0],
                     "논문p_n1100": paper.get(t, (None, None))[1] if t in paper else None})
    rows.sort(key=lambda r: r["p값"])
    print(f"{'상호작용항':24s} {'β':>8s} {'p':>8s} {'OR':>7s} {'논문β/p':>13s}")
    for r in rows:
        pp = paper.get(r["상호작용항"])
        ptxt = f"{pp[0]:+.3f}/{pp[1]:.3f}" if pp else ""
        mark = "  ◀p<.20" if r["p값"] < 0.20 else ""
        print(f"{r['상호작용항']:24s} {r['β']:8.3f} {r['p값']:8.4f} {r['OR']:7.3f} {ptxt:>13s}{mark}")
    save(pd.DataFrame(rows), "table76.csv")


def table77(df):
    hdr("표 7.7  모델 분류성능 (셀21 방식 + CV는 5-fold f1)")
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler as SS
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
    from imblearn.over_sampling import SMOTENC
    from imblearn.pipeline import Pipeline as ImbPipeline
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    y = df[TARGET]
    FEATURES = [c for c in df.columns if c != TARGET]
    CATCOLS = ["안전조직수준", "위원회수준", "인증보유", "위험성평가수준", "전문지도",
               "고용노동부감독", "안전보건공단지원", "공사규모", "발주처", "공사종류"]
    X = df[FEATURES].copy()
    cat_idx = [X.columns.get_loc(c) for c in CATCOLS]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    Xtr_r, ytr_r = SMOTENC(categorical_features=cat_idx, random_state=42).fit_resample(Xtr, ytr)
    scl = SS().fit(Xtr_r)
    Xtr_sc, Xte_sc = scl.transform(Xtr_r), scl.transform(Xte)

    def mk(n):
        return {"로지스틱회귀": LogisticRegression(max_iter=1000, random_state=42),
                "랜덤포레스트": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
                "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0, n_jobs=-1),
                "LightGBM": LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)}[n]

    def pipe(n):
        steps = [("smote", SMOTENC(categorical_features=cat_idx, random_state=42))]
        steps += ([("sc", SS()), ("clf", mk(n))] if n == "로지스틱회귀" else [("clf", mk(n))])
        return ImbPipeline(steps)

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    paper = {"랜덤포레스트": (0.521, 0.517, 0.705), "로지스틱회귀": (None, 0.466, 0.703)}
    rows = []
    print(f"{'모델':9s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUC':>6s} {'CV-F1(5)':>9s} "
          f"{'논문CVF1/F1/AUC':>16s}")
    for n in ["로지스틱회귀", "랜덤포레스트", "XGBoost", "LightGBM"]:
        clf = mk(n)
        if n == "로지스틱회귀":
            clf.fit(Xtr_sc, ytr_r); yp = clf.predict(Xte_sc); ypr = clf.predict_proba(Xte_sc)[:, 1]
        else:
            clf.fit(Xtr_r, ytr_r); yp = clf.predict(Xte); ypr = clf.predict_proba(Xte)[:, 1]
        acc = accuracy_score(yte, yp); pre = precision_score(yte, yp, zero_division=0)
        rc = recall_score(yte, yp, zero_division=0); f1 = f1_score(yte, yp, zero_division=0)
        auc = roc_auc_score(yte, ypr)
        cvf1 = cross_val_score(pipe(n), X, y, cv=cv5, scoring="f1", n_jobs=-1).mean()
        pk = paper.get(n)
        ptxt = f"{(('%.3f'%pk[0]) if pk[0] else '-')}/{pk[1]:.3f}/{pk[2]:.3f}" if pk else ""
        print(f"{n:9s} {acc:6.3f} {pre:6.3f} {rc:6.3f} {f1:6.3f} {auc:6.3f} {cvf1:9.3f} {ptxt:>16s}")
        rows.append({"모델": n, "정확도": round(acc, 4), "정밀도": round(pre, 4), "재현율": round(rc, 4),
                     "F1": round(f1, 4), "AUC": round(auc, 4), "CV_F1_5fold": round(cvf1, 4),
                     "논문CVF1_n1100": pk[0] if pk else None, "논문F1_n1100": pk[1] if pk else None,
                     "논문AUC_n1100": pk[2] if pk else None})
    print("주: CV=StratifiedKFold(5), scoring=f1 (논문 본문 표기와 통일). 테스트셋은 8:2 단일분할.")
    save(pd.DataFrame(rows), "table77.csv")


def table78(df):
    hdr("표 7.8  SHAP 변수중요도 top10 (더미코딩 RF+TreeExplainer, 논문명 일치)")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTENC
    import shap

    y = df[TARGET]
    Z = pd.DataFrame(StandardScaler().fit_transform(df[STD_VARS]), columns=STD_VARS, index=df.index)
    D = make_dummies(df)
    Xfull = pd.concat([Z, D], axis=1).astype(float)
    cat_idx = [Xfull.columns.get_loc(c) for c in Xfull.columns if "_" in c]
    Xtr, Xte, ytr, yte = train_test_split(Xfull, y, test_size=0.2, random_state=42, stratify=y)
    Xtr_r, ytr_r = SMOTENC(categorical_features=cat_idx, random_state=42).fit_resample(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1).fit(Xtr_r, ytr_r)
    sv = shap.TreeExplainer(rf).shap_values(Xte)
    arr = sv[1] if isinstance(sv, list) else sv
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    imp = pd.Series(np.abs(arr).mean(0), index=Xte.columns).sort_values(ascending=False)
    paper_rank = ["기성공정률_1", "공사규모_1", "안전보건공단지원", "기성공정률_2", "고용노동부감독",
                  "발주처_1", "외국인비율", "공사종류_6", "정리정돈상태", "위원회수준"]
    rows = []
    print(f"{'순위':>3s} {'재현 변수':16s} {'mean|SHAP|':>10s}   {'논문 변수(n1100)':16s} 일치")
    for i in range(10):
        rv = imp.index[i]
        pv = paper_rank[i]
        match = "OK" if rv == pv else ("in10" if pv in imp.index[:10] else "DIFF")
        print(f"{i+1:3d} {rv:16s} {imp.iloc[i]:10.4f}   {pv:16s} {match}")
        rows.append({"순위": i + 1, "변수": rv, "mean_abs_SHAP": round(imp.iloc[i], 4),
                     "논문순위변수_n1100": pv, "일치": match})
    hk_rank = list(imp.index).index(HK) + 1 if HK in imp.index else None
    print(f"주: 정리정돈상태 재현순위={hk_rank}위 (논문 9위). ★발주처/공사종류 기준범주 임시.")
    save(pd.DataFrame(rows), "table78.csv")


def main():
    df = pd.read_csv(DATA)
    print(f"데이터: {DATA.name} | n={len(df)} | 사고={int(df[TARGET].sum())}({df[TARGET].mean()*100:.1f}%)")
    print("기준범주: 공사규모=2(중)·기성공정률=3(확정) / 발주처=2·공사종류=7(★임시·코드북 대기)")
    table71(df)
    table72(df)
    models, _ = build_hier(df)
    table73_74(models)
    table75(models["M5"])
    table76(models["M5"])
    table77(df)
    table78(df)
    print("\n완료: outputs/tables/table71.csv ~ table78.csv")


if __name__ == "__main__":
    main()
