# -*- coding: utf-8 -*-
"""
regenerate_tables_357.py

목적
----
data/전처리_최종.csv (n=1375)를 최종 표본으로 확정하고, 논문 Table 7.3 / 7.5 / 7.6 을
논문 기준범주(reference category)로 재계산해 교체용 표를 생성한다.
기존 노트북 셀은 일절 수정하지 않는다.

설계 (verify_paper_tables.py 방식 재사용)
- statsmodels Logit, 전체표본 n=1375, SMOTENC 미적용.
- 연속/서열(독립8 + 조절3 + 외국인비율) Z표준화. 통제 4종은 더미코딩.
- M1=통제만, M2=+내부관리(A), M3=+현장행동(B), M4=+조절, M5=+상호작용24.

★ 기준범주(reference) — 논문 Table 7.5 주석 기준
   공사규모 : 2(중)        ← docs L71에서 1=소/2=중/3=대 확인됨 (확정)
   기성공정률: 3(30-50% 구간) ← 논문 주석 (확정)
   발주처   : '사기업/개인' 코드 ← ★레포에 코드북 없음 → 미확정. 아래 PROV_발주처 로 임시 지정
   공사종류 : '전기/통신/기타' 코드 ← ★레포에 코드북 없음 → 미확정. 아래 PROV_공사종류 로 임시 지정

   주의: 발주처/공사종류 기준범주 선택은 '그 변수의 더미 계수 레이블'에만 영향을 주며,
         Table 7.3(R²/AIC/BIC), Table 7.4, Table 7.6, 그리고 정리정돈상태·안전보건공단지원·
         모든 상호작용 계수/p값에는 전혀 영향을 주지 않는다(동일 모델 재매개변수화).
         코드북 확보 시 PROV_* 만 바꾸면 7.5의 발주처/공사종류 행이 정확히 갱신된다.
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
DATA = ROOT / "data" / "전처리_최종.csv"
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTCSV = OUTDIR / "regenerated_357.csv"

GROUP_A = ["안전조직수준", "위원회수준", "인증보유"]
GROUP_B = ["위험성평가수준", "교육훈련도움", "정리정돈상태", "작업중지권", "작업반장기여"]
MODERATORS = ["전문지도", "고용노동부감독", "안전보건공단지원"]
CAT_CTRL = ["공사규모", "발주처", "기성공정률", "공사종류"]
CONT_CTRL = ["외국인비율"]
TARGET = "사고발생"
HK = "정리정돈상태"
STD_VARS = GROUP_A + GROUP_B + MODERATORS + CONT_CTRL

# ── 기준범주 지정 ──────────────────────────────────────────────────────
REF = {
    "공사규모": 2,    # 중 (확정)
    "기성공정률": 3,  # 30-50% 구간 (확정)
    "발주처": 2,      # ★PROV: '사기업/개인' 미확정 → 임시(빈도 최다 범주). 코드북 확보 시 교체
    "공사종류": 7,    # ★PROV: '전기/통신/기타' 미확정 → 임시. 코드북 확보 시 교체
}
PROVISIONAL = {"발주처", "공사종류"}  # 레이블 미확정 표시용

ROWS = []


def rec(table, item, metric, paper, repro, note=""):
    ROWS.append({"표": table, "항목": item, "지표": metric,
                 "논문값": paper, "재현값": repro, "비고": note})


def hdr(t):
    print("\n" + "=" * 82)
    print(" " + t)
    print("=" * 82)


def make_dummies(df):
    """REF 기준범주를 제거한 더미 생성. 빈도도 함께 반환."""
    parts, freq = [], []
    for c in CAT_CTRL:
        d = pd.get_dummies(df[c].astype(int), prefix=c).astype(float)
        ref_col = f"{c}_{REF[c]}"
        prov = " (★기준범주 미확정·임시)" if c in PROVISIONAL else ""
        if ref_col not in d.columns:
            raise ValueError(f"{c}: 기준범주 코드 {REF[c]} 가 데이터에 없음")
        kept = [col for col in d.columns if col != ref_col]
        for col in d.columns:
            code = col.split("_")[-1]
            n = int(d[col].sum())
            is_ref = (col == ref_col)
            freq.append((col, code, n, n / len(df) * 100, is_ref, c in PROVISIONAL))
        d = d[kept]
        parts.append(d)
        _ = prov
    D = pd.concat(parts, axis=1)
    return D, freq


def fit_logit(X, y):
    return sm.Logit(y, sm.add_constant(X.astype(float))).fit(disp=0, maxiter=200)


def lr_test(full, reduced):
    chi2 = 2 * (full.llf - reduced.llf)
    dfd = int(full.df_model - reduced.df_model)
    p = stats.chi2.sf(chi2, dfd) if dfd > 0 else np.nan
    return chi2, dfd, p


def coefrow(m, term):
    b = m.params[term]; se = m.bse[term]; z = m.tvalues[term]
    p = m.pvalues[term]; lo, hi = m.conf_int().loc[term]
    return dict(beta=b, se=se, z=z, p=p, OR=np.exp(b),
               cilo=np.exp(lo), cihi=np.exp(hi))


def main():
    df = pd.read_csv(DATA)
    y = df[TARGET]
    print(f"데이터: {DATA.name} | n={len(df)} | 사고={int(y.sum())}({y.mean()*100:.1f}%)")

    # 표준화
    Z = pd.DataFrame(StandardScaler().fit_transform(df[STD_VARS]),
                     columns=STD_VARS, index=df.index)
    D, freq = make_dummies(df)

    # ── STEP 1: 더미 빈도 ────────────────────────────────────────────
    hdr("STEP 1  더미 변수 빈도 (기준범주 = 제거)")
    print(f"{'변수_코드':16s} {'n':>6s} {'%':>7s}  비고")
    for col, code, n, pct, is_ref, prov in freq:
        tag = "← 기준범주(제거)" if is_ref else "더미 생성"
        if prov:
            tag += "  [★레이블 미확정]"
        print(f"{col:16s} {n:6d} {pct:6.1f}%  {tag}")
    print("\n확정: 공사규모 기준=2(중), 기성공정률 기준=3(30-50%)")
    print("미확정(임시): 발주처 기준=2, 공사종류 기준=7  ← 코드북 확보 후 PROV_* 교체 필요")

    # ── STEP 2: 위계 M1~M5 ───────────────────────────────────────────
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
    models = {"M1": M1, "M2": M2, "M3": M3, "M4": M4, "M5": M5}

    hdr("Table 7.3 (재계산)  위계 로지스틱 M1~M5  [논문값 / n=1375 재현값 / 방향]")
    paper73 = {"M1": (0.150, 1151.41), "M5": (0.171, 1192.57)}
    print(f"{'모델':4s} {'k':>3s} {'R2재현':>8s} {'(논문R2)':>8s} {'AIC재현':>10s} "
          f"{'(논문AIC)':>10s} {'BIC재현':>10s}")
    for nm, m in models.items():
        pr = paper73.get(nm)
        pr2 = f"{pr[0]:8.3f}" if pr else "       -"
        pai = f"{pr[1]:10.2f}" if pr else "         -"
        print(f"{nm:4s} {int(m.df_model):3d} {m.prsquared:8.4f} {pr2} "
              f"{m.aic:10.2f} {pai} {m.bic:10.2f}")
        rec("7.3", nm, "McFadden_R2", pr[0] if pr else None, round(m.prsquared, 4))
        rec("7.3", nm, "AIC", pr[1] if pr else None, round(m.aic, 2))
        rec("7.3", nm, "BIC", None, round(m.bic, 2))
    print("주: AIC는 표본크기(n=1375)에 비례 → 논문(n≈1100)보다 큼. R²는 표본크기에 둔감해 근접.")

    # 위계 LR검정 (참고)
    print("\n[위계 우도비검정]")
    for nm, full, red in [("M1→M2", M2, M1), ("M2→M3", M3, M2),
                          ("M3→M4", M4, M3), ("M4→M5", M5, M4)]:
        chi2, dfd, p = lr_test(full, red)
        print(f"  {nm}: ΔR2={full.prsquared-red.prsquared:+.4f}  "
              f"chi2={chi2:.3f} df={dfd} p={p:.4f} {'(유의)' if p<.05 else '(비유의)'}")
        rec("7.4", nm, "LR_p", None, round(p, 4))

    # ── Table 7.5: M5 주효과 전체 ────────────────────────────────────
    hdr("Table 7.5 (재계산)  M5 주효과 전체  [β, SE, z, p, OR, 95%CI]")
    paper75 = {  # 논문 보고 일부 (β, p)
        "정리정돈상태": (-0.168, 0.101), "안전보건공단지원": (0.179, 0.080),
        "공사규모_1": (-0.365, 0.001), "기성공정률_1": (-0.853, 0.001),
        "기성공정률_2": (-0.370, 0.001),
    }
    main_terms = (CONT_CTRL + GROUP_A + GROUP_B + MODERATORS +
                  [c for c in M5.params.index if any(c.startswith(f"{k}_") for k in CAT_CTRL)])
    print(f"{'항':16s} {'β':>8s} {'SE':>7s} {'z':>7s} {'p':>8s} {'OR':>7s} "
          f"{'95%CI':>16s} {'논문β/p':>14s}")
    for term in main_terms:
        if term not in M5.params.index:
            continue
        r = coefrow(M5, term)
        prov = "★" if any(term.startswith(f"{k}_") and k in PROVISIONAL for k in CAT_CTRL) else " "
        pp = paper75.get(term)
        ptxt = f"{pp[0]:+.3f}/{pp[1]:.3f}" if pp else ""
        sig = "***" if r["p"] < .001 else "**" if r["p"] < .01 else "*" if r["p"] < .05 else ""
        print(f"{prov}{term:15s} {r['beta']:8.3f} {r['se']:7.3f} {r['z']:7.3f} "
              f"{r['p']:8.4f} {r['OR']:7.3f} [{r['cilo']:.3f},{r['cihi']:.3f}] "
              f"{sig:3s} {ptxt:>14s}")
        note = "발주처/공사종류는 기준범주 미확정" if prov == "★" else ""
        rec("7.5", term, "β", pp[0] if pp else None, round(r["beta"], 4), note)
        rec("7.5", term, "p", pp[1] if pp else None, round(r["p"], 4), note)
    print("주: ★ = 발주처/공사종류 더미(기준범주 미확정·임시). 그 외 항은 기준범주와 무관하게 확정.")

    # ── Table 7.6: 상호작용 24개 ─────────────────────────────────────
    hdr("Table 7.6 (재계산)  상호작용 24개 전체")
    paper76 = {"작업반장기여×전문지도": (-0.183, 0.055),
               "인증보유×고용노동부감독": (0.161, 0.063),
               "작업중지권×전문지도": (0.136, 0.151)}
    int_terms = [f"{iv}×{mo}" for iv in GROUP_A + GROUP_B for mo in MODERATORS]
    rows76 = []
    for term in int_terms:
        r = coefrow(M5, term)
        rows76.append((term, r))
        rec("7.6", term, "β", paper76.get(term, (None,))[0], round(r["beta"], 4))
        rec("7.6", term, "p", paper76.get(term, (None, None))[1] if term in paper76 else None,
            round(r["p"], 4))
    rows76.sort(key=lambda t: t[1]["p"])
    print(f"{'상호작용항':24s} {'β':>8s} {'p':>8s} {'OR':>7s}  {'논문β/p':>14s}")
    for term, r in rows76:
        pp = paper76.get(term)
        ptxt = f"{pp[0]:+.3f}/{pp[1]:.3f}" if pp else ""
        mark = "  ◀ p<0.20" if r["p"] < 0.20 else ""
        print(f"{term:24s} {r['beta']:8.3f} {r['p']:8.4f} {r['OR']:7.3f}  "
              f"{ptxt:>14s}{mark}")

    # ── STEP 3: 서사 직결 항목 ───────────────────────────────────────
    hdr("STEP 3  서사 직결 항목 (8.2 / 8.3 / 9.1)")
    def show(term, paper_b, paper_p, paper_dir):
        r = coefrow(M5, term)
        sign = "+" if r["beta"] > 0 else "-"
        sigchg = ("유의" if r["p"] < .05 else "비유의")
        keep = "부호 유지" if ((paper_b > 0) == (r["beta"] > 0)) else "★부호 뒤집힘★"
        print(f"\n● {term}")
        print(f"   논문 : β={paper_b:+.3f}, p={paper_p:.3f} ({paper_dir})")
        print(f"   재현 : β={r['beta']:+.3f} ({sign}), SE={r['se']:.3f}, "
              f"p={r['p']:.4f} → {sigchg}, OR={r['OR']:.3f} [{r['cilo']:.3f},{r['cihi']:.3f}]")
        print(f"   판정 : {keep}")
        rec("STEP3", term, "부호/유의", f"{paper_b:+.3f}/p{paper_p:.3f}",
            f"{r['beta']:+.3f}/p{r['p']:.4f}", keep)

    show("정리정돈상태", -0.168, 0.101, "비유의→유의 전환 확인")
    show("인증보유×고용노동부감독", 0.161, 0.063, "+방향, ★부호 유지 필수")
    show("작업반장기여×전문지도", -0.183, 0.055, "-방향")
    show("안전보건공단지원", 0.179, 0.080, "주효과")

    # ── 저장 ─────────────────────────────────────────────────────────
    pd.DataFrame(ROWS).to_csv(OUTCSV, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {OUTCSV}")


if __name__ == "__main__":
    main()
