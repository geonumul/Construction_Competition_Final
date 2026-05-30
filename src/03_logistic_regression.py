"""03 위계적 LR + VIF — table73~77."""
import sys, io, itertools
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils_font import setup_style

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data' / 'processed_2021.csv'
TABLES = BASE / 'outputs' / 'tables'
CACHE = BASE / 'outputs' / '_intermediate'
TABLES.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

df = pd.read_csv(DATA)
TARGET = 'accident'

VARS_A = ['안전조직수준', '위원회수준', '인증보유']
VARS_B = ['위험성평가수준', '교육훈련도움', '정리정돈상태', '작업중지권', '작업반장기여']
VARS_MOD = ['전문지도', '고용노동부감독', '안전보건공단지원']
VARS_CONTROL_NOM = ['공사규모', '발주처', '기성공정률', '공사종류']
VARS_CONTROL_NUM = ['외국인비율']

BASELINES = {
    '공사규모':   2,   # 중규모 (120~800억)
    '발주처':     2,   # 사기업·개인
    '기성공정률': 3,   # 30~50%
    '공사종류':   7,   # 전기·통신·기타
}


def make_dummies(data, baselines):
    out = data.copy()
    for col, baseline in baselines.items():
        d = pd.get_dummies(out[col], prefix=col, dtype=int)
        drop = f'{col}_{baseline}'
        if drop in d.columns:
            d = d.drop(columns=drop)
        out = out.drop(columns=col)
        out = pd.concat([out, d], axis=1)
    return out


df_enc = make_dummies(df, BASELINES)

DUMMY_COLS = []
for col, baseline in BASELINES.items():
    for v in sorted(df[col].unique()):
        if v != baseline:
            DUMMY_COLS.append(f'{col}_{v}')

CONTROL_COLS = DUMMY_COLS + VARS_CONTROL_NUM
FEATURE_COLS = CONTROL_COLS + VARS_A + VARS_B + VARS_MOD
print(f'더미 컬럼 ({len(DUMMY_COLS)}): {DUMMY_COLS}')
print(f'CONTROL_COLS ({len(CONTROL_COLS)}) | FEATURE_COLS ({len(FEATURE_COLS)})')


# Train/Test split (8:2 stratified)
X = df_enc[FEATURE_COLS].astype(float).copy()
y = df_enc[TARGET].astype(int).copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f'Train: {X_train.shape}  Test: {X_test.shape}')

# Z-score (fit train, transform test)
scaler = StandardScaler()
X_train_z = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index)
X_test_z  = pd.DataFrame(scaler.transform(X_test),       columns=FEATURE_COLS, index=X_test.index)


# ============ Table 7.3 — VIF (z-scored 학습셋, 전체 27 + 더미 11 = 27) ============
X_vif = add_constant(X_train_z)
vif_rows = []
for i, col in enumerate(X_vif.columns):
    if col == 'const':
        continue
    vif = variance_inflation_factor(X_vif.values, i)
    vif_rows.append({'변수': col, 'VIF': round(vif, 3)})
t73 = pd.DataFrame(vif_rows).sort_values('VIF', ascending=False).reset_index(drop=True)
t73.insert(0, '순위', np.arange(1, len(t73) + 1))
t73.to_csv(TABLES / 'table73_vif.csv', index=False, encoding='utf-8-sig')
print(f'\n[Table 7.3] 최대 VIF = {t73.VIF.max():.3f} / 기준 < 5')
print(t73.head(5).to_string(index=False))
print(t73.tail(2).to_string(index=False))


# ============ 위계적 LR 모형 (M1~M5) ============
INTERACTION_PAIRS = list(itertools.product(VARS_A + VARS_B, VARS_MOD))  # 8 × 3
INTERACTION_COLS = [f'{x}__x__{m}' for x, m in INTERACTION_PAIRS]

X_train_full = X_train_z.copy()
for x, m in INTERACTION_PAIRS:
    X_train_full[f'{x}__x__{m}'] = X_train_z[x] * X_train_z[m]

cols_M1 = CONTROL_COLS
cols_M2 = cols_M1 + VARS_A
cols_M3 = cols_M2 + VARS_B
cols_M4 = cols_M3 + VARS_MOD
cols_M5 = cols_M4 + INTERACTION_COLS


def fit(cols):
    X_ = add_constant(X_train_full[cols].astype(float))
    return sm.Logit(y_train, X_).fit(disp=False, maxiter=200)


m1, m2, m3, m4, m5 = (fit(c) for c in (cols_M1, cols_M2, cols_M3, cols_M4, cols_M5))


# ============ Table 7.4 — Hierarchical LR ============
t74_rows = []
for name, model, group in [
    ('M1', m1, '통제변수'),
    ('M2', m2, '+ 독립A(내부관리)'),
    ('M3', m3, '+ 독립A + B(안전행동)'),
    ('M4', m4, '+ 조절변수'),
    ('M5', m5, '+ 상호작용항'),
]:
    t74_rows.append({
        '모형': name,
        '투입 변수군': group,
        'Pseudo R²': round(model.prsquared, 4),
        'AIC': round(model.aic, 2),
        'BIC': round(model.bic, 2),
    })
t74 = pd.DataFrame(t74_rows)
t74.to_csv(TABLES / 'table74_hierarchical_lr.csv', index=False, encoding='utf-8-sig')
print('\n[Table 7.4] 위계적 LR 모형 비교')
print(t74.to_string(index=False))


# ============ Table 7.5 — LR Test ============
def lr_test(restricted, full):
    lr_stat = 2 * (full.llf - restricted.llf)
    df_diff = full.df_model - restricted.df_model
    p = 1 - st.chi2.cdf(lr_stat, df_diff)
    dr2 = full.prsquared - restricted.prsquared
    return lr_stat, int(df_diff), p, dr2


comparisons = [
    ('M1 → M2 (독립A 추가)', m1, m2),
    ('M2 → M3 (독립B 추가)', m2, m3),
    ('M3 → M4 (조절변수 추가)', m3, m4),
    ('M4 → M5 (상호작용 추가)', m4, m5),
]
t75_rows = []
for name, r, f in comparisons:
    lr, dof, p, dr2 = lr_test(r, f)
    sig_str = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    t75_rows.append({
        '비교': name,
        'ΔR²': round(dr2, 4),
        'LR statistic': round(lr, 3),
        'df': dof,
        'p-value': round(p, 4),
        '유의도': sig_str,
    })
t75 = pd.DataFrame(t75_rows)
t75.to_csv(TABLES / 'table75_lr_test.csv', index=False, encoding='utf-8-sig')
print('\n[Table 7.5] 우도비 검정')
print(t75.to_string(index=False))


# ============ Table 7.6 — M5 주효과 ============
def coef_table(model, cols):
    out = []
    for c in cols:
        if c not in model.params.index:
            continue
        b = model.params[c]
        se = model.bse[c]
        z = model.tvalues[c]
        p = model.pvalues[c]
        ci = model.conf_int().loc[c]
        sig_str = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05
                   else '†' if p < 0.10 else '')
        out.append({
            '변수': c,
            'β': round(b, 4),
            'SE': round(se, 4),
            'z': round(z, 3),
            'p-value': round(p, 4),
            'OR': round(np.exp(b), 4),
            '95%CI 하한': round(np.exp(ci[0]), 4),
            '95%CI 상한': round(np.exp(ci[1]), 4),
            '유의도': sig_str,
        })
    return pd.DataFrame(out)


t76 = coef_table(m5, ['const'] + cols_M4)
t76.to_csv(TABLES / 'table76_m5_coefficients.csv', index=False, encoding='utf-8-sig')
print(f'\n[Table 7.6] M5 주효과 (총 {len(t76)} 행)')
key_rows = t76[t76['변수'].isin(['정리정돈상태', '안전보건공단지원', '공사규모_1', '기성공정률_1', '기성공정률_2'])]
print(key_rows.to_string(index=False))


# ============ Table 7.7 — 24개 상호작용항 ============
t77 = coef_table(m5, INTERACTION_COLS)
t77 = t77.sort_values('p-value').reset_index(drop=True)
t77['상호작용항'] = t77['변수'].str.replace('__x__', ' × ', regex=False)
t77 = t77[['상호작용항', 'β', 'SE', 'z', 'p-value', 'OR', '95%CI 하한', '95%CI 상한', '유의도']]
t77.to_csv(TABLES / 'table77_interactions.csv', index=False, encoding='utf-8-sig')
print('\n[Table 7.7] 24개 상호작용항 (p 오름차순 Top 5)')
print(t77.head(5).to_string(index=False))


# ============ 캐시 저장 ============
joblib.dump({
    'X_train': X_train, 'X_test': X_test,
    'y_train': y_train, 'y_test': y_test,
    'X_train_z': X_train_z, 'X_test_z': X_test_z,
    'FEATURE_COLS': FEATURE_COLS,
    'DUMMY_COLS': DUMMY_COLS,
    'CONTROL_COLS': CONTROL_COLS,
    'VARS_A': VARS_A, 'VARS_B': VARS_B, 'VARS_MOD': VARS_MOD,
    'VARS_CONTROL_NUM': VARS_CONTROL_NUM,
    'INTERACTION_COLS': INTERACTION_COLS,
    'scaler': scaler,
    'm4_pvalues': m4.pvalues,
    'm5_results': {
        '정리정돈상태_OR': float(np.exp(m5.params['정리정돈상태'])),
        '정리정돈상태_p':  float(m5.pvalues['정리정돈상태']),
        '인증_고용감독_OR': float(np.exp(m5.params['인증보유__x__고용노동부감독'])),
        '인증_고용감독_p':  float(m5.pvalues['인증보유__x__고용노동부감독']),
    },
}, CACHE / 'preprocessing_cache.joblib')
print('\n✓ preprocessing_cache.joblib 저장')
print('03_logistic_regression 완료')
