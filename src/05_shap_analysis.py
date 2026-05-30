"""05 SHAP 분석 — table79, figures 73~76."""
import sys, io
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, str(Path(__file__).parent))
from utils_font import setup_style, reapply_font_to_current_axes

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
setup_style()

BASE = Path(__file__).resolve().parent.parent
TABLES = BASE / 'outputs' / 'tables'
FIGURES = BASE / 'outputs' / 'figures'
CACHE = BASE / 'outputs' / '_intermediate'

RANDOM_STATE = 42

prep = joblib.load(CACHE / 'preprocessing_cache.joblib')
ml = joblib.load(CACHE / 'ml_cache.joblib')

X_test = prep['X_test']
FEATURE_COLS = prep['FEATURE_COLS']
VARS_A = prep['VARS_A']
VARS_B = prep['VARS_B']
VARS_MOD = prep['VARS_MOD']
DUMMY_COLS = prep['DUMMY_COLS']
VARS_CONTROL_NUM = prep['VARS_CONTROL_NUM']

best_name = ml['best_name']
gs_objects = ml['gs_objects']
print(f'분석 대상 모델: {best_name}')


# ============ SHAP 계산 ============
if best_name == 'Logistic Regression':
    pipe = gs_objects[best_name].best_estimator_
    scaler = pipe.named_steps.get('scaler')
    if scaler is not None:
        X_for_shap = pd.DataFrame(
            scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index
        )
    else:
        X_for_shap = X_test
    explainer = shap.LinearExplainer(pipe.named_steps['clf'], X_for_shap)
    shap_raw = explainer.shap_values(X_for_shap)
    X_for_plot = X_for_shap
    tree_based = False
else:
    pipe = gs_objects[best_name].best_estimator_
    clf = pipe.named_steps['clf']
    explainer = shap.TreeExplainer(clf)
    shap_raw = explainer.shap_values(X_test)
    X_for_plot = X_test
    tree_based = True

# Normalize shape
if hasattr(shap_raw, 'values'):
    sv = shap_raw.values
    shap_values = sv[..., 1] if sv.ndim == 3 else sv
elif isinstance(shap_raw, list):
    shap_values = shap_raw[1]
elif np.ndim(shap_raw) == 3:
    shap_values = np.array(shap_raw)[..., 1]
else:
    shap_values = np.array(shap_raw)

print(f'SHAP shape: {shap_values.shape}')


# ============ Table 7.9 — Top 10 ============
control_set = set(DUMMY_COLS + VARS_CONTROL_NUM)


def vtype(v):
    if v in VARS_A:   return '독립 A (내부 안전관리)'
    if v in VARS_B:   return '독립 B (현장 안전행동)'
    if v in VARS_MOD: return '조절 (외부기관)'
    return '통제 (현장 특성)'


mean_abs = np.abs(shap_values).mean(axis=0)
t79_all = pd.DataFrame({'변수': FEATURE_COLS, 'mean|SHAP|': np.round(mean_abs, 5)})
t79_all = t79_all.sort_values('mean|SHAP|', ascending=False).reset_index(drop=True)
t79 = t79_all.head(10).copy()
t79.insert(0, '순위', np.arange(1, len(t79) + 1))
t79['변수 유형'] = t79['변수'].apply(vtype)
t79 = t79[['순위', '변수', '변수 유형', 'mean|SHAP|']]
t79.to_csv(TABLES / 'table79_shap_importance.csv', index=False, encoding='utf-8-sig')
print(f'\n[Table 7.9] Top 10 ({best_name})')
print(t79.to_string(index=False))


# ============ Figure 7.3 — Bar Plot Top 10 (회색 단색) ============
bar = t79.iloc[::-1].copy()
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(bar['변수'], bar['mean|SHAP|'],
        color='#606060', edgecolor='black', linewidth=0.8, height=0.72)
xmax = bar['mean|SHAP|'].max()
for i, val in enumerate(bar['mean|SHAP|']):
    ax.text(val + xmax * 0.01, i, f'{val:.4f}',
            va='center', fontsize=10, color='black')
ax.set_xlim(0, xmax * 1.15)
ax.set_xlabel('mean(|SHAP value|)')
ax.grid(axis='x', alpha=0.3)
fig.text(0.5, -0.02,
         f'그림 7.3. SHAP 평균 절댓값 기반 변수 중요도({best_name}, Top 10)',
         ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(FIGURES / 'figure73_shap_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print('\n✓ figure73_shap_bar.png')


# ============ Figure 7.4(a) — Summary Plot 전체 27변수 (회색조) ============
shap.summary_plot(shap_values, X_for_plot, feature_names=FEATURE_COLS,
                   show=False, max_display=27,
                   cmap=plt.get_cmap('Greys'))
fig = plt.gcf(); fig.set_size_inches(10, 9.5)
reapply_font_to_current_axes()
fig.text(0.5, -0.005, '그림 7.4(a). SHAP Summary Plot — 전체 27변수',
         ha='center', fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0.005, 1, 0.99])
fig.savefig(FIGURES / 'figure74a_shap_summary_all.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ figure74a_shap_summary_all.png')


# ============ Figure 7.4(b) — RQ 11변수 (통제변수 마스킹) ============
focus_vars = VARS_A + VARS_B + VARS_MOD   # 11
focus_idx = [FEATURE_COLS.index(v) for v in focus_vars]
shap_focus = shap_values[:, focus_idx]
X_focus = X_for_plot.iloc[:, focus_idx]

shap.summary_plot(shap_focus, X_focus, feature_names=focus_vars,
                   show=False, max_display=11,
                   cmap=plt.get_cmap('Greys'))
fig = plt.gcf(); fig.set_size_inches(9.5, 6.5)
reapply_font_to_current_axes()
fig.text(0.5, -0.005,
         '그림 7.4(b). SHAP Summary Plot — 독립·조절 11변수 (통제변수 시각적 마스킹)',
         ha='center', fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0.005, 1, 0.99])
fig.savefig(FIGURES / 'figure74b_shap_summary_rqonly.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ figure74b_shap_summary_rqonly.png')


# ============ Figure 7.5 — Dependence (정리정돈) interaction_index=None ============
key = '정리정돈상태'
key_idx = FEATURE_COLS.index(key)
shap.dependence_plot(key_idx, shap_values, X_for_plot,
                      feature_names=FEATURE_COLS,
                      interaction_index=None,
                      color='#303030',
                      show=False)
fig = plt.gcf(); fig.set_size_inches(8, 5.5)
ax = plt.gca()
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlim(0.5, 5.5)
ax.grid(alpha=0.3)
ax.axhline(0, color='black', linestyle=':', linewidth=0.8)
reapply_font_to_current_axes()
fig.text(0.5, -0.005, '그림 7.5. 정리정돈상태의 SHAP 임계값 및 Dependence 분석',
         ha='center', fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0.005, 1, 0.99])
fig.savefig(FIGURES / 'figure75_shap_dependence_정리정돈.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ figure75_shap_dependence_정리정돈.png')


# ============ Figure 7.6 — Interaction Heatmap 8×3 ============
if tree_based:
    n_sample = min(200, len(X_for_plot))
    X_int = X_for_plot.sample(n=n_sample, random_state=RANDOM_STATE)
    inter_raw = explainer.shap_interaction_values(X_int)
    if hasattr(inter_raw, 'values'):
        iv = inter_raw.values
        inter_vals = iv[..., 1] if iv.ndim == 4 else iv
    elif isinstance(inter_raw, list):
        inter_vals = inter_raw[1]
    elif np.ndim(inter_raw) == 4:
        inter_vals = np.array(inter_raw)[..., 1]
    else:
        inter_vals = np.array(inter_raw)

    ivs = VARS_A + VARS_B   # 8
    mods = VARS_MOD          # 3
    mat = np.zeros((len(ivs), len(mods)))
    for i, x in enumerate(ivs):
        for j, m in enumerate(mods):
            ii = FEATURE_COLS.index(x)
            jj = FEATURE_COLS.index(m)
            mat[i, j] = np.abs(inter_vals[:, ii, jj]).mean()

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    im = ax.imshow(mat, cmap='Greys', aspect='auto')
    for i in range(len(ivs)):
        for j in range(len(mods)):
            v = mat[i, j]
            text_color = 'white' if v > mat.max() * 0.55 else 'black'
            ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                    fontsize=10, color=text_color)
    ax.set_xticks(range(len(mods)))
    ax.set_xticklabels(mods, fontsize=10)
    ax.set_yticks(range(len(ivs)))
    ax.set_yticklabels(ivs, fontsize=10)
    ax.set_xlabel('조절변수')
    ax.set_ylabel('독립변수')
    ax.set_xticks(np.arange(len(mods) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(ivs) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linewidth=0.8)
    ax.tick_params(which='minor', length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('mean|SHAP interaction|')
    fig.text(0.5, -0.005,
             '그림 7.6. SHAP 기반 조절효과 상호작용 히트맵(8 독립 × 3 조절)',
             ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.005, 1, 0.99])
    fig.savefig(FIGURES / 'figure76_shap_interaction_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✓ figure76_shap_interaction_heatmap.png')
else:
    print('⚠ 최적이 LR이라 SHAP interaction values 미지원. figure76 생략.')


# ============ SHAP 캐시 ============
_cache = {
    'shap_values': shap_values,
    'X_for_plot': X_for_plot,
    'top10': t79.to_dict('records'),
    'mean_abs_all': mean_abs,
    'feature_cols': FEATURE_COLS,
}
if tree_based:
    _cache['interaction_mat'] = mat              # 8×3
    _cache['interaction_rows'] = VARS_A + VARS_B
    _cache['interaction_cols'] = VARS_MOD
joblib.dump(_cache, CACHE / 'shap_cache.joblib')
print('\n✓ shap_cache.joblib 저장')
print('05_shap_analysis 완료')
