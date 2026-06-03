"""논문 제출용 4개 그림 — results/ 에 출력.

기존 분석 결과(ml_cache, shap_cache) 그대로 활용. 시각화만 새 사양.
- Malgun Gothic 한글 / Times New Roman 영문
- 정확한 mm 치수, 카테고리 색상, B&W 호환
- 11_roc_curves.png, 13_shap_bar.png, 12_shap_dependence_housekeeping.png, 14_shap_interaction_heatmap.png
"""
import sys, io
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============ 글로벌 스타일 (한글 Malgun, 영문 Times) ============
def _setup_paper_style():
    installed = {f.name for f in fm.fontManager.ttflist}
    family_kr = 'Malgun Gothic' if 'Malgun Gothic' in installed else (
        'Noto Sans KR' if 'Noto Sans KR' in installed else 'DejaVu Sans')
    # Times New Roman 우선, fallback Latin → Malgun
    plt.rcParams['font.family']      = family_kr
    plt.rcParams['font.sans-serif']  = [family_kr, 'DejaVu Sans']
    plt.rcParams['font.serif']       = ['Times New Roman', family_kr, 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size']        = 8
    plt.rcParams['axes.labelsize']   = 8
    plt.rcParams['xtick.labelsize']  = 7
    plt.rcParams['ytick.labelsize']  = 7
    plt.rcParams['legend.fontsize']  = 7
    plt.rcParams['axes.edgecolor']   = '#666666'
    plt.rcParams['axes.linewidth']   = 0.5
    plt.rcParams['xtick.color']      = '#333333'
    plt.rcParams['ytick.color']      = '#333333'
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['grid.color']       = '#E0E0E0'
    plt.rcParams['grid.linewidth']   = 0.5
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor']   = 'white'
    plt.rcParams['savefig.dpi']      = 300
    plt.rcParams['savefig.facecolor'] = 'white'
    # NB: savefig.bbox 미설정 — 정확 figsize 보존
    return family_kr

FAMILY = _setup_paper_style()
print(f'폰트: {FAMILY} (한글) + Times New Roman (영문)')

# 모형별 색·선스타일·마커 (명세 §그림 7.1)
MODEL_STYLE = {
    'Logistic Regression': dict(color='#1F4E79', ls='-',  short='LR'),       # 실선
    'Random Forest':       dict(color='#ED7D31', ls='--', short='RF'),       # 파선
    'XGBoost':             dict(color='#70AD47', ls=':',  short='XGBoost'),  # 점선
    'LightGBM':            dict(color='#C00000', ls='-.', short='LightGBM'), # 일점쇄선
}

# 카테고리 색 (그림 7.2)
COLOR_CONTROL = '#A6A6A6'   # 통제
COLOR_MOD     = '#ED7D31'   # 조절
COLOR_INV_A   = '#1F4E79'   # 독립 A 내부 안전관리
COLOR_INV_B   = '#70AD47'   # 독립 B 현장 안전행동

# 표 7.7 게재 AUC (논문)
PAPER_AUC = {
    'Logistic Regression': 0.703,
    'Random Forest':       0.705,
    'XGBoost':             0.694,
    'LightGBM':            0.713,
}


# ============ 경로 ============
BASE = Path(__file__).resolve().parent.parent
CACHE = BASE / 'outputs' / '_intermediate'
OUT = BASE / 'results'
OUT.mkdir(parents=True, exist_ok=True)

prep = joblib.load(CACHE / 'preprocessing_cache.joblib')
ml = joblib.load(CACHE / 'ml_cache.joblib')
shap_c = joblib.load(CACHE / 'shap_cache.joblib')

X_test = prep['X_test']
y_test = prep['y_test']
FEATURE_COLS = prep['FEATURE_COLS']
VARS_A = prep['VARS_A']
VARS_B = prep['VARS_B']
VARS_MOD = prep['VARS_MOD']
DUMMY_COLS = prep['DUMMY_COLS']
VARS_CONTROL_NUM = prep['VARS_CONTROL_NUM']

gs_objects = ml['gs_objects']
results = ml['results']
shap_values = shap_c['shap_values']
X_for_plot = shap_c['X_for_plot']


def report_dims(path):
    from PIL import Image
    im = Image.open(path)
    w_in, h_in = im.size[0]/300, im.size[1]/300
    print(f'  {path.name:<45s} {im.size[0]:>5d}×{im.size[1]:>5d}px @300dpi '
          f'= {w_in*25.4:.2f}×{h_in*25.4:.2f} mm')


# ============ 그림 7.1 — ROC (79.17 × 64.71 mm) ============
W = 79.17 / 25.4; H = 64.71 / 25.4
fig = plt.figure(figsize=(W, H))
gs = GridSpec(1, 1, left=0.18, right=0.97, top=0.96, bottom=0.16, figure=fig)
ax = fig.add_subplot(gs[0])

print('\n=== AUC 비교 (논문 표 7.7 vs 실제 재계산) ===')
print(f'  {"모형":<22s}  {"표 7.7":>8s}  {"실제":>8s}  {"차이":>8s}')
auc_actual = {}
for name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']:
    proba = gs_objects[name].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_actual[name] = results[name]['AUC']
    diff = auc_actual[name] - PAPER_AUC[name]
    flag = '✓' if abs(diff) <= 0.005 else '⚠'
    print(f'  {name:<22s}  {PAPER_AUC[name]:>8.3f}  {auc_actual[name]:>8.3f}  {diff:>+8.3f}  {flag}')
    style = MODEL_STYLE[name]
    ax.plot(fpr, tpr,
            color=style['color'], linestyle=style['ls'], linewidth=1.2,
            label=f"{style['short']} (AUC={auc_actual[name]:.3f})")

ax.plot([0, 1], [0, 1], color='#888888', linestyle=':', linewidth=0.7, label='Chance')
ax.set_xlabel('False Positive Rate', fontsize=8)
ax.set_ylabel('True Positive Rate', fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc='lower right', frameon=True, framealpha=0.95,
          edgecolor='#666666', fontsize=6.5,
          handlelength=1.8, handletextpad=0.4, borderpad=0.3)
ax.grid(alpha=1.0, color='#E0E0E0', linewidth=0.4)
fig.set_size_inches(W, H, forward=True)
fig.savefig(OUT / '11_roc_curves.png', dpi=300, bbox_inches=None, pad_inches=0)
plt.close()
report_dims(OUT / '11_roc_curves.png')


# ============ 그림 7.2 — SHAP Bar 전체 27변수 (125.85 × 78.46 mm) ============
# 파랑톤 2단 분리: 주요변수(독립A·B·조절 11) 진한 파랑, 통제(16) 옅은 파랑
COLOR_FOCUS    = '#008BFB'   # legacy fig6_정리정돈 파랑 (SHAP 표준 cmap)
COLOR_CTRL_LT  = '#B3DCFE'   # 동일 톤 옅은 변형 (주요·통제 명도 2단)

focus_set = set(VARS_A + VARS_B + VARS_MOD)

mean_abs = np.abs(shap_values).mean(axis=0)
bar_df = pd.DataFrame({'변수': FEATURE_COLS, 'mean|SHAP|': mean_abs})
bar_df['is_focus'] = bar_df['변수'].isin(focus_set)
bar_df = bar_df.sort_values('mean|SHAP|', ascending=True).reset_index(drop=True)
bar_colors = [COLOR_FOCUS if f else COLOR_CTRL_LT for f in bar_df['is_focus']]

W = 125.85 / 25.4; H = 78.46 / 25.4
fig = plt.figure(figsize=(W, H))
gs = GridSpec(1, 1, left=0.27, right=0.97, top=0.97, bottom=0.17, figure=fig)
ax = fig.add_subplot(gs[0])
ax.barh(np.arange(len(bar_df)), bar_df['mean|SHAP|'],
        color=bar_colors, edgecolor='none', height=0.72)

xmax = bar_df['mean|SHAP|'].max()
for i, val in enumerate(bar_df['mean|SHAP|']):
    ax.text(val + xmax * 0.008, i, f'{val:.4f}',
            va='center', fontsize=6, color='#222222')

ax.set_yticks(np.arange(len(bar_df)))
ax.set_yticklabels(bar_df['변수'], fontsize=6.5)
ax.set_xlabel('mean(|SHAP value|)', fontsize=8)
ax.set_xlim(0, xmax * 1.18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=1.0, color='#E0E0E0', linewidth=0.4)

handles = [
    Patch(facecolor=COLOR_FOCUS,   edgecolor='none', label='주요변수 (독립·조절)'),
    Patch(facecolor=COLOR_CTRL_LT, edgecolor='none', label='통제변수'),
]
ax.legend(handles=handles, loc='lower right', frameon=True, framealpha=0.95,
          edgecolor='#666666', fontsize=6.5,
          handlelength=1.4, handletextpad=0.4, borderpad=0.3)

fig.set_size_inches(W, H, forward=True)
fig.savefig(OUT / '13_shap_bar.png', dpi=300, bbox_inches=None, pad_inches=0)
plt.close()
report_dims(OUT / '13_shap_bar.png')


# ============ 그림 7.4 — Dependence 정리정돈 (92.83 × 50.52 mm) ============
key = '정리정돈상태'
key_idx = FEATURE_COLS.index(key)
xv = X_for_plot[key].values
yv = shap_values[:, key_idx]

W = 92.83 / 25.4; H = 50.52 / 25.4
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.12, 0.26, 0.85, 0.69])

# SHAP 표준 cmap 양 끝 색 (legacy fig6_정리정돈_Dependence 톤 일치)
rng = np.random.RandomState(42)
xj = xv + rng.uniform(-0.10, 0.10, size=len(xv))
colors = np.where(yv > 0, '#FF0052', '#008BFB')   # 양수 빨강, 음수 파랑
ax.scatter(xj, yv, s=8, alpha=0.6, c=colors, edgecolors='none')

ax.axhline(0, color='#888888', linestyle='--', linewidth=0.6)
ax.axvline(3, color='#888888', linestyle='--', linewidth=0.6)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlim(0.5, 5.5)
ax.set_xlabel('정리정돈상태', fontsize=7)
ax.set_ylabel('SHAP value', fontsize=7)
ax.tick_params(axis='both', labelsize=6)
ax.grid(alpha=1.0, color='#E0E0E0', linewidth=0.4)
fig.set_size_inches(W, H, forward=True)
fig.savefig(OUT / '12_shap_dependence_housekeeping.png', dpi=300, bbox_inches=None, pad_inches=0)
plt.close()
report_dims(OUT / '12_shap_dependence_housekeeping.png')


# ============ 그림 7.5 — Interaction Heatmap (93.03 × 63.97 mm) ============
mat = shap_c.get('interaction_mat')
rows = shap_c.get('interaction_rows')
cols = shap_c.get('interaction_cols')
if mat is None:
    raise RuntimeError('interaction_mat not in shap_cache. 05_shap_analysis 재실행 필요.')

W = 93.03 / 25.4; H = 63.97 / 25.4
fig = plt.figure(figsize=(W, H))
gs = GridSpec(1, 2, width_ratios=[24, 1], wspace=0.05,
              left=0.30, right=0.85, top=0.94, bottom=0.22, figure=fig)
ax = fig.add_subplot(gs[0])
cax = fig.add_subplot(gs[1])

im = ax.imshow(mat, cmap='YlOrRd', aspect='auto')
for i in range(len(rows)):
    for j in range(len(cols)):
        v = mat[i, j]
        text_color = 'white' if v > mat.max() * 0.55 else 'black'
        ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                fontsize=5.5, color=text_color)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=6)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7)
ax.tick_params(axis='both', length=0)
ax.set_xlabel('조절변수', fontsize=7)
ax.set_ylabel('독립변수', fontsize=7)
ax.grid(False)

cbar = fig.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=6)
cbar.outline.set_edgecolor('#666666')
cbar.outline.set_linewidth(0.5)

fig.set_size_inches(W, H, forward=True)
fig.savefig(OUT / '14_shap_interaction_heatmap.png', dpi=300, bbox_inches=None, pad_inches=0)
plt.close()
report_dims(OUT / '14_shap_interaction_heatmap.png')

print(f'\n4개 그림 저장 완료 → {OUT}')
