"""02 기술통계 — table71, table72, figure71."""
import sys, io
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from utils_font import setup_style

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data' / 'processed_2021.csv'
TABLES = BASE / 'outputs' / 'tables'
FIGURES = BASE / 'outputs' / 'figures'
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

setup_style()

df = pd.read_csv(DATA)
print(f'Loaded: {df.shape}')

TARGET = 'accident'

CONT_VARS = ['위험성평가수준', '교육훈련도움', '정리정돈상태', '작업중지권', '작업반장기여', '외국인비율']
CAT_VARS  = ['안전조직수준', '위원회수준', '인증보유',
             '전문지도', '고용노동부감독', '안전보건공단지원',
             '공사규모', '발주처', '기성공정률', '공사종류']


def sig(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


# ============ Table 7.1 — 연속/순서형 ============
rows = []
for v in CONT_VARS:
    g0 = df.loc[df[TARGET] == 0, v]
    g1 = df.loc[df[TARGET] == 1, v]
    t, p = st.ttest_ind(g0, g1, equal_var=False)
    rows.append({
        '변수': v,
        '전체 평균(SD)': f'{df[v].mean():.2f} ({df[v].std():.2f})',
        '미발생 평균(SD)': f'{g0.mean():.2f} ({g0.std():.2f})',
        '발생 평균(SD)':   f'{g1.mean():.2f} ({g1.std():.2f})',
        't': round(t, 3),
        'p-value': round(p, 4),
        '유의도': sig(p),
    })
t71 = pd.DataFrame(rows)
t71.to_csv(TABLES / 'table71_descriptive_continuous.csv', index=False, encoding='utf-8-sig')
print('\n[Table 7.1] 연속·순서형')
print(t71.to_string(index=False))


# ============ Table 7.2 — 범주형 ============
rows = []
for v in CAT_VARS:
    ct = pd.crosstab(df[v], df[TARGET])
    chi2, p, dof, _ = st.chi2_contingency(ct)
    rows.append({
        '변수': v,
        '범주 수': int(df[v].nunique()),
        'χ²': round(chi2, 3),
        'df': int(dof),
        'p-value': round(p, 4),
        '유의도': sig(p),
    })
t72 = pd.DataFrame(rows)
t72.to_csv(TABLES / 'table72_descriptive_categorical.csv', index=False, encoding='utf-8-sig')
print('\n[Table 7.2] 범주형')
print(t72.to_string(index=False))


# ============ Figure 7.1 — 종속변수 분포 ============
fig, ax = plt.subplots(figsize=(7, 4.5))
counts = df[TARGET].value_counts().sort_index()
labels = [
    f'미발생\n(n={counts[0]:,}, {counts[0]/len(df):.1%})',
    f'발생\n(n={counts[1]:,}, {counts[1]/len(df):.1%})'
]
bars = ax.bar(labels, counts.values,
              color=['#909090', '#404040'],
              edgecolor='black', linewidth=1.2,
              hatch=['', '////'])
ax.set_ylabel('사업장 수')
ax.set_ylim(0, max(counts.values) * 1.12)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, v + max(counts.values)*0.02,
            f'{v:,}', ha='center', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
fig.text(0.5, -0.02, '그림 7.1. 종속변수 분포',
         ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(FIGURES / 'figure71_y_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print('\n✓ figure71_y_distribution.png')
print('\n02_descriptive 완료')
