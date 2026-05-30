"""06 연구 모형 그림 — figure21."""
import sys, io
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from utils_font import setup_style

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
setup_style()

BASE = Path(__file__).resolve().parent.parent
FIGURES = BASE / 'outputs' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)


fig, ax = plt.subplots(figsize=(14, 7.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8.5)
ax.axis('off')


def draw_box(x, y, w, h, title, items, style='solid', lw=1.5, bold_title=True):
    rect = Rectangle((x, y), w, h, facecolor='white',
                      edgecolor='black', linewidth=lw, linestyle=style)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.28, title, ha='center', va='top',
            fontsize=11, fontweight='bold' if bold_title else 'normal')
    items_text = '\n'.join('· ' + s for s in items)
    ax.text(x + 0.2, y + h - 0.85, items_text, ha='left', va='top', fontsize=10)


# 독립변수 A (좌상)
A_BOX = (0.7, 5.2, 3.8, 2.0)
draw_box(*A_BOX, '독립변수 A · 내부 안전관리체계',
          ['안전조직수준', '위원회수준', '인증보유'])

# 독립변수 B (좌하)
B_BOX = (0.7, 1.8, 3.8, 2.4)
draw_box(*B_BOX, '독립변수 B · 현장 안전행동',
          ['위험성평가수준', '교육훈련도움', '정리정돈상태',
           '작업중지권', '작업반장기여'])

# 조절변수 (상단 중앙)
M_BOX = (6.5, 6.0, 3.3, 1.8)
draw_box(*M_BOX, '조절변수 · 외부기관 개입',
          ['전문지도', '고용노동부감독', '안전보건공단지원'],
          style='--')

# 종속변수 (우)
Y_BOX = (10.7, 3.5, 2.9, 2.0)
draw_box(*Y_BOX, '종속변수',
          ['산업재해 발생 여부', '(0 = 미발생, 1 = 발생)'],
          lw=2.0)

# 통제변수 (하단)
C_BOX = (5.0, 0.3, 7.0, 0.9)
rect = Rectangle((C_BOX[0], C_BOX[1]), C_BOX[2], C_BOX[3],
                  facecolor='white', edgecolor='black',
                  linewidth=1.0, linestyle=':')
ax.add_patch(rect)
ax.text(C_BOX[0] + 0.15, C_BOX[1] + C_BOX[3]/2,
        '통제변수:  공사규모 · 발주처 · 기성공정률 · 공사종류 · 외국인비율',
        ha='left', va='center', fontsize=10)


# === 화살표 ===
# H1: A → 종속 (주효과, 굵은 실선)
ax.annotate('', xy=(Y_BOX[0], 4.7), xytext=(A_BOX[0] + A_BOX[2], 6.2),
            arrowprops=dict(arrowstyle='->', linewidth=2.0, color='black',
                            mutation_scale=22))
ax.text(7.5, 5.7, 'H1 (A 주효과)',
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                   edgecolor='black', linewidth=1.0))

# H2: B → 종속 (주효과, 굵은 실선)
ax.annotate('', xy=(Y_BOX[0], 4.0), xytext=(B_BOX[0] + B_BOX[2], 3.0),
            arrowprops=dict(arrowstyle='->', linewidth=2.0, color='black',
                            mutation_scale=22))
ax.text(7.5, 3.2, 'H2 (B 주효과)',
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                   edgecolor='black', linewidth=1.0))

# H3a: 조절 → A→종속 경로 (점선)
ax.annotate('', xy=(8.0, 5.7), xytext=(8.0, 6.0),
            arrowprops=dict(arrowstyle='->', linewidth=1.5, color='black',
                            linestyle='--', mutation_scale=18))
ax.text(8.2, 5.9, 'H3a 조절', ha='left', fontsize=9.5, fontstyle='italic')

# H3b: 조절 → B→종속 경로 (점선)
ax.annotate('', xy=(8.0, 3.0), xytext=(8.0, 6.0),
            arrowprops=dict(arrowstyle='->', linewidth=1.5, color='black',
                            linestyle='--', mutation_scale=18))
ax.text(8.2, 4.5, 'H3b 조절', ha='left', fontsize=9.5, fontstyle='italic')


fig.text(0.5, 0.02, '그림 2.1. 연구 모형',
         ha='center', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(FIGURES / 'figure21_research_model.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ figure21_research_model.png')
print('06_research_model_figure 완료')
