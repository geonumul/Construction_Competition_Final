"""한글 폰트 강제 적용 + 흑백 인쇄 호환 matplotlib 스타일.

모든 그림 생성 스크립트 상단에서:
    from utils_font import setup_style
    setup_style()
"""
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os


# 흑백 인쇄용 회색조 팔레트 + 선 스타일 + 마커
GRAY_PALETTE = ['#000000', '#404040', '#808080', '#B0B0B0', '#FFFFFF']
LINE_STYLES  = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]  # solid/dashed/dotted/dashdot/dash-dot-dot
MARKERS      = ['o', 's', '^', 'D', 'x', 'v', 'P']
HATCHES      = ['', '///', '\\\\\\', 'xxx', '...', '+++', '|||']


def _find_korean_font():
    system = platform.system()
    if system == 'Darwin':
        candidates = [
            ('AppleGothic',        '/Library/Fonts/AppleGothic.ttf'),
            ('Apple SD Gothic Neo','/System/Library/Fonts/AppleSDGothicNeo.ttc'),
            ('NanumGothic',        '/Library/Fonts/NanumGothic.ttf'),
        ]
    elif system == 'Windows':
        candidates = [
            ('Malgun Gothic',      r'C:\Windows\Fonts\malgun.ttf'),
            ('NanumGothic',        r'C:\Windows\Fonts\NanumGothic.ttf'),
        ]
    else:
        candidates = [
            ('NanumGothic',        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'),
            ('Noto Sans CJK KR',   '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        ]
    for name, path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            actual = fm.FontProperties(fname=path).get_name()
            return actual
    return None


def setup_style():
    """한글 폰트 + 흑백 호환 rcParams 적용."""
    font = _find_korean_font()
    if font is None:
        raise RuntimeError(
            '한글 폰트 미발견. macOS: AppleGothic / Windows: Malgun Gothic / '
            'Linux: NanumGothic 설치 필요.'
        )
    plt.rcParams['font.family']      = font
    plt.rcParams['font.sans-serif']  = [font, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size']        = 11
    plt.rcParams['figure.dpi']       = 300
    plt.rcParams['savefig.dpi']      = 300
    plt.rcParams['savefig.bbox']     = 'tight'

    # 흑백 인쇄 호환
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', GRAY_PALETTE[:4])
    plt.rcParams['axes.spines.top']   = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.edgecolor']    = 'black'
    plt.rcParams['axes.linewidth']    = 1.0
    plt.rcParams['grid.color']        = '#808080'
    plt.rcParams['grid.alpha']        = 0.3
    plt.rcParams['grid.linewidth']    = 0.5
    plt.rcParams['lines.linewidth']   = 1.5
    plt.rcParams['lines.markersize']  = 7
    plt.rcParams['figure.facecolor']  = 'white'
    plt.rcParams['axes.facecolor']    = 'white'
    return font


def reapply_font_to_current_axes():
    """SHAP 등 라이브러리 plot 호출 후 폰트 재적용."""
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(plt.rcParams['font.family'])
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontfamily=plt.rcParams['font.family'])
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontfamily=plt.rcParams['font.family'])
    if ax.get_title():
        ax.set_title(ax.get_title(), fontfamily=plt.rcParams['font.family'])
