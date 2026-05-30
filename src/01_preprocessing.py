"""01 전처리 — raw_2021.csv → processed_2021.csv

§1 표본 선정 (1,502 → 1,375), §2 변수 정의 (종속1·독립A 3·독립B 5·조절3·통제5).
체크포인트마다 카운트 출력. 최종 1,375 아니면 중단.
"""
import sys
import io
from pathlib import Path
import numpy as np
import pandas as pd

# 출력 인코딩
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / 'data' / 'raw_2021.csv'
OUT = BASE / 'data' / 'processed_2021.csv'

RANDOM_STATE = 42


# ============================================================
# §1. 표본 선정 (1,502 → 1,375)
# ============================================================
print('=' * 60)
print('§1. 표본 선정')
print('=' * 60)

df = pd.read_csv(RAW, encoding='cp949')
n0 = len(df)
print(f'  단계 0 — 원자료 ............................... {n0:>5d}')
assert n0 == 1502, f'원자료 행 수 불일치: {n0} ≠ 1502'

# 1단계: 종속변수 무응답 (Q27_3_1~3 모두 NaN) 제거
mask = ~(df['Q27_3_1'].isna() & df['Q27_3_2'].isna() & df['Q27_3_3'].isna())
df = df.loc[mask].copy()
print(f'  단계 1 — 종속변수 무응답 제거 .................. {len(df):>5d}  ({n0 - len(df)}개 제거)')

# 2단계: 종속변수 이상치 (Q27_3_3==30 단일 사례) 제거
prev = len(df)
df = df.loc[df['Q27_3_3'] != 30].copy()
print(f'  단계 2 — 종속변수 이상치 제거 .................. {len(df):>5d}  ({prev - len(df)}개 제거)')

# 3단계: 안전조직 무응답 (Q6==9) 제거
prev = len(df)
df = df.loc[df['Q6'] != 9].copy()
print(f'  단계 3 — 안전조직 무응답 제거 .................. {len(df):>5d}  ({prev - len(df)}개 제거)')

# 4단계: 위원회 무응답·불명 (Q10 ∈ {4,9} OR Q10_1==9) 제거
# 본문 §3.1은 Q10=4·9만 명시하나, '위원회 운영 여부 무응답'은 위원회 항목
# 응답 자체가 불완전한 Q10_1=9도 포함하는 것이 동일 자료 선행연구·legacy의 관행.
prev = len(df)
df = df.loc[~(df['Q10'].isin([4, 9]) | (df['Q10_1'] == 9))].copy()
print(f'  단계 4 — 위원회 무응답·불명 제거 ............... {len(df):>5d}  ({prev - len(df)}개 제거)')

# 5단계: 위험성평가 미응답 (Q14 NaN) 제거
prev = len(df)
df = df.loc[df['Q14'].notna()].copy()
print(f'  단계 5 — 위험성평가 미응답 제거 ................ {len(df):>5d}  ({prev - len(df)}개 제거)')

# 6단계: 전문지도 무응답 (Q9==9) 제거
prev = len(df)
df = df.loc[df['Q9'] != 9].copy()
print(f'  단계 6 — 전문지도 무응답 제거 .................. {len(df):>5d}  ({prev - len(df)}개 제거)')

print()
assert len(df) == 1375, f'❌ 최종 표본 {len(df)} ≠ 1,375 — 중단'
print(f'  ✓ 최종 표본 1,375 확정')

df = df.reset_index(drop=True)


# ============================================================
# §2. 변수 구성
# ============================================================
print()
print('=' * 60)
print('§2. 변수 구성')
print('=' * 60)

out = pd.DataFrame(index=df.index)

# --- 종속변수 ---
y_sum = df['Q27_3_1'].fillna(0) + df['Q27_3_2'].fillna(0) + df['Q27_3_3'].fillna(0)
out['accident'] = (y_sum >= 1).astype(int)
y_dist = out['accident'].value_counts().sort_index().to_dict()
y_pct = (out['accident'].value_counts(normalize=True).sort_index() * 100).round(2).to_dict()
print(f'\naccident: {y_dist} ({y_pct[0]}% / {y_pct[1]}%)')
assert y_dist == {0: 984, 1: 391}, f'❌ 종속변수 분포 불일치: {y_dist}'
print('  ✓ 기대 {0:984, 1:391} 일치')

# --- 독립 A — 내부 안전관리체계 ---
out['안전조직수준'] = ((df['Q6'] == 1) | (df['Q7_1'] == 1)).astype(int)
out['위원회수준']   = ((df['Q10'].isin([1, 2])) & (df['Q10_1'] == 1)).astype(int)  # broad
out['인증보유']     = ((df['Q12_1'] == 1) | (df['Q12_2'] == 1)).astype(int)

# --- 독립 B — 현장 안전행동 ---
def _risk_level(row):
    if row['Q14'] == 1:
        return 0
    if row['Q14'] in [2, 3]:
        if row['Q14_2_2'] == 1:
            return 2
        if row['Q14_2_2'] == 2:
            return 1
    return np.nan
out['위험성평가수준'] = df.apply(_risk_level, axis=1).astype('Int64')
out['교육훈련도움']   = df['Q16_10'].astype(int)
out['정리정돈상태']   = df['Q16_14'].astype(int)
out['작업중지권']     = df['Q16_17'].astype(int)
out['작업반장기여']   = df['Q17_4'].astype(int)

# --- 조절변수 — 외부기관 개입 ---
out['전문지도']         = (df['Q9']  == 1).astype(int)
out['고용노동부감독']    = (df['Q30'] == 1).astype(int)
out['안전보건공단지원']  = (df['Q31'] == 1).astype(int)

# --- 통제변수 — 현장 특성 ---
# 공사규모 (SQ2): 1→소규모, 2→중규모(기준), 3·4→대규모
def _size(v):
    if v == 1: return 1
    if v == 2: return 2
    if v in (3, 4): return 3
    return np.nan
out['공사규모'] = df['SQ2'].apply(_size).astype('Int64')

# 발주처 (Q1_4): 1·2→정부·공공, 3→사기업·개인(기준), 4·5→자체·기타
def _owner(v):
    if v in (1, 2): return 1
    if v == 3:      return 2
    if v in (4, 5): return 3
    return np.nan
out['발주처'] = df['Q1_4'].apply(_owner).astype('Int64')

# 기성공정률 (Q1_5): 1~6 raw
out['기성공정률'] = df['Q1_5'].astype(int)

# 공사종류 (Q2): 12 → 7
def _kind(v):
    if v == 1:           return 1   # 아파트
    if v == 2:           return 2   # 공장
    if v == 3:           return 3   # 근린생활시설/빌딩
    if v in (4, 5):      return 4   # 학교/병원/건축기타
    if v in (6, 7, 8):   return 5   # 도로/철도/교량
    if v in (9, 10):     return 6   # 상하수도/토목
    if v in (11, 12):    return 7   # 전기·통신·기타
    return np.nan
out['공사종류'] = df['Q2'].apply(_kind).astype('Int64')

# 외국인비율 (Q4_3 / Q4_1 * 100); legacy와 동일하게 단순 비율, 99999 미처리
out['외국인비율'] = (df['Q4_3'].astype(float) / df['Q4_1'].astype(float) * 100).round(2)

# 결측 확인
nans = out.isna().sum()
nans = nans[nans > 0]
if len(nans):
    print(f'\n⚠ 결측치 발견:\n{nans}')
    assert False, '전처리 후 결측 있으면 안 됨'
else:
    print('\n✓ 결측치 없음')

# ============================================================
# 분포 출력
# ============================================================
print()
print('=' * 60)
print('변수별 분포 요약')
print('=' * 60)
for c in ['안전조직수준', '위원회수준', '인증보유', '위험성평가수준',
          '전문지도', '고용노동부감독', '안전보건공단지원',
          '공사규모', '발주처', '기성공정률', '공사종류']:
    print(f'  {c:<14s} {out[c].value_counts().sort_index().to_dict()}')

for c in ['교육훈련도움', '정리정돈상태', '작업중지권', '작업반장기여', '외국인비율']:
    print(f'  {c:<14s} mean={out[c].mean():.2f} sd={out[c].std():.2f}')

# ============================================================
# 저장
# ============================================================
out.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f'\n✓ 저장: {OUT}')
print(f'  shape: {out.shape}  (행 {out.shape[0]} × 열 {out.shape[1]})')
