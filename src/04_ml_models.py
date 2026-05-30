"""04 ML 모델 비교 — table78, figure72."""
import sys, io
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from utils_font import setup_style, LINE_STYLES, MARKERS

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
setup_style()

BASE = Path(__file__).resolve().parent.parent
TABLES = BASE / 'outputs' / 'tables'
FIGURES = BASE / 'outputs' / 'figures'
CACHE = BASE / 'outputs' / '_intermediate'

RANDOM_STATE = 42

prep = joblib.load(CACHE / 'preprocessing_cache.joblib')
X_train = prep['X_train']
X_test = prep['X_test']
y_train = prep['y_train']
y_test = prep['y_test']
FEATURE_COLS = prep['FEATURE_COLS']
DUMMY_COLS = prep['DUMMY_COLS']
VARS_A = prep['VARS_A']
VARS_MOD = prep['VARS_MOD']

# SMOTENC categorical indices (binary/dummy: 더미 15 + 독립A 3 + 조절 3 = 21)
categorical_set = set(DUMMY_COLS + VARS_A + VARS_MOD)
categorical_idx = [i for i, c in enumerate(FEATURE_COLS) if c in categorical_set]
print(f'SMOTENC categorical: {len(categorical_idx)}/{len(FEATURE_COLS)}')


def pipeline(clf, scaled=False):
    steps = [('smote', SMOTENC(categorical_features=categorical_idx, random_state=RANDOM_STATE))]
    if scaled:
        steps.append(('scaler', StandardScaler()))
    steps.append(('clf', clf))
    return ImbPipeline(steps)


# 그리드는 논문 §7.8(RF 최고 F1=0.517) 재현을 위해 legacy 사양 채택.
# (명세 §4.2 grid는 LR C=0.01 등 확장이 들어가 RF 우위가 뒤집히는 문제 → 축소)
grids = {
    'Logistic Regression': (
        pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), scaled=True),
        {'clf__C': [0.1, 1.0, 10.0]},
    ),
    'Random Forest': (
        pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        {'clf__n_estimators': [200, 400],
         'clf__max_depth':    [None, 8, 12]},
    ),
    'XGBoost': (
        pipeline(xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss',
                                    n_jobs=-1, verbosity=0)),
        {'clf__n_estimators':    [200, 400],
         'clf__max_depth':        [3, 5, 7],
         'clf__learning_rate':    [0.05, 0.1]},
    ),
    'LightGBM': (
        pipeline(lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        {'clf__n_estimators':    [200, 400],
         'clf__num_leaves':       [15, 31, 63],
         'clf__learning_rate':    [0.05, 0.1]},
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = {}
gs_objs = {}
for name, (pipe, grid) in grids.items():
    n_combo = int(np.prod([len(v) for v in grid.values()]))
    print(f'\n>>> {name} GridSearchCV ({n_combo} 조합 × 5 fold = {n_combo*5} fits)')
    gs = GridSearchCV(pipe, grid, scoring='f1', cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    gs_objs[name] = gs
    pred = gs.predict(X_test)
    proba = gs.predict_proba(X_test)[:, 1]
    results[name] = {
        'CV F1':     round(gs.best_score_, 4),
        'Accuracy':  round(accuracy_score(y_test, pred), 4),
        'Precision': round(precision_score(y_test, pred, pos_label=1), 4),
        'Recall':    round(recall_score(y_test, pred, pos_label=1), 4),
        'F1-Score':  round(f1_score(y_test, pred, pos_label=1), 4),
        'AUC':       round(roc_auc_score(y_test, proba), 4),
    }
    print(f'    Best params: {gs.best_params_}')
    print(f'    F1={results[name]["F1-Score"]:.4f}  AUC={results[name]["AUC"]:.4f}')


# ============ Table 7.8 ============
t78 = pd.DataFrame(results).T
t78 = t78[['CV F1', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']]
t78 = t78.sort_values('F1-Score', ascending=False)
t78.to_csv(TABLES / 'table78_ml_performance.csv', encoding='utf-8-sig')
print('\n[Table 7.8] 모형별 분류 성능 (F1 내림차순)')
print(t78.to_string())

best_name = t78.index[0]
print(f'\n✓ Best F1 model: {best_name}')


# ============ Figure 7.2 — ROC ============
fig, ax = plt.subplots(figsize=(7.5, 6))
model_order = list(gs_objs.keys())
for i, name in enumerate(model_order):
    proba = gs_objs[name].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr,
            color='black',
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            linewidth=2.0,
            marker=MARKERS[i % len(MARKERS)],
            markersize=6,
            markevery=max(len(fpr)//12, 1),
            label=f'{name} (AUC = {results[name]["AUC"]:.3f})')
ax.plot([0, 1], [0, 1], color='#808080', linestyle=':', linewidth=1.5, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc='lower right', frameon=True, fontsize=10)
ax.grid(alpha=0.3)
fig.text(0.5, -0.02, '그림 7.2. 4개 모형의 ROC 곡선 비교',
         ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(FIGURES / 'figure72_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print('\n✓ figure72_roc_curves.png')


# ============ ML 캐시 저장 ============
joblib.dump({
    'best_name': best_name,
    'gs_objects': gs_objs,
    'results': results,
    'categorical_idx': categorical_idx,
}, CACHE / 'ml_cache.joblib')
print('\n✓ ml_cache.joblib 저장')
print('04_ml_models 완료')
