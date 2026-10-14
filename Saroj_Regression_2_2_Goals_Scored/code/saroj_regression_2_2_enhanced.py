from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'Saroj_Regression_2_2_Goals_Scored' / 'data' / 'regression_2_2.csv'
OUT = ROOT / 'Saroj_Regression_2_2_Goals_Scored' / 'outputs' / 'enhanced'
OUT.mkdir(parents=True, exist_ok=True)

PREDICTORS = [
    'pre_avg_goals_for',
    'pre_goal_difference_per_game',
    'pre_recent3_avg_goals_for',
    'pre_recent3_points_per_game',
    'opponent_pre_avg_goals_against',
    'opponent_pre_goal_difference_per_game',
    'opponent_pre_recent3_avg_goals_against',
    'opponent_pre_recent3_goal_difference',
]
TARGET = 'goals_scored'


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))




def bootstrap_metric_cis(y_true, y_pred, n_boot=4000, seed=140):
    """Non-parametric 95% bootstrap CIs for MAE, RMSE and mean residual."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    maes, rmses, mean_resids = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        maes.append(np.mean(np.abs(yt - yp)))
        rmses.append(np.sqrt(np.mean((yt - yp) ** 2)))
        mean_resids.append(np.mean(yt - yp))
    def ci(a):
        lo, hi = np.percentile(a, [2.5, 97.5])
        return [float(lo), float(hi)]
    return {
        'mae_95_ci': ci(maes),
        'rmse_95_ci': ci(rmses),
        'mean_residual_95_ci': ci(mean_resids),
    }


def model_candidates():
    candidates = [('OLS', LinearRegression(), {})]
    for a in [0.01, 0.1, 1.0, 10.0, 100.0]:
        candidates.append((f'Ridge(alpha={a})', Pipeline([
            ('scale', StandardScaler()), ('model', Ridge(alpha=a))
        ]), {'algorithm': 'Ridge', 'alpha': a}))
    for a in [0.001, 0.01, 0.1, 0.5, 1.0]:
        candidates.append((f'Lasso(alpha={a})', Pipeline([
            ('scale', StandardScaler()), ('model', Lasso(alpha=a, max_iter=20000))
        ]), {'algorithm': 'Lasso', 'alpha': a}))
    for a in [0.001, 0.01, 0.1, 0.5, 1.0]:
        for ratio in [0.2, 0.5, 0.8]:
            candidates.append((f'ElasticNet(alpha={a},l1_ratio={ratio})', Pipeline([
                ('scale', StandardScaler()),
                ('model', ElasticNet(alpha=a, l1_ratio=ratio, max_iter=20000))
            ]), {'algorithm': 'ElasticNet', 'alpha': a, 'l1_ratio': ratio}))
    return candidates


def rows_for_ids(df, ids):
    return df[df['match_id'].isin(set(ids))].copy()


def main():
    df = pd.read_csv(DATA)
    if len(df) != 208:
        raise RuntimeError(f'Expected 208 rows, got {len(df)}')
    if len(PREDICTORS) != 8:
        raise RuntimeError('LR2.2 must use exactly 8 predictors')
    if df[PREDICTORS].isna().any().any():
        raise RuntimeError('Missing predictor values found')
    if not (df.groupby('match_id').size() == 2).all():
        raise RuntimeError('Every match must have exactly 2 team rows')

    # ---------- EDA ----------
    df[PREDICTORS + [TARGET]].describe().T.to_csv(OUT / 'descriptive_statistics.csv')
    correlations = df[PREDICTORS + [TARGET]].corr(numeric_only=True)
    correlations.to_csv(OUT / 'correlation_matrix.csv')
    target_corr = correlations[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
    target_corr.rename('correlation_with_goals_scored').to_csv(OUT / 'predictor_target_correlations.csv')

    plt.figure(figsize=(6, 4))
    plt.hist(df[TARGET], bins=np.arange(df[TARGET].min() - 0.5, df[TARGET].max() + 1.5, 1), edgecolor='black')
    plt.xlabel('Goals scored')
    plt.ylabel('Team-match observations')
    plt.title('LR2.2 Target Distribution')
    plt.tight_layout()
    plt.savefig(OUT / 'eda_target_distribution.png', dpi=180)
    plt.close()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(correlations.values, aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlations.columns)))
    ax.set_xticklabels(correlations.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(correlations.index)))
    ax.set_yticklabels(correlations.index, fontsize=7)
    fig.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('LR2.2 Correlation Matrix')
    fig.tight_layout()
    fig.savefig(OUT / 'eda_correlation_matrix.png', dpi=180)
    plt.close(fig)

    plt.figure(figsize=(7, 4.5))
    target_corr.sort_values().plot(kind='barh')
    plt.xlabel('Pearson correlation with target')
    plt.ylabel('Predictor')
    plt.title('Goals scored: Predictor-Target Correlations')
    plt.tight_layout()
    plt.savefig(OUT / 'eda_predictor_target_correlations.png', dpi=180)
    plt.close()

    # Final chronological holdout split by match ID keeps the two team rows together.
    match_ids = df['match_id'].drop_duplicates().tolist()
    split = int(len(match_ids) * 0.80)
    train_ids, test_ids = match_ids[:split], match_ids[split:]
    train, test = rows_for_ids(df, train_ids), rows_for_ids(df, test_ids)
    X_train, y_train = train[PREDICTORS], train[TARGET]
    X_test, y_test = test[PREDICTORS], test[TARGET]

    # ---------- Match-grouped time-series CV ----------
    tscv = TimeSeriesSplit(n_splits=5)
    candidates = model_candidates()
    cv_rows = []
    for name, model, params in candidates:
        fold_results = []
        for fold, (tr_i, va_i) in enumerate(tscv.split(np.array(train_ids)), start=1):
            fold_train_ids = [train_ids[i] for i in tr_i]
            fold_val_ids = [train_ids[i] for i in va_i]
            tr = rows_for_ids(df, fold_train_ids)
            va = rows_for_ids(df, fold_val_ids)
            model.fit(tr[PREDICTORS], tr[TARGET])
            p = model.predict(va[PREDICTORS])
            fold_results.append({
                'fold': fold,
                'r2': r2_score(va[TARGET], p),
                'mae': mean_absolute_error(va[TARGET], p),
                'rmse': rmse(va[TARGET], p),
            })
        cv_rows.append({
            'candidate': name,
            **params,
            'cv_mean_r2': np.mean([x['r2'] for x in fold_results]),
            'cv_mean_mae': np.mean([x['mae'] for x in fold_results]),
            'cv_mean_rmse': np.mean([x['rmse'] for x in fold_results]),
            'cv_sd_rmse': np.std([x['rmse'] for x in fold_results], ddof=1),
        })

    cv_table = pd.DataFrame(cv_rows).sort_values(['cv_mean_rmse', 'cv_mean_mae']).reset_index(drop=True)
    cv_table.to_csv(OUT / 'cv_model_comparison.csv', index=False)
    top_cv = cv_table.head(10).sort_values('cv_mean_rmse', ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(top_cv['candidate'], top_cv['cv_mean_rmse'], xerr=top_cv['cv_sd_rmse'])
    plt.xlabel('Mean cross-validated RMSE')
    plt.ylabel('Candidate model')
    plt.title('Top Cross-Validated Model Candidates')
    plt.tight_layout()
    plt.savefig(OUT / 'cv_top_models_rmse.png', dpi=180)
    plt.close()
    best_name = cv_table.iloc[0]['candidate']
    best_model = next(m for n, m, _ in candidates if n == best_name)

    baseline = LinearRegression().fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = {
        'model': 'OLS baseline',
        'r2': r2_score(y_test, baseline_pred),
        'mae': mean_absolute_error(y_test, baseline_pred),
        'rmse': rmse(y_test, baseline_pred),
    }
    pd.DataFrame({'predictor': PREDICTORS, 'coefficient': baseline.coef_}).to_csv(
        OUT / 'baseline_coefficients.csv', index=False
    )

    best_model.fit(X_train, y_train)
    best_pred = best_model.predict(X_test)
    best_metrics = {
        'model': str(best_name),
        'r2': r2_score(y_test, best_pred),
        'mae': mean_absolute_error(y_test, best_pred),
        'rmse': rmse(y_test, best_pred),
    }
    bootstrap_cis = bootstrap_metric_cis(y_test, best_pred)
    pd.DataFrame({
        'metric': ['MAE', 'RMSE', 'Mean residual'],
        'lower_95': [bootstrap_cis['mae_95_ci'][0], bootstrap_cis['rmse_95_ci'][0], bootstrap_cis['mean_residual_95_ci'][0]],
        'upper_95': [bootstrap_cis['mae_95_ci'][1], bootstrap_cis['rmse_95_ci'][1], bootstrap_cis['mean_residual_95_ci'][1]],
    }).to_csv(OUT / 'selected_holdout_bootstrap_95_ci.csv', index=False)
    pd.DataFrame([baseline_metrics, best_metrics]).to_csv(OUT / 'holdout_model_comparison.csv', index=False)
    pd.DataFrame({
        'match_id': test['match_id'],
        'team': test['team'],
        'opponent': test['opponent'],
        'actual_goals': y_test,
        'baseline_prediction': baseline_pred,
        'selected_model_prediction': best_pred,
    }).to_csv(OUT / 'holdout_predictions.csv', index=False)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, best_pred)
    lo = min(y_test.min(), best_pred.min())
    hi = max(y_test.max(), best_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle='--')
    plt.xlabel('Actual goals scored')
    plt.ylabel('Predicted goals scored')
    plt.title(f'LR2.2 Selected Model: {best_name}')
    plt.tight_layout()
    plt.savefig(OUT / 'selected_actual_vs_predicted.png', dpi=180)
    plt.close()

    residuals = y_test.to_numpy() - best_pred
    plt.figure(figsize=(6, 5))
    plt.scatter(best_pred, residuals)
    plt.axhline(0)
    plt.xlabel('Predicted goals')
    plt.ylabel('Residual')
    plt.title('LR2.2 Selected Model Residuals')
    plt.tight_layout()
    plt.savefig(OUT / 'selected_residuals.png', dpi=180)
    plt.close()

    improvement_rmse = baseline_metrics['rmse'] - best_metrics['rmse']
    improvement_mae = baseline_metrics['mae'] - best_metrics['mae']
    summary = {
        'rows': len(df),
        'train_rows': len(train),
        'test_rows': len(test),
        'train_match_ids': len(train_ids),
        'test_match_ids': len(test_ids),
        'predictor_count': len(PREDICTORS),
        'cv_folds': 5,
        'selected_model': str(best_name),
        'baseline_holdout': baseline_metrics,
        'selected_holdout': best_metrics,
        'selected_holdout_bootstrap_95_ci': bootstrap_cis,
        'rmse_improvement_vs_baseline': improvement_rmse,
        'mae_improvement_vs_baseline': improvement_mae,
    }
    (OUT / 'enhanced_results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with open(OUT / 'enhanced_results.txt', 'w', encoding='utf-8') as f:
        f.write('LR2.2 ENHANCED ANALYSIS\n')
        f.write(f"Rows={len(df)}, train={len(train)}, test={len(test)}, predictors={len(PREDICTORS)}\n")
        f.write(f"Train matches={len(train_ids)}, test matches={len(test_ids)}\n")
        f.write('Validation: 5-fold TimeSeriesSplit over match IDs; both team rows stay together.\n')
        f.write(f"Selected by lowest mean CV RMSE: {best_name}\n")
        f.write(f"Baseline OLS holdout: R2={baseline_metrics['r2']:.4f}, MAE={baseline_metrics['mae']:.4f}, RMSE={baseline_metrics['rmse']:.4f}\n")
        f.write(f"Selected holdout: R2={best_metrics['r2']:.4f}, MAE={best_metrics['mae']:.4f}, RMSE={best_metrics['rmse']:.4f}\n")
        f.write(f"Bootstrap 95% CI - MAE: [{bootstrap_cis['mae_95_ci'][0]:.4f}, {bootstrap_cis['mae_95_ci'][1]:.4f}]\n")
        f.write(f"Bootstrap 95% CI - RMSE: [{bootstrap_cis['rmse_95_ci'][0]:.4f}, {bootstrap_cis['rmse_95_ci'][1]:.4f}]\n")
        f.write(f"Bootstrap 95% CI - mean residual: [{bootstrap_cis['mean_residual_95_ci'][0]:.4f}, {bootstrap_cis['mean_residual_95_ci'][1]:.4f}]\n")
        f.write(f"RMSE improvement (positive is better)={improvement_rmse:.4f}\n")
        f.write(f"MAE improvement (positive is better)={improvement_mae:.4f}\n")

    print((OUT / 'enhanced_results.txt').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
