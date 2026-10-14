from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Abhinav_Data_Validation_Model_Comparison' / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)

paths = {
    'LR2.1 Goal Difference': ROOT / 'Raman_Regression_2_1_Goal_Difference' / 'outputs' / 'enhanced' / 'enhanced_results.json',
    'LR2.2 Goals Scored': ROOT / 'Saroj_Regression_2_2_Goals_Scored' / 'outputs' / 'enhanced' / 'enhanced_results.json',
}

rows = []
for task, path in paths.items():
    d = json.loads(path.read_text(encoding='utf-8'))
    b = d['baseline_holdout']
    s = d['selected_holdout']
    rows.append({
        'task': task,
        'selected_model': d['selected_model'],
        'baseline_r2': b['r2'],
        'baseline_mae': b['mae'],
        'baseline_rmse': b['rmse'],
        'selected_r2': s['r2'],
        'selected_mae': s['mae'],
        'selected_rmse': s['rmse'],
        'rmse_improvement': d['rmse_improvement_vs_baseline'],
        'mae_improvement': d['mae_improvement_vs_baseline'],
    })

result = pd.DataFrame(rows)
result.to_csv(OUT / 'enhanced_model_comparison.csv', index=False)
with open(OUT / 'enhanced_model_comparison.txt', 'w', encoding='utf-8') as f:
    for _, r in result.iterrows():
        f.write(f"{r['task']}\n")
        f.write(f"  Selected model: {r['selected_model']}\n")
        f.write(f"  Baseline: R2={r['baseline_r2']:.4f}, MAE={r['baseline_mae']:.4f}, RMSE={r['baseline_rmse']:.4f}\n")
        f.write(f"  Selected: R2={r['selected_r2']:.4f}, MAE={r['selected_mae']:.4f}, RMSE={r['selected_rmse']:.4f}\n")
        f.write(f"  RMSE improvement={r['rmse_improvement']:.4f}; MAE improvement={r['mae_improvement']:.4f}\n\n")
    f.write('Important: LR2.1 and LR2.2 predict different targets, so MAE/RMSE should be interpreted within each task rather than directly compared across tasks.\n')
print(result.to_string(index=False))
