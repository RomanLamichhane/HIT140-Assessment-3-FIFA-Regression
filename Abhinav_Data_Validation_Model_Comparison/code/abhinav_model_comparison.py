from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Abhinav_Data_Validation_Model_Comparison" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

files = {
    "LR2.1 Goal Difference": ROOT / "Raman_Regression_2_1_Goal_Difference" / "outputs" / "results.txt",
    "LR2.2 Goals Scored": ROOT / "Saroj_Regression_2_2_Goals_Scored" / "outputs" / "results.txt",
}
rows = []
for name, path in files.items():
    text = path.read_text(encoding="utf-8")
    vals = {}
    for metric in ["R2", "MAE", "RMSE"]:
        m = re.search(rf"^{metric}:\s*([-0-9.]+)", text, flags=re.M)
        vals[metric] = float(m.group(1)) if m else None
    rows.append({"model": name, **vals})

pd.DataFrame(rows).to_csv(OUT / "model_comparison.csv", index=False)
with open(OUT / "model_comparison.txt", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(f"{r['model']}: R2={r['R2']:.4f}, MAE={r['MAE']:.4f}, RMSE={r['RMSE']:.4f}\n")
    f.write("\nNote: MAE and RMSE are measured on different targets, so compare them within each model rather than directly between models.\n")
print("Model comparison written.")
