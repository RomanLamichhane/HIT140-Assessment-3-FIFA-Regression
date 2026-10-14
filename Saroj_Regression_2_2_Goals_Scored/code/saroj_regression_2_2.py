from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Saroj_Regression_2_2_Goals_Scored" / "data" / "regression_2_2.csv"
OUT = ROOT / "Saroj_Regression_2_2_Goals_Scored" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

PREDICTORS = [
    "pre_avg_goals_for",
    "pre_goal_difference_per_game",
    "pre_recent3_avg_goals_for",
    "pre_recent3_points_per_game",
    "opponent_pre_avg_goals_against",
    "opponent_pre_goal_difference_per_game",
    "opponent_pre_recent3_avg_goals_against",
    "opponent_pre_recent3_goal_difference",
]
TARGET = "goals_scored"

df = pd.read_csv(DATA)
if len(df) != 208:
    raise RuntimeError(f"Expected 208 rows, got {len(df)}")
if len(PREDICTORS) != 8:
    raise RuntimeError("LR2.2 must use exactly 8 predictors")

# Split by match ID so both team rows from the same match stay together.
match_ids = df["match_id"].drop_duplicates().tolist()
split = int(len(match_ids) * 0.80)
train_ids = set(match_ids[:split])
test_ids = set(match_ids[split:])
train = df[df["match_id"].isin(train_ids)].copy()
test = df[df["match_id"].isin(test_ids)].copy()

model = LinearRegression()
model.fit(train[PREDICTORS], train[TARGET])
pred = model.predict(test[PREDICTORS])

r2 = r2_score(test[TARGET], pred)
mae = mean_absolute_error(test[TARGET], pred)
rmse = np.sqrt(mean_squared_error(test[TARGET], pred))

pd.DataFrame({"predictor": PREDICTORS, "coefficient": model.coef_}).to_csv(
    OUT / "coefficients.csv", index=False
)
pd.DataFrame({
    "match_id": test["match_id"],
    "team": test["team"],
    "opponent": test["opponent"],
    "actual_goals": test[TARGET],
    "predicted_goals": pred,
}).to_csv(OUT / "predictions.csv", index=False)

with open(OUT / "results.txt", "w", encoding="utf-8") as f:
    f.write(f"Rows: {len(df)}\nTrain rows: {len(train)}\nTest rows: {len(test)}\n")
    f.write(f"R2: {r2:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\n")

plt.figure(figsize=(6, 5))
plt.scatter(test[TARGET], pred)
plt.xlabel("Actual goals scored")
plt.ylabel("Predicted goals scored")
plt.title("LR2.2 Actual vs Predicted")
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=160)
plt.close()

residuals = test[TARGET] - pred
plt.figure(figsize=(6, 5))
plt.scatter(pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted goals")
plt.ylabel("Residual")
plt.title("LR2.2 Residual Plot")
plt.tight_layout()
plt.savefig(OUT / "residuals.png", dpi=160)
plt.close()

print(f"LR2.2 complete. R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
