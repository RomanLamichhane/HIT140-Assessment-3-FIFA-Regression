from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Raman_Regression_2_1_Goal_Difference" / "data" / "regression_2_1.csv"
OUT = ROOT / "Raman_Regression_2_1_Goal_Difference" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

PREDICTORS = [
    "pre_avg_goals_for_diff",
    "pre_avg_goals_against_diff",
    "pre_goal_difference_per_game_diff",
    "pre_recent3_goal_difference_diff",
    "pre_recent3_points_per_game_diff",
    "pre_games_played_diff",
    "host_advantage",
    "is_knockout",
]
TARGET = "goal_difference"

df = pd.read_csv(DATA)
if len(df) != 104:
    raise RuntimeError(f"Expected 104 rows, got {len(df)}")
if len(PREDICTORS) != 8:
    raise RuntimeError("LR2.1 must use exactly 8 predictors")

# Chronological 80/20 hold-out split.
split = int(len(df) * 0.80)
train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

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
    "home_team": test["home_team"],
    "away_team": test["away_team"],
    "actual_goal_difference": test[TARGET],
    "predicted_goal_difference": pred,
}).to_csv(OUT / "predictions.csv", index=False)

with open(OUT / "results.txt", "w", encoding="utf-8") as f:
    f.write(f"Rows: {len(df)}\nTrain rows: {len(train)}\nTest rows: {len(test)}\n")
    f.write(f"R2: {r2:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\n")

plt.figure(figsize=(6, 5))
plt.scatter(test[TARGET], pred)
plt.xlabel("Actual goal difference")
plt.ylabel("Predicted goal difference")
plt.title("LR2.1 Actual vs Predicted")
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=160)
plt.close()

residuals = test[TARGET] - pred
plt.figure(figsize=(6, 5))
plt.scatter(pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted goal difference")
plt.ylabel("Residual")
plt.title("LR2.1 Residual Plot")
plt.tight_layout()
plt.savefig(OUT / "residuals.png", dpi=160)
plt.close()

print(f"LR2.1 complete. R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
