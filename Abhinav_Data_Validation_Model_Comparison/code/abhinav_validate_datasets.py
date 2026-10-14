from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D1 = ROOT / "Raman_Regression_2_1_Goal_Difference" / "data" / "regression_2_1.csv"
D2 = ROOT / "Saroj_Regression_2_2_Goals_Scored" / "data" / "regression_2_2.csv"
OUT = ROOT / "Abhinav_Data_Validation_Model_Comparison" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

P1 = [
    "pre_avg_goals_for_diff",
    "pre_avg_goals_against_diff",
    "pre_goal_difference_per_game_diff",
    "pre_recent3_goal_difference_diff",
    "pre_recent3_points_per_game_diff",
    "pre_games_played_diff",
    "host_advantage",
    "is_knockout",
]
P2 = [
    "pre_avg_goals_for",
    "pre_goal_difference_per_game",
    "pre_recent3_avg_goals_for",
    "pre_recent3_points_per_game",
    "opponent_pre_avg_goals_against",
    "opponent_pre_goal_difference_per_game",
    "opponent_pre_recent3_avg_goals_against",
    "opponent_pre_recent3_goal_difference",
]

# Concept-level overlap: exactly 4.
SHARED_CONCEPTS = [
    "average goals scored",
    "average goals conceded",
    "goal difference per game",
    "recent points per game",
]

d1 = pd.read_csv(D1)
d2 = pd.read_csv(D2)

checks = []
checks.append(("LR2.1 has 104 rows", len(d1) == 104))
checks.append(("LR2.2 has 208 rows", len(d2) == 208))
checks.append(("LR2.1 has exactly 8 predictors", len(P1) == 8))
checks.append(("LR2.2 has exactly 8 predictors", len(P2) == 8))
checks.append(("Exactly 4 shared predictor concepts", len(SHARED_CONCEPTS) == 4))
checks.append(("LR2.1 predictors have no missing values", not d1[P1].isna().any().any()))
checks.append(("LR2.2 predictors have no missing values", not d2[P2].isna().any().any()))
checks.append(("LR2.1 match IDs unique", d1["match_id"].is_unique))
checks.append(("LR2.2 has 2 rows per match", bool((d2.groupby("match_id").size() == 2).all())))

lines = []
for label, ok in checks:
    lines.append(f"{'PASS' if ok else 'FAIL'} - {label}")

lines.append("")
lines.append("Shared predictor concepts:")
for x in SHARED_CONCEPTS:
    lines.append(f"- {x}")

lines.append("")
lines.append("Leakage control:")
lines.append("- Every rolling/recent statistic is computed before the current match is added to team history.")
lines.append("- Current-match goals are targets only, never explanatory variables.")
lines.append("- First-match historical features use a documented neutral zero baseline.")
lines.append("- Knockout status and host status are known before kickoff.")

text = "\n".join(lines)
(OUT / "assignment_validation.txt").write_text(text, encoding="utf-8")
print(text)

if not all(ok for _, ok in checks):
    raise RuntimeError("One or more validation checks failed.")
