from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "Shared_Data" / "raw" / "fbref_matches_raw.csv"
OUT1 = ROOT / "Raman_Regression_2_1_Goal_Difference" / "data" / "regression_2_1.csv"
OUT2 = ROOT / "Saroj_Regression_2_2_Goals_Scored" / "data" / "regression_2_2.csv"
PROC = ROOT / "Shared_Data" / "processed" / "team_match_history.csv"
HOSTS = {"USA", "Mexico", "Canada"}

def parse_score(s: str):
    s = str(s).strip().replace("–", "-").replace("—", "-")
    m = re.search(r"(?<!\()(?<!\d)(\d+)\s*-\s*(\d+)(?!\d)", s)
    if not m:
        raise ValueError(f"Could not parse score: {s}")
    return int(m.group(1)), int(m.group(2))

def pre_stats(history):
    if not history:
        return {
            "pre_avg_goals_for": 0.0,
            "pre_avg_goals_against": 0.0,
            "pre_goal_difference_per_game": 0.0,
            "pre_recent3_avg_goals_for": 0.0,
            "pre_recent3_avg_goals_against": 0.0,
            "pre_recent3_points_per_game": 0.0,
            "pre_recent3_goal_difference": 0.0,
            "pre_failed_to_score_rate": 0.0,
            "pre_win_rate": 0.0,
            "pre_draw_rate": 0.0,
            "pre_points_per_game": 0.0,
            "pre_games_played": 0.0,
            "pre_last_date": pd.NaT,
        }

    h = pd.DataFrame(history)
    n = len(h)
    wins = (h["gf"] > h["ga"]).sum()
    draws = (h["gf"] == h["ga"]).sum()
    points = wins * 3 + draws

    recent = h.tail(3)
    r_wins = (recent["gf"] > recent["ga"]).sum()
    r_draws = (recent["gf"] == recent["ga"]).sum()
    r_points = r_wins * 3 + r_draws

    return {
        "pre_avg_goals_for": h["gf"].mean(),
        "pre_avg_goals_against": h["ga"].mean(),
        "pre_goal_difference_per_game": (h["gf"] - h["ga"]).mean(),
        "pre_recent3_avg_goals_for": recent["gf"].mean(),
        "pre_recent3_avg_goals_against": recent["ga"].mean(),
        "pre_recent3_points_per_game": r_points / len(recent),
        "pre_recent3_goal_difference": (recent["gf"] - recent["ga"]).mean(),
        "pre_failed_to_score_rate": (h["gf"] == 0).mean(),
        "pre_win_rate": wins / n,
        "pre_draw_rate": draws / n,
        "pre_points_per_game": points / n,
        "pre_games_played": float(n),
        "pre_last_date": h["date"].max(),
    }

def main():
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    required = ["Date", "Round", "Home", "Away", "Score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    if len(df) != 104:
        raise RuntimeError(f"Expected 104 matches, found {len(df)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.reset_index(drop=True)
    df["match_id"] = [f"M{i:03d}" for i in range(1, len(df) + 1)]

    scores = df["Score"].map(parse_score)
    df["home_goals"] = [x[0] for x in scores]
    df["away_goals"] = [x[1] for x in scores]

    histories = {}
    lr21_rows = []
    lr22_rows = []
    hist_rows = []

    for _, r in df.iterrows():
        date = r["Date"]
        home, away = r["Home"], r["Away"]
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        round_name = str(r["Round"])

        hs = pre_stats(histories.get(home, []))
        aws = pre_stats(histories.get(away, []))
        host_advantage = (1 if home in HOSTS else 0) - (1 if away in HOSTS else 0)
        is_knockout = 0 if "group" in round_name.lower() else 1

        # LR2.1: exactly 8 predictors.
        lr21_rows.append({
            "match_id": r["match_id"],
            "date": date.date().isoformat(),
            "round": round_name,
            "home_team": home,
            "away_team": away,
            "goal_difference": hg - ag,
            "pre_avg_goals_for_diff": hs["pre_avg_goals_for"] - aws["pre_avg_goals_for"],
            "pre_avg_goals_against_diff": hs["pre_avg_goals_against"] - aws["pre_avg_goals_against"],
            "pre_goal_difference_per_game_diff": hs["pre_goal_difference_per_game"] - aws["pre_goal_difference_per_game"],
            "pre_recent3_goal_difference_diff": hs["pre_recent3_goal_difference"] - aws["pre_recent3_goal_difference"],
            "pre_recent3_points_per_game_diff": hs["pre_recent3_points_per_game"] - aws["pre_recent3_points_per_game"],
            "pre_games_played_diff": hs["pre_games_played"] - aws["pre_games_played"],
            "host_advantage": host_advantage,
            "is_knockout": is_knockout,
        })

        # LR2.2: exactly 8 predictors, two rows per match.
        for team, opp, gf, ga, ts, os_ in [
            (home, away, hg, ag, hs, aws),
            (away, home, ag, hg, aws, hs),
        ]:
            lr22_rows.append({
                "match_id": r["match_id"],
                "date": date.date().isoformat(),
                "round": round_name,
                "team": team,
                "opponent": opp,
                "goals_scored": gf,
                "pre_avg_goals_for": ts["pre_avg_goals_for"],
                "pre_goal_difference_per_game": ts["pre_goal_difference_per_game"],
                "pre_recent3_avg_goals_for": ts["pre_recent3_avg_goals_for"],
                "pre_recent3_points_per_game": ts["pre_recent3_points_per_game"],
                "opponent_pre_avg_goals_against": os_["pre_avg_goals_against"],
                "opponent_pre_goal_difference_per_game": os_["pre_goal_difference_per_game"],
                "opponent_pre_recent3_avg_goals_against": os_["pre_recent3_avg_goals_against"],
                "opponent_pre_recent3_goal_difference": os_["pre_recent3_goal_difference"],
            })
            hist_rows.append({
                "match_id": r["match_id"],
                "date": date.date().isoformat(),
                "team": team,
                "opponent": opp,
                "gf": gf,
                "ga": ga,
            })

        # IMPORTANT: update history only after creating current-match predictors.
        histories.setdefault(home, []).append({"date": date, "gf": hg, "ga": ag})
        histories.setdefault(away, []).append({"date": date, "gf": ag, "ga": hg})

    lr21 = pd.DataFrame(lr21_rows)
    lr22 = pd.DataFrame(lr22_rows)

    if len(lr21) != 104:
        raise RuntimeError(f"LR2.1 must have 104 rows, got {len(lr21)}")
    if len(lr22) != 208:
        raise RuntimeError(f"LR2.2 must have 208 rows, got {len(lr22)}")

    OUT1.parent.mkdir(parents=True, exist_ok=True)
    OUT2.parent.mkdir(parents=True, exist_ok=True)
    PROC.parent.mkdir(parents=True, exist_ok=True)

    lr21.to_csv(OUT1, index=False)
    lr22.to_csv(OUT2, index=False)
    pd.DataFrame(hist_rows).to_csv(PROC, index=False)

    print(f"Created {OUT1} ({len(lr21)} rows)")
    print(f"Created {OUT2} ({len(lr22)} rows)")
    print("All explanatory variables are computed only from matches completed before the current match.")

if __name__ == "__main__":
    main()
